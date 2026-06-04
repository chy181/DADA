import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm
import warnings
import os
import torch
from torch import optim
import torch.multiprocessing
import torch.nn as nn
from exp.exp_basic import Exp_Basic
from models.scale.MSWindow import Model as Energy
from utils.utils import EarlyStopping, adjust_learning_rate, ForeverDataIterator
from utils.metrics import evaluate, evaluate_SPOT, evaluate_tab
from layers.diffusion_real_anomaly_cond_fn import extract_anomaly_prototypes_from_test_loader, build_patch_level_anomaly_cond_fn

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings('ignore')


class Exp_MMask(Exp_Basic):
    def __init__(self, args):
        super(Exp_MMask, self).__init__(args)

    def _build_model(self):
        # model config
        Model = self.model_dict[self.args.model]
        model = Model(
            win_size=self.args.win_size,
            patch_len=self.args.patch_len,
            mask_mode=self.args.mask_mode,
            hidden_dim=self.args.hidden_dim,
            repr_dim=self.args.repr_dim,
            depth=self.args.depth,
            adp_bottleneck=self.args.adp_bottleneck,
            bottleneck_dims=self.args.bn_dims,
            k=self.args.k,
            revin=self.args.revin,
            backbone=self.args.backbone,
            max_iters=self.args.max_iters,
            prototype=self.args.prototype,
            n_channel=self.args.n_channel,
            n_cluster=self.args.n_cluster,
            device=self.device,
            epsilon=self.args.epsilon,
            temp_bern=self.args.temp_bern,
            classification=self.args.classification,
            use_vae=self.args.use_vae,
            use_diffusion=self.args.use_diffusion,
        ).float()
        # self.awl = AutomaticWeightedLoss(2)
        model = Energy(self.args, model).to(self.device)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # swa
        self.swa_model = optim.swa_utils.AveragedModel(model)
        self.swa_start = 5
        return model

    def set_trainable_modules(self, epoch):
        num_to_unfreeze = (epoch // self.args.unfreeze_interval) + 1
        for idx, module_name in enumerate(self.module_unfreeze_order):
            module = getattr(self.model, module_name, None)
            if module is not None:
                requires_grad = idx < num_to_unfreeze
                for param in module.parameters():
                    param.requires_grad = requires_grad

    def _select_optimizer(self, type):
        # optim
        if type == "all" or type == "scratch":  # unfreeze all
            model_optim = optim.AdamW(
                self.model.parameters(), lr=self.args.learning_rate)
        elif type == "stage1":
            if self.args.prototype == "close": # full fine-tune
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                for param in self.model.parameters():
                    param.requires_grad = False
            if self.args.prototype in ["TC", "TC_res"]:
                for param in self.model.backbone.Cluster_assigner.parameters():
                    param.requires_grad = True
            if self.args.prototype in ["TC_res"]:
                for param in self.model.backbone.graph_builder.parameters():
                    param.requires_grad = True
            if getattr(self.args, "wgad", False):
                for param in self.model.backbone.wgad_module.parameters():
                    param.requires_grad = True
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            model_optim = optim.AdamW(
                params, lr=self.args.learning_rate)
        elif type == "stage2":
            if self.args.prototype == "close": # full fine-tune
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                for param in self.model.parameters():
                    param.requires_grad = False
            if self.args.prototype in ["TC", "TC_res"]:
                for param in self.model.backbone.Cluster_assigner.parameters():
                    param.requires_grad = True
            if self.args.prototype in ["TC_res"]:
                for param in self.model.backbone.graph_builder.parameters():
                    param.requires_grad = True
            if self.args.classification:
                for param in self.model.backbone.generator_classifier.parameters():
                    param.requires_grad = True
            if getattr(self.args, "wgad", False):
                for param in self.model.backbone.wgad_module.parameters():
                    param.requires_grad = True
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            model_optim = optim.AdamW(
                params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.args.loss == "MAE":
            criterion = nn.L1Loss()
        elif self.args.loss == "BCE":
            criterion = nn.BCELoss()
        return criterion

    def _cal_loss(self, batch_x, output, criterion, label=None):
        if label is not None:
            batch_x = torch.masked_select(batch_x, label)
            output = torch.masked_select(output, label)
        loss = criterion(batch_x, output)
        return loss

    def _load_pretrain_model(self, type, pretrain_setting, pretrain_epoch):
        """
        load pretrained model, and return optim based on type

        Args:
            type (str): finetune type or scratch
            pretrain_setting (str): _description_
            pretrain_epoch (int):

        Returns:
            model_optim: _description_
        """
        # load model
        if type != "scratch":  # pretrained
            print(f'fine tune: {type}. loading pretrain model...', end='')
            pretrain_model_path = os.path.join(
                self.args.checkpoints, pretrain_setting, f"pretrain_checkpoint_{pretrain_epoch}.pth")
            if type in ["stage1", "stage2"]:
                pretrain_state_dict = torch.load(pretrain_model_path, map_location=self.device)
                current_state_dict = self.model.backbone.state_dict()

                matched_state_dict = {}
                skipped_keys = []
                for key, param in pretrain_state_dict.items():
                    normalized_key = key
                    if normalized_key.startswith("module."):
                        normalized_key = normalized_key[len("module."):]
                    if normalized_key.startswith("backbone."):
                        normalized_key = normalized_key[len("backbone."):]

                    if normalized_key in current_state_dict:
                        if current_state_dict[normalized_key].shape == param.shape:
                            matched_state_dict[normalized_key] = param
                        else:
                            print(
                                f"跳过不匹配的参数: {key} -> {normalized_key} | 预训练形状: {param.shape} | 当前形状: {current_state_dict[normalized_key].shape}")
                            skipped_keys.append(key)
                    else:
                        print(f"跳过不存在的参数: {key}")
                        skipped_keys.append(key)

                self.model.backbone.load_state_dict(matched_state_dict, strict=False)
            else:
                self.model.backbone.load_state_dict(torch.load(pretrain_model_path))
            print(pretrain_model_path, "\tdone!")
        else:  # unpretrained
            print(f'scratch')

    def _load_test_model(self, test, setting):
        """
        Args:
            test (int):
            setting (_type_): _description_

        Returns:
            folder_path: _description_
        """
        option = self.args.option
        # save result path
        if self.args.mask_mode != "nomask":
            copies = self.args.copies
        else:
            copies = 1
        folder_path = f'./test_results/option_{option}/{self.args.mask_mode}_{copies}/' + setting + f'/'
        if test == 0:
            print('testing: untrained')
        if test == 1:
            print('testing: fine tune')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))
        elif test == 2:
            print('testing: zero shot')
            pretrain_model_path = os.path.join(self.args.checkpoints, setting,
                                               f"pretrain_checkpoint_{self.args.pretrain_epochs}.pth")
            print(pretrain_model_path)
            self.model.load_state_dict(torch.load(pretrain_model_path))
            folder_path = folder_path + f'{self.args.model}_{self.args.pretrain_des}_zero_shot_epoch{self.args.pretrain_epochs}/{self.args.dataset}/'
        elif test == 3:
            print('testing: zero shot')
            pretrain_model_path = os.path.join(self.args.checkpoints, setting,
                                               f"pretrain_checkpoint_{self.args.pretrain_epochs}_swa.pth")
            pretrained_dict = torch.load(pretrain_model_path)
            model_dict = self.model.state_dict()
            for key in pretrained_dict:
                if key != "n_averaged":
                    name = key.replace('module.', '')
                    model_dict[name] = pretrained_dict[key]
            self.model.load_state_dict(model_dict)
            folder_path = folder_path + f'{self.args.model}_{self.args.pretrain_des}_zero_shot_swa_epoch{self.args.pretrain_epochs}/{self.args.dataset}/'
        elif test == 4:
            print('testing: zero shot(after fine tune)')
            pretrain_model_path = os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')
            print(pretrain_model_path)
            self.model.load_state_dict(torch.load(pretrain_model_path))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path

    def _cal_anomaly_score(self, batch_x, output, anomaly_criterion):
        '''
        input:
            output(tensor): [copies, b, win_size, dim],
            batch_x(tensor): [b, win_size, dim],

        '''
        if self.args.option == 1:
            output_mean = torch.mean(output, dim=0)  # b x t x c
            score = anomaly_criterion(output_mean, batch_x)
            score = torch.mean(score, dim=-1)  # b x t
        elif self.args.option == 2:
            score = torch.var(output, dim=0)  # b x t x c
            score = torch.mean(score, dim=-1)  # b x t
        elif self.args.option == 3:
            output_mean = torch.mean(output, dim=0)  # b x t x c
            score1 = anomaly_criterion(output_mean, batch_x)
            score2 = torch.var(output, dim=0)  # b x t x c
            score = score1 + score2
            score = torch.mean(score, dim=-1)  # b x t
        return score

    def pretrain(self, setting):
        # data
        train_loader_norm, train_loader_anorm = self._get_pretrain_data(step=50)
        if self.args.abnorm_inject: abnorm_iter = ForeverDataIterator(train_loader_anorm)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)
        train_steps = len(train_loader_norm)
        print("Norm sample num: ", train_steps * self.args.pretrain_batch_size)
        print("Anorm sample num: ", len(train_loader_anorm) * self.args.pretrain_batch_size)
        # setting
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        abnorm_threshold = None
        decay = 0.99
        iter_abnorm = 0
        train_loss_abnorm = 0
        # training
        for epoch in range(self.args.pretrain_epochs):
            iter_norm = 0
            train_loss_norm = 0
            train_balance_loss = 0
            self.model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader_norm):
                model_optim.zero_grad()
                # 1.norm
                batch_x = batch_x.float().to(self.device)
                output, repr, balance_loss = self.model(batch_x)
                loss = self._cal_loss(batch_x, output, criterion)
                train_loss_norm += loss.item()
                iter_norm += 1
                # 2.abnorm
                if self.args.abnorm_inject:
                    abnorm_data, abnorm_label = next(abnorm_iter)
                    abnorm_data = abnorm_data.float().to(self.device)
                    abnorm_label = abnorm_label.bool().to(self.device)
                    output_anorm, repr_anorm, balance_loss_ = self.model(abnorm_data, grl=self.args.grl)
                    balance_loss = (balance_loss + balance_loss_)
                    abnorm_loss = self._cal_loss(abnorm_data, output_anorm, criterion, label=abnorm_label)
                    if self.args.grl:
                        if abnorm_threshold is None: abnorm_threshold = abnorm_loss.item()
                        if abnorm_loss.item() <= abnorm_threshold + 0.1:
                            abnorm_threshold = decay * abnorm_threshold + (1 - decay) * abnorm_loss.item()
                            train_loss_abnorm += abnorm_loss.item()
                            iter_abnorm += 1
                            loss = loss + abnorm_loss
                            # loss = self.awl(loss, abnorm_loss)
                    else:
                        train_loss_abnorm += abnorm_loss.item()
                        iter_abnorm += 1
                        loss = loss - abnorm_loss

                if self.args.add_balance_loss:
                    train_balance_loss += balance_loss.item()
                    loss = loss + balance_loss

                if (i + 1) % 100 == 0:
                    if self.args.abnorm_inject:
                        if self.args.add_balance_loss:
                            print(
                                f"\titers: {i + 1}, epoch: {epoch + 1} | loss_norm: {train_loss_norm / iter_norm:.4f}, loss_abnorm_average: {train_loss_abnorm / iter_abnorm:.4f}, loss_balance: {train_balance_loss / iter_norm:.4f}")
                        else:
                            print(
                                f"\titers: {i + 1}, epoch: {epoch + 1} | loss_norm: {train_loss_norm / iter_norm:.4f}, loss_abnorm_average: {train_loss_abnorm / iter_abnorm:.4f}")
                    else:
                        if self.args.add_balance_loss:
                            print(
                                f"\titers: {i + 1}, epoch: {epoch + 1} | loss_norm: {train_loss_norm / iter_norm:.4f}, loss_balance: {train_balance_loss / iter_norm:.4f}")
                        else:
                            print(
                                f"\titers: {i + 1}, epoch: {epoch + 1} | loss_norm: {train_loss_norm / iter_norm:.4f}")

                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss_norm / iter_norm:.4f}")
            if epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                torch.save(self.swa_model.state_dict(), path + "/" + f"pretrain_checkpoint_{epoch + 1}_swa.pth")
            else:
                torch.save(self.model.state_dict(), path + "/" + f"pretrain_checkpoint_{epoch + 1}.pth")
                adjust_learning_rate(model_optim, epoch + 1, self.args)

    def _train_diffusion_ts(self, train_loader):
        diffusion = self.model.backbone.generator_classifier.diffusion_ts
        diffusion.to(self.device)

        for p in diffusion.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=self.args.diffusion_lr)
        diffusion.train()

        for ep in range(self.args.diffusion_epochs):
            total = 0.0
            n = 0
            for batch_x, discrete, batch_y in train_loader:
                batch_x = batch_x.float().to(self.device)
                # batch_x: (B, win_size, C)
                optimizer.zero_grad()
                loss = diffusion(batch_x)
                loss.backward()
                optimizer.step()
                total += loss.item()
                n += 1
            print(f"[Diffusion] epoch {ep + 1}/{self.args.diffusion_epochs} loss={total / max(n, 1):.6f}")

    def train(self, setting, finetune="scratch", pretrain_setting="not pretrain", pretrain_epoch=1):
        # data
        train_data, train_loader = self._get_data(flag='train', discrete=True)
        vali_data, vali_loader = self._get_data(flag='val', discrete=True)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)

        if self.args.use_diffusion:
            train_data_diffusion_ts, train_loader_diffusion_ts = self._get_data_diffusion_ts(flag='train', discrete=True)
            test_data_diffusion_ts, test_loader_diffusion_ts = self._get_data_diffusion_ts(flag='test', discrete=True)
            self._train_diffusion_ts(train_loader_diffusion_ts)
            gen = self.model.backbone.generator_classifier
            gen.set_patch_embed_meanc_fn(self.model.backbone.patch_embed_meanc)
            classifier_fn = lambda patch_meanc: gen.classifier(patch_meanc)
            anomaly_proto, raw_windows = extract_anomaly_prototypes_from_test_loader(
                test_loader=test_loader_diffusion_ts,
                patch_embed_meanc_fn=self.model.backbone.patch_embed_meanc,
                window_size=100,
                top_k=50,
                anomaly_window_usage_ratio=0.1,  # x%异常窗口
                device="cuda"
            )
            cond_fn = build_patch_level_anomaly_cond_fn(
                patch_embed_meanc_fn=self.model.backbone.patch_embed_meanc,
                patch_classifier_fn=classifier_fn,
                use_cls_gradient=True,  # 开启分类器引导
                use_proto_gradient=True,  # 开启原型引导
                anomaly_proto=anomaly_proto,
                guidance_scale=0.5,
            )
            gen.set_cond_fn(cond_fn)

        if self.args.star:
            n_discrete = train_data.n_discrete
            n_continuous = train_data.n_continuous
            discrete_nums = train_data.discrete_nums
            self.model.backbone.init_star(self.args.star, self.args.num_experts, self.args.num_shared_experts, self.args.K, self.args.rank, self.args.gama, n_continuous, n_discrete, discrete_nums)
            self.model.to(self.device)

        print("Train length: ", len(train_data))
        train_steps = len(train_loader)
        print("Sample num: ", train_steps * self.args.batch_size)
        # self.model = Energy(self.args, self.model).to(self.device)
        print('loading pre-train model')
        self._load_pretrain_model(type=finetune, pretrain_setting=pretrain_setting, pretrain_epoch=pretrain_epoch)
        model_optim = self._select_optimizer(type=finetune)
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")
        total_params = sum(
            p.numel() for p in self.model.parameters()
        )
        print(f"Total parameters: {total_params}")
        total_params = sum(
            p.numel() for p in self.model.backbone.parameters()
        )
        print(f"Total backbone parameters: {total_params}")

        # training
        for epoch in range(self.args.train_epochs):
            iter_norm = 0
            train_loss = 0
            train_loss_rec = 0
            train_loss_s = 0
            train_loss_gc = 0
            train_loss_generator = 0
            train_loss_classifier = 0
            self.model.train()
            for i, (batch_x, discrete, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                # 1.norm
                batch_x = batch_x.float().to(self.device)
                discrete = discrete.to(self.device)

                loss, loss_rec, loss_s, loss_gc, loss_generator, loss_classifier = self.model.train_function(batch_x,discrete)

                train_loss += loss.item()
                train_loss_rec += loss_rec.item()
                train_loss_s += loss_s.item()
                train_loss_gc += loss_gc.item()
                train_loss_generator += loss_generator.item()
                train_loss_classifier += loss_classifier.item()
                iter_norm += 1

                loss.backward()
                model_optim.step()

            vali_loss, vali_loss_rec, vali_loss_s, vali_loss_gc, vali_loss_generator, vali_loss_classifier = self.vali(vali_loader, criterion)
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss / iter_norm:.4f} Vali Loss: {vali_loss:.4f} "
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            if epoch % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

    def vali(self, vali_loader, criterion):
        loss_list = []
        loss_rec_list = []
        loss_s_list = []
        loss_gc_list = []
        loss_generator_list = []
        loss_classifier_list = []
        self.model.eval()

        for i, (batch_x, discrete, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            discrete = discrete.to(self.device)
            with torch.enable_grad():
                batch_x.requires_grad_(True)
                discrete.requires_grad_(False)
                loss, loss_rec, loss_s, loss_gc, loss_generator, loss_classifier = self.model.train_function(batch_x,
                                                                                                             discrete)
                loss_list.append(loss.detach().item())
                loss_rec_list.append(loss_rec.item())
                loss_s_list.append(loss_s.detach().item())
                loss_gc_list.append(loss_gc.detach().item())
                loss_generator_list.append(loss_generator.detach().item())
                loss_classifier_list.append(loss_classifier.detach().item())

        loss = np.average(loss_list)
        loss_rec = np.average(loss_rec_list)
        loss_s = np.average(loss_s_list)
        loss_gc = np.average(loss_gc_list)
        loss_generator = np.average(loss_generator_list)
        loss_classifier = np.average(loss_classifier_list)
        self.model.train()
        return loss, loss_rec, loss_s, loss_gc, loss_generator, loss_classifier

    def test_spot(self, setting, test, qs=None):
        print("copies: ", self.args.copies)
        test_data, test_loader = self._get_data(flag='test', discrete=True)
        init_data, init_loader = self._get_data(flag='init', discrete=True)
        folder_path = self._load_test_model(test=test, setting=setting)
        self.model.eval()

        # cal anomaly_socres
        test_scores = []
        init_scores = []
        test_labels = []
        with (torch.no_grad()):
            for i, (batch_x, discrete, batch_y) in enumerate(init_loader):
                batch_x = batch_x.float().to(self.device)
                score = self.model.inference(batch_x, discrete)
                init_scores.append(score)
            for i, (batch_x, discrete, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                score = self.model.inference(batch_x, discrete)
                test_scores.append(score)
                test_labels.append(batch_y)
        init_scores = np.concatenate(init_scores, axis=0).reshape(-1)
        init_scores = np.array(init_scores)
        test_scores = np.concatenate(test_scores, axis=0).reshape(-1)
        test_scores = np.array(test_scores)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        # evaluate_SPOT(gt, init_scores, test_scores, metrics=self.args.metrics, save_path=folder_path, qs=qs,
        #               verbose=True)
        from ts_ad_evaluation import Evaluator
        evaluator = Evaluator(gt, test_scores, folder_path)
        evaluator.evaluate(metrics=self.args.metric, affiliation=self.args.t)

        # tab
        print("copies: ", self.args.copies)
        train_data, train_loader = self._get_data(flag='train', discrete=True)
        test_data, test_loader = self._get_data(flag='test', discrete=True)
        init_data, init_loader = self._get_data(flag='init', discrete=True)
        folder_path = f'./test_results/option_{self.args.option}/{self.args.mask_mode}_{self.args.copies}/' + setting + f'/'
        self.model.eval()
        scores = dict(train=[], test=[], init=[])
        labels = []
        loads = dict(train=train_loader, test=test_loader, init=init_loader)
        with torch.no_grad():
            for key in loads.keys():
                print(f'Inference in {key} dataset')
                total = len(loads[key].dataset) // loads[key].batch_size
                for i, (batch_x, discrete, batch_y) in tqdm(enumerate(loads[key]), total=total):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = (batch_y != 0).float()  # 转成0/1 二分类
                    score = self.model.inference(batch_x, discrete)
                    scores[key].append(score)
                    if key == 'test':
                        labels.append(batch_y)
        for key in scores.keys():
            scores[key] = np.array(np.concatenate(scores[key], axis=0).reshape(-1))
        labels = np.concatenate(labels, axis=0).reshape(-1)
        gt = np.array(labels).astype(int)
        pd.DataFrame(dict(rec=scores['test'], label=gt)).to_csv('energy.csv')
        combined_energy = np.concatenate((scores['train'], scores['test']), axis=0)
        preds = {}
        for ratio in self.args.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (scores['test'] > threshold).astype(int)
        evaluate_tab(gt, scores['test'], metrics=self.args.metrics, save_path=folder_path, verbose=True, preds=preds)

    def test(self, setting, test):
        print("copies: ", self.args.copies)
        train, train_loader = self._get_data(flag='train', step=self.args.win_size, discrete=True)
        test_data, test_loader = self._get_data(flag='test', step=self.args.win_size)
        folder_path = self._load_test_model(test=test, setting=setting)
        anomaly_criterion = nn.MSELoss(reduce=False)
        self.model.eval()
        # cal anomaly_socres
        test_scores = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                score = self.model.inference(batch_x)
                test_scores.append(score)
                test_labels.append(batch_y)
        test_scores = np.concatenate(test_scores, axis=0).reshape(-1)
        test_scores = np.array(test_scores)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        evaluate(gt, test_scores, metrics=self.args.metrics, save_path=folder_path, verbose=True)
        from ts_ad_evaluation import Evaluator
        evaluator = Evaluator(gt, test_scores, folder_path)
        evaluator.evaluate(metrics=self.args.metric, affiliation=self.args.t)
