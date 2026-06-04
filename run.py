import argparse
import os
import sys
import torch
from exp import *
import random
import numpy as np
from utils.utils import name_with_datetime, Logger
torch.cuda.empty_cache()

if __name__ == '__main__':
    # fix seed
    fix_seed = 2024
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)

    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='exp_mmask', help='task name, options:[exp_mmask]')
    parser.add_argument('--model', type=str, required=True, default='MMask', help='model name, options: [MMask]')
    parser.add_argument('--is_pretraining', type=int, required=True, default=0, help='status')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--test_untrained', action='store_true', help='if test untrained model', default=False)
    parser.add_argument('--test_zeroshot', action='store_true', default=False, help='status')
    parser.add_argument('--finetune', type=str, default='none', help='')
    parser.add_argument('--pretrain_des', type=str, default='', help='description')
    parser.add_argument('--des', type=str, default='', help='description')

    # data loader
    parser.add_argument('--data_path', type=str, default='./dataset/evaluation_dataset', help='root path of the dataset folders')
    parser.add_argument('--dataset', type=str, default='PSM', help='dataset name')
    parser.add_argument("--percentage", type=float, default=1.0, help="the percentage(*100) of train data")
    parser.add_argument("--pretrain_datasets", type=str, default="ASD,SKAB,NIPS_TS_Creditcard,UCR,KPI", help="pretrain datasets")
    parser.add_argument("--pretrain_nums", type=int, default=-1, help="the nums of pretrain data")
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # anomaly detection task
    parser.add_argument("--metrics", type=str, nargs="+", default=["best_f1", "affiliation", "auc_roc", "f1"], help="metrics")

    # model define
    parser.add_argument('--win_size', type=int, default=100, help='')
    parser.add_argument('--patch_len', type=int, default=5, help='for mmask model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='for mmask model')
    parser.add_argument('--repr_dim', type=int, default=256, help='for mmask model')
    parser.add_argument('--depth', type=int, default=5, help='for mmask model')
    parser.add_argument("--mask_mode", type=str, default='symmetry', help="for mmask model")
    parser.add_argument('--copies', type=int, default=2, help='for mmask model')
    parser.add_argument('--adp_bottleneck', action='store_true', default=False, help='for mmask model')
    parser.add_argument('--bn_dims', type=int, nargs="+", default=[8, 16, 32, 64, 128, 256], help='for mmask model')
    parser.add_argument('--k', type=int, default=1, help='')
    parser.add_argument('--abnorm_inject', action='store_true', default=False, help='for mmask model')
    parser.add_argument('--grl', action='store_true', default=False, help='for mmask model')
    parser.add_argument('--revin', action='store_true', default=False, help='for mmask model')
    parser.add_argument('--add_balance_loss', action='store_true', default=False, help='for mmask model')
    parser.add_argument('--backbone', type=str, default="dilated_conv", help='')
    parser.add_argument('--threshold_method', type=str, default="fixed", help='')
    parser.add_argument('--option', type=int, default=1, help='for model test, select function to calculate anomaly score')

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--pretrain_epochs', type=int, default=1, help='pretrain epochs')
    parser.add_argument('--pretrain_batch_size', type=int, default=32, help='batch size of pretrain input data')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--max_iters', type=int, default=1e5, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # add
    parser.add_argument('--metric', type=str, nargs="+", default="affiliation", help="metric")
    parser.add_argument('--q', type=float, nargs="+", default=[0.03], help="for SPOT")
    parser.add_argument('--t', type=float, nargs="+", default=[0.06], help="threshold found by SPOT")
    parser.add_argument('--lora', action='store_true', default=False, help='lora')
    parser.add_argument('--data_path_huawei', type=str, default='./dataset/dataset_huawei/02_huawei', help='root path of the dataset folders')
    parser.add_argument('--unfreeze_interval', type=int, default=1, help='how many epochs between unfreezing new module')
    parser.add_argument('--use_channel_processor', action='store_true', default=False, help='use channel processor')
    parser.add_argument('--channel_processor_dropout', type=float, default=0.1, help='channel processor dropout')
    parser.add_argument('--channel_processor_mode', type=str, default="conv", help="channel processor mode")
    parser.add_argument('--channel_nums', type=int, default=10, help='the number of channels')
    parser.add_argument('--use_multi_scale', action='store_true', default=False, help='use multi scale processor')
    parser.add_argument('--multi_patch_lens', type=int, nargs="+", default=[10, 20], help='multi patch len')
    parser.add_argument('--use_evaluate_SPOT', action='store_true', default=False, help='use evaluate_SPOT')
    parser.add_argument('--test_zeroshot_afterft', action='store_true', default=False, help='status')
    parser.add_argument('--prototype_abnorm', action='store_true', default=False, help='use prototype abnorm')
    parser.add_argument('--normal_proto_num', type=int, default=10, help='the number of normal prototypes')
    parser.add_argument('--abnormal_proto_num', type=int, default=10, help='the number of abnormal prototypes')
    parser.add_argument('--n_clusters', type=int, default=10, help='the number of clusters')
    parser.add_argument('--update_proto_every', type=int, default=1, help='how many epochs between updating prototypes')
    parser.add_argument('--smoothed_full_finetuning', action='store_true', default=False, help='smoothed full finetuning')
    parser.add_argument('--mix_noise', type=int, default=1, help='mix_noise')
    parser.add_argument('--noise_std', type=float, default=0.05, help='noise_std')
    parser.add_argument('--dsc_margin', type=float, default=0.5, help='dsc_margin')
    parser.add_argument('--test_notrain', action='store_true', default=False, help='status')
    parser.add_argument('--start_level', type=int, default=3, help='DWTMLEAD')
    parser.add_argument('--quantile_epsilon', type=float, default=0.05, help='DWTMLEAD')
    parser.add_argument('--n_neighbors', type=int, default=20, help='LOF')
    parser.add_argument('--contamination', type=float, default=0.1, help='LOF')
    parser.add_argument('--abnorm_synthesis', action='store_true', default=False, help='abnormal synthesis')
    parser.add_argument('--use_anomaly_library', action='store_true', help='是否使用异常库')
    parser.add_argument('--distance_weight', type=float, default=0.1, help='异常库距离权重')
    parser.add_argument('--visualize_clusters', action='store_true', help='是否可视化聚类结果')
    parser.add_argument('--max_viz_samples', type=int, default=1000, help='可视化时最大样本数量（避免内存问题）')
    parser.add_argument('--build_both_libraries', action='store_true', help='是否同时构建正常库和异常库')
    parser.add_argument('--normal_clusters', type=int, default=1, help='正常库的聚类数量')
    parser.add_argument('--anomaly_clusters', type=int, default=1, help='异常库的聚类数量')
    parser.add_argument('--num_noise_sources', type=int, default=1000, help='同时使用的异常噪声源数量')
    parser.add_argument('--min_noise_strength', type=float, default=0.05, help='最小噪声强度')
    parser.add_argument('--max_noise_strength', type=float, default=0.3, help='最大噪声强度')
    parser.add_argument('--noise_diversity', type=float, default=0.8, help='噪声多样性系数，控制随机性程度')
    parser.add_argument('--noise_ratio', type=float, default=0.5, help='噪声比例')
    parser.add_argument('--noise_level', type=float, default=0.1, help='噪声水平')
    parser.add_argument('--warmup_steps', type=int, default=20000)
    parser.add_argument('--warmup_max_ratio', type=float, default=0.2)
    # CATCH
    parser.add_argument('--inference_patch_stride', type=int, default=1)
    parser.add_argument('--inference_patch_size', type=int, default=32)
    parser.add_argument('--auxi_loss', type=str, default='MAE')
    parser.add_argument('--auxi_type', type=str, default='complex')
    parser.add_argument('--auxi_mode', type=str, default='fft')
    parser.add_argument('--auxi_lambda', type=float, default=0.005)
    parser.add_argument('--score_lambda', type=float, default=0.05)
    parser.add_argument('--module_first', type=bool, default=True)
    parser.add_argument('--mask', type=bool, default=False)
    parser.add_argument('--anomaly_ratio', type=list, default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20,
                                                               21, 22, 23, 24, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], help='anomaly ratio of dataset')

    # multi scale
    parser.add_argument('--multi_scale', type=str, default="multi", choices=['close', 'multi'], help="scale module")
    parser.add_argument('--scale_win_size', type=int, default=2500, help='window size for multi scale')
    parser.add_argument('--scale_step', type=int, default=2500, help='step for multi scale')
    parser.add_argument('--scales', type=int, nargs="+", default=[1, 5, 25], help='downsample scale for multi scale')

    # residual prototype
    parser.add_argument('--prototype', type=str, default="TC_res", choices=['close', 'TC', 'TC_res'], help="prototype module")
    parser.add_argument('--n_channel', type=int, default=10, help='the number of channels')
    parser.add_argument('--n_cluster', type=int, default=10, help='the number of clusters')
    parser.add_argument('--alpha1', type=float, default=0.5, help='score weight of similarity score')
    parser.add_argument('--beta1', type=float, default=0.001, help='loss weight of similarity loss')
    parser.add_argument('--epsilon', type=float, default=0.05, help='sinkhorn epsilon')
    parser.add_argument('--temp_bern', type=float, default=0.07, help='bernoulli temperature')

    # star
    parser.add_argument('--star', action='store_true')
    parser.add_argument('--num_shared_experts', type=int, default=7)
    parser.add_argument('--num_experts', type=int, default=10)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--gama', type=float, default=0.5)

    # anomaly generation and classification
    parser.add_argument('--classification', action='store_true', help="anomaly generation and classification module")
    parser.add_argument('--use_vae', action='store_true', help="anomaly generation by vae")
    parser.add_argument('--use_diffusion', action='store_true', help="anomaly generation by diffusion")
    parser.add_argument('--weight_strategy', type=str, default='fixed', choices=['fixed', 'softmax'], help='score weight strategy')
    parser.add_argument('--alpha2', type=float, default=0.5, help='score weight of classification score')
    parser.add_argument('--beta2', type=float, default=1.0, help='loss weight of classification loss')
    parser.add_argument('--diffusion_epochs', type=int, default=10)
    parser.add_argument('--diffusion_lr', type=float, default=1e-4)
    parser.add_argument('--diffusion_context_ratio', type=float, default=0.6)
    parser.add_argument('--diffusion_guidance_scale', type=float, default=0.5)

    # WGAD
    parser.add_argument('--wgad', action='store_true', help='enable WGAD plugin')
    parser.add_argument('--wgad_horizon', type=int, default=20, help='future length used inside WGAD plugin')
    parser.add_argument('--wgad_wave_scales', type=int, default=16)
    parser.add_argument('--wgad_gcn_layers', type=int, default=1)
    parser.add_argument('--wgad_dropout', type=float, default=0.0)
    parser.add_argument('--wgad_head_dropout', type=float, default=0.0)
    parser.add_argument('--wgad_score_lambda', type=float, default=0.1)
    parser.add_argument('--wgad_lambda_cl', type=float, default=1.0)
    parser.add_argument('--wgad_lambda_wavelet', type=float, default=0.01)
    parser.add_argument('--wgad_loss_weight', type=float, default=1.0)
    parser.add_argument('--wgad_alpha', type=float, default=0.5, help='fusion weight of WGAD score against DADA reconstruction score')
    parser.add_argument('--weight_strategy_blend', type=str, default='fixed', choices=['fixed', 'softmax'], help='score weight strategy')
    parser.add_argument('--wgad_score_mode', type=str, default='product', choices=['product', 'rec', 'pre', 'one_minus_cl', 'time_rec', 'rec_times_cl', 'rec_over_pre'])
    parser.add_argument('--wgad_window_size', type=int, default=1, help='patch-level window size for WGAD spatio-temporal graph; 1 keeps channel-only graph')
    parser.add_argument('--wgad_window_stride', type=int, default=1, help='patch-level stride for WGAD spatio-temporal graph')
    parser.add_argument('--wgad_affine', action='store_true')
    parser.add_argument('--wgad_subtract_last', action='store_true')

    # Self-interpretation
    parser.add_argument('--self_imp', action='store_true', help='enable pointwise self-interpretation plugin')
    parser.add_argument('--self_imp_alpha', type=float, default=0.5, help='fusion weight of self-imp score against DADA base score')
    parser.add_argument('--self_imp_steps', type=int, default=20, help='optimization steps used inside self-imp inference')
    parser.add_argument('--self_imp_lr', type=float, default=1e-2, help='optimizer lr used inside self-imp inference')
    parser.add_argument('--self_imp_l1', type=float, default=1e-2, help='sparsity weight used inside self-imp inference')
    parser.add_argument('--self_imp_tv', type=float, default=1e-2, help='total variation weight used inside self-imp inference')
    parser.add_argument('--self_imp_huber_delta', type=float, default=1.0, help='Huber delta for self-imp score reconstruction')
    parser.add_argument('--self_imp_hidden_dim', type=int, default=16, help='hidden dim of the monotonic calibrator in self-imp')
    parser.add_argument('--self_imp_raw_hidden_dim', type=int, default=16, help='hidden dim of the raw-data encoder in self-imp')
    parser.add_argument('--self_imp_raw_weight', type=float, default=0.1, help='raw-data prior weight used inside self-imp inference')
    parser.add_argument('--self_imp_seed', type=int, default=2024, help='random seed used inside self-imp inference')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # task
    if args.task_name == 'exp_mmask':
        Exp = Exp_MMask

    # setting and log
    if args.test_untrained: # not train, random init
        setting = "not pretrain/{}_{}/{}".format(args.model, args.des, args.dataset)
    elif args.test_notrain:
        setting = "no train/{}/{}".format(args.model, args.dataset)
    else: # train
        if args.finetune == "scratch": # scratch
            pretrain_setting = "not pretrain"
            setting = 'not pretrain/{}_{}/{}_{}'.format(args.model, args.des, args.dataset, args.percentage)
            if args.is_training: # redirect print
                os.makedirs(f'logs/not pretrain/{args.model}_{args.des}', exist_ok=True)
                sys.stdout = Logger(f"logs/{name_with_datetime(setting)}.log")
        else: # need pretrain
            pretrain_setting = '{}_{}'.format(
                args.model,
                args.pretrain_des,
            )
            setting = '{}/{}_{}_finetune_{}/{}_{}'.format(
                pretrain_setting,
                args.model,
                args.des,
                args.finetune,
                args.dataset,
                args.percentage,
            )
            if args.is_pretraining:
                os.makedirs('logs_pretrain', exist_ok=True)
                sys.stdout = Logger(f"logs_pretrain/{name_with_datetime(pretrain_setting)}.log")
            elif args.is_training:
                os.makedirs(f'logs/{pretrain_setting}/{args.model}_{args.des}_finetune_{args.finetune}', exist_ok=True)
                sys.stdout = Logger(f"logs/{name_with_datetime(setting)}.log")

    qs_dict = {
        "CICIDS": [0.001, 0.005, 0.01, 0.015, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025,0.026, 0.027, 0.0272, 0.0273, 0.0274, 0.0275, 0.276, 0.0277, 0.0278, 0.028, 0.029, 0.03, 0.031, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
            , 0.15, 0.2, 0.3, 0.4, 0.5, 1],
        "Creditcard": [0.001, 0.005, 0.01, 0.015, 0.02, 0.022, 0.023, 0.024, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
                       , 0.15, 0.2, 0.3, 0.4, 0.5, 1],
        "GECCO": [0.001, 0.005, 0.01, 0.018, 0.019, 0.020, 0.0205, 0.021, 0.022, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
                  , 0.15, 0.2, 0.3, 0.4, 0.5, 1],
        "SWAN": [0.001, 0.005, 0.01, 0.015, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.0255, 0.026, 0.027, 0.0275, 0.028, 0.029, 0.03, 0.031, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
            , 0.15, 0.2, 0.3, 0.4, 0.5, 1],
        "MSL": [0.001, 0.005, 0.01, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09, 0.10
        , 0.15, 0.2, 0.3, 0.4, 0.5, 1],
        "PSM": [0.001, 0.005, 0.01, 0.015, 0.017, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.027, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
                , 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 1],
        "SMAP": [0.001, 0.005, 0.01, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.03, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.043, 0.044, 0.045, 0.046,
                 0.047, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.2, 0.3, 0.4, 0.5, 1],
        "SMD": [0.001, 0.005, 0.01, 0.012, 0.015, 0.018, 0.019, 0.02, 0.021, 0.022, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
            , 0.15, 0.2, 0.3, 0.4, 0.5, 1],
        "SWAT": [0.001, 0.005, 0.01, 0.012, 0.015, 0.016, 0.017, 0.02, 0.023, 0.025, 0.03, 0.031, 0.032, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
            , 0.15, 0.2, 0.3, 0.4, 0.5, 1],
        "01_huawei": [0.0001, 0.001,0.005, 0.009, 0.01, 0.012, 0.015, 0.0195, 0.02, 0.0205, 0.021, 0.022, 0.023, 0.0231, 0.0232, 0.0233, 0.0234, 0.0235, 0.024,
                      0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,  0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10,
                      20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000],
        "02_huawei": [0.0001, 0.001,0.005, 0.009, 0.01, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.04,
                      0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10 , 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200,
                      250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000],
        "huaweiCompressor": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                             0.2, 0.3, 0.4, 0.5, 0.6, 1, 2, 3, 4, 5, 10],
        "huaweiLeakage": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2,
                          0.3, 0.4, 0.5, 0.6, 1, 2, 3, 4, 5, 10],
    }

    print('Args in experiment:')
    print(args)
    if args.is_pretraining: # pretrain
        exp = Exp(args)  # set experiments
        print('>>>>>>>start pretrain : {}<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(pretrain_setting))
        exp.pretrain(pretrain_setting)
        torch.cuda.empty_cache()
    elif args.is_training: # fine-tune
        for ii in range(args.itr):
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if args.finetune == "classifier":
                exp.train_classifier(setting, finetune=args.finetune, pretrain_setting=pretrain_setting,
                          pretrain_epoch=args.pretrain_epochs)
            else:
                exp.train(setting, finetune=args.finetune, pretrain_setting=pretrain_setting,
                          pretrain_epoch=args.pretrain_epochs)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if args.finetune == "classifier":
                exp.test_classifier(setting, test=1, qs=qs_dict[f"{args.dataset}"])
            else:
                if args.threshold_method == "fixed":
                    exp.test(setting, test=1)
                elif args.threshold_method == "spot":
                    exp.test_spot(setting, test=1, qs=qs_dict[f"{args.dataset}"])
            torch.cuda.empty_cache()
    elif args.test_zeroshot: # zero-shot
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(pretrain_setting))
        if args.threshold_method == "fixed":
            exp.test(pretrain_setting, test=2)
        elif args.threshold_method == "spot":
            exp.test_spot(pretrain_setting, test=2, qs=qs_dict[f"{args.dataset}"])
        torch.cuda.empty_cache()
    elif args.test_zeroshot_afterft: # zero-shot(after fine tune)
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        if args.threshold_method == "fixed":
            exp.test(setting, test=4)
        elif args.threshold_method == "spot":
            exp.test_spot(setting, test=4, qs=qs_dict[f"{args.dataset}"])
        torch.cuda.empty_cache()
    elif args.test_untrained: # random init
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        if args.threshold_method == "fixed":
            exp.test(setting, test=0)
        elif args.threshold_method == "spot":
            exp.test_spot(setting, test=0, qs=qs_dict[f"{args.dataset}"])
        torch.cuda.empty_cache()
    elif args.test_notrain: # notrain
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        if args.threshold_method == "fixed":
            exp.test(setting)
        elif args.threshold_method == "spot":
            exp.test_spot(setting, qs=qs_dict[f"{args.dataset}"])
        torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        if args.threshold_method == "fixed":
            exp.test(setting, test=1)
        elif args.threshold_method == "spot":
            exp.test_spot(setting, test=1, qs=qs_dict[f"{args.dataset}"])
        torch.cuda.empty_cache()
