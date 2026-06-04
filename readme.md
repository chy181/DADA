# Towards A General Time Series Anomaly Detector with Adaptive Bottlenecks And Dual Adversarial Decoders增强方式研究

## Requirements

代码在Python 3.8下运行。

```shell
pip install -r requirements.txt
```

## 数据准备

数据集文件应放置在 './dataset/evaluation_dataset/data/'下。文件结构如下：

```
./dataset
 └── evaluation_dataset
     ├── data
     │    ├── MSL.csv
     │    ├── SMAP.csv
     │    ├── PSM.csv
     │    ├── Creditcard.csv
     │    ├── GECCO.csv
     │    ├── other_dataset.csv
     │    └── ...
     └── DETECT_META.csv
```

其中，'./dataset/evaluation_dataset/data/'下的公开数据集（MSL.csv、SMAP.csv、PSM.csv、Creditcard.csv、GECCO.csv）的数据格式为长表（Long Format），加载时会通过'./data_provider/data_provider.py'中的'def read_data'先转换为宽表（Wide Format），然后划分为训练集和测试集，再输入模型。

具体地，长表格式：包含'date'、'value'、'cols'三列，'date'是整数索引或者日期索引（每个变量按时间先后排序），'value'是变量值，'cols'是变量名，每个变量先后堆叠，即先存变量A的所有数据点，再存变量B的所有数据点，最后存'label'列的所有数据点；宽表格式：包含'date'、变量A、变量B、...、'label'多列，每个变量作为一列，每行对应一个数据点。

若使用自定义数据集，需将数据集预处理为宽表格式或者长表格式，长表格式需通过'./data_provider/data_provider.py'中的'def read_data'先转换为宽表格式，宽表格式无需转换。将预处理好的数据集defined_dataset.csv放在'./dataset/evaluation_dataset/data/'下，并在'./dataset/evaluation_dataset/DETECT_META.csv'中增加defined_dataset.csv的相关信息，其中'file_name'改为'defined_dataset.csv'、'train_lens'改为训练集长度，其他保持一致即可。

## 训练和评估模型

脚本文件位于 './scripts/'下。以公开数据集SMAP为例：

### 运行：

```shell
sh ./scripts/fine-tune/stage1/SMAP/DADA.sh
```

### 多尺度功能参数：

```shell
parser.add_argument('--multi_scale', type=str, default='multi', choices=['close', 'multi'], help='scale module')
parser.add_argument('--scale_win_size', type=int, default=2500, help='window size for multi scale')
parser.add_argument('--scale_step', type=int, default=2500, help='step for multi scale')
parser.add_argument('--scales', type=int, nargs="+", default=[1, 5, 25], help='downsample scale for multi scale')
```

其中，'--multi_scale'控制是否启用多尺度功能，'close'为关闭，'multi'为启用。'--scale_win_size'是样本窗口大小。'--scale_step'是样本窗口滑动步长。'--scales'是下采样尺度。'--scale_win_size'需至少是'--scales'中最大尺度值的100倍且能被各尺度值整除。

### 残差聚类功能参数：

```shell
parser.add_argument('--prototype', type=str, default='TC_res', choices=['close', 'TC', 'TC_res'], help='prototype module')
parser.add_argument('--n_channel', type=int, default=10, help='the number of channels')
parser.add_argument('--n_cluster', type=int, default=10, help='the number of clusters')
parser.add_argument('--alpha', type=float, default=0.5, help='score weight of similarity score')
parser.add_argument('--beta', type=float, default=0.001, help='loss weight of similarity loss')
parser.add_argument('--epsilon', type=float, default=0.05, help='sinkhorn epsilon')
parser.add_argument('--temp_bern', type=float, default=0.07, help='bernoulli temperature')
```

其中，'--prototype'控制是否启用聚类原型功能，'close'为关闭，'TC'为启用原始特征聚类原型功能，'TC_res'为启用残差特征聚类原型功能。'--n_channel'是数据变量数，需根据数据设置。'--n_cluster'是原型聚类数，建议[10-100]。'--alpha'是聚类原型分数权重，建议[0.1-0.9]。'--beta'是聚类原型loss权重，建议[0.0001, 0.001, 0.01, 0.1, 1.0]。'--epsilon'和'--temp_bern‘是原型学习参数，建议不变。

### 状态感知功能参数：

```shell
parser.add_argument('--star', action='store_true')
parser.add_argument('--num_shared_experts', type=int, default=7)
parser.add_argument('--num_experts', type=int, default=10)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--rank', type=int, default=2)
parser.add_argument('--gama', type=float, default=0.5)
```

其中，'--star'控制是否启用状态感知功能。'num_shared_experts'是共享记忆个数。'num_experts'是可选择记忆个数。'K'是选择记忆个数。'rank'是低秩比率。'gama'是适配影响系数。

### 异常生成与判别功能参数：

```shell
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
```

其中，'--classification'控制是否启用异常生成与判别功能。'--use_vae'控制是否启用vae对抗生成方式。'--use_diffusion'控制是否启用扩散引导生成方式。'--weight_strategy'是重构和分类分数的加权策略，'fixed'为固定加权，'softmax'为自适应动态加权。'--alpha2'是分类分数权重，建议[0.1-0.9]。'--beta2'是分类loss权重，建议[0.0001, 0.001, 0.01, 0.1, 1.0]。'--diffusion_epochs'是扩散模型的训练轮次。'--diffusion_lr'是扩散模型的训练学习率。'--diffusion_context_ratio'是扩散模型的生成上下文。'--diffusion_guidance_scale'是真实异常样本对扩散模型的梯度引导强度。

### 二阶段技术方案模块入口：

`./二阶段技术方案.docx`中的三个二阶段增强模块与代码入口对应如下：

- 少量异常引导的有监督微调：核心实现位于`./layers/generator_classifier.py`中的`Generator_Classifier`，由`./models/mmask_model.py`中的`generator_classifier.training_step(...)`和`inference_step(...)`接入主干。
- 双域时空关联一致性检测：核心实现位于`./models/wgad/Plugin.py`，由`./models/mmask_model.py`中的`init_wgad(...)`初始化，并在`./models/scale/MSWindow.py`中通过`wgad_module(...)`和`wgad_module.inference(...)`参与训练与评分。
- 基于测试时训练的自解释多评分聚合：核心实现位于`./models/self_imp/Plugin.py`，由`./models/mmask_model.py`中的`init_self_imp(...)`初始化，并在`./models/scale/MSWindow.py`中使用重构、频域和WGAD等`score_channels`调用`self_imp_module.inference(...)`。

### 双域时空关联功能参数：

```shell
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
```

其中，'--wgad'控制是否启用双域时空关联一致性检测。'--wgad_horizon'是未来预测分支长度，需小于'--win_size'，且'--win_size - --wgad_horizon'需能被'--patch_len'整除。'--wgad_window_size'是时空图的二级窗口跨度，设为1时退化为当前patch内通道图，设为大于1时在相邻patch和通道上构建局部时空图；当前'--wgad_window_stride'固定建议为1。'--wgad_wave_scales'控制小波域尺度，'--wgad_gcn_layers'控制图卷积层数，'--wgad_dropout'和'--wgad_head_dropout'控制正则化。'--wgad_lambda_cl'、'--wgad_lambda_wavelet'和'--wgad_loss_weight'分别控制一致性分类、小波域重构和WGAD总loss权重。'--wgad_score_lambda'控制时域分数与小波域分数的融合比例。'--wgad_score_mode'选择WGAD内部异常分数组合方式。'--wgad_alpha'和'--weight_strategy_blend'控制WGAD分数与DADA主重构分数的融合，'fixed'为固定加权，'softmax'为动态加权。'--wgad_affine'和'--wgad_subtract_last'用于WGAD内部归一化增强。


### 多评分聚合功能参数：

```shell
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
```

其中，'--self_imp'控制是否启用基于测试时训练的自解释多评分聚合。该模块在推理阶段接收重构分数、频域分数以及可选WGAD分数等多路'score_channels'，通过单调校准器和原始序列先验得到点级聚合分数。'--self_imp_steps'和'--self_imp_lr'控制每个测试窗口内的自适应优化步数和学习率。'--self_imp_l1'约束通道权重稀疏性，便于突出主要评分来源；'--self_imp_tv'约束时间连续性，降低点级分数抖动；'--self_imp_huber_delta'控制Huber重构项的鲁棒性。'--self_imp_hidden_dim'和'--self_imp_raw_hidden_dim'分别设置评分校准器和原始数据编码器隐层维度。'--self_imp_raw_weight'控制原始序列先验对聚合分数的影响。'--self_imp_alpha'控制self-imp聚合分数与DADA当前主分数的融合比例，融合策略沿用'--weight_strategy_blend'。'--self_imp_seed'用于固定测试时优化初始化，便于复现实验。


测试结果保存在'./test_results/'下。
