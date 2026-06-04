import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
from .data_loader import huaweiCompressor, huaweiLeakageLoader

warnings.filterwarnings('ignore')


def data_provider(args, root_path, datasets, batch_size, win_size=100, step=100, flag="train", percentage=1, discrete=False, continuous_cols=None, discrete_cols=None):
    """构造下游评估/微调阶段的数据集与 DataLoader。

    该入口统一处理公开评估数据和华为业务数据。公开数据通过 `DETECT_META.csv`
    定位 CSV 文件与训练段长度，再由 `TrainSegLoader` 完成长表转宽表、训练/
    测试切分、连续/离散列拆分、标准化和滑窗；业务数据走专用 loader。
    """
    if flag == "train": shuffle = True
    else: shuffle = False

    print(f"loading {datasets}({flag}) percentage: {percentage*100}% ...", end="")
    if datasets == "huaweiCompressor":
        data_set = huaweiCompressor(args, flag=flag, discrete=discrete)
    elif datasets == "huaweiLeakage":
        data_set = huaweiLeakageLoader(args, flag=flag, discrete=discrete)
    else:
        file_paths, train_lens = read_meta(root_path=root_path, dataset=datasets)
        discrete_channels = None
        # 去掉某些离散通道
        # if datasets == "MSL": discrete_channels = range(1, 55)
        # if datasets == "SMAP": discrete_channels = range(1, 25)
        # if datasets == "SWAT": discrete_channels = [2, 4, 9, 10, 11, 13, 15, 19, 20, 21, 22, 29, 30, 31, 32, 33, 42, 43,
        #                                             48, 50]
        data_set = TrainSegLoader(file_paths, train_lens, win_size, step, flag, percentage, discrete=discrete,
                                  continuous_cols=continuous_cols, discrete_cols=discrete_cols)

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=8, drop_last=False)
    print("done!")
    return data_set, data_loader

# original
# class TrainSegLoader(Dataset):
#     def __init__(self, data_path, train_length, win_size, step, flag="train", percentage=0.1, discrete=False,continuous_cols=None, discrete_cols=None):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#
#         # 1.read data
#         data = read_data(data_path)
#
#         # 2.train
#         train_data = data.iloc[:train_length, :]
#
#         # 3.test
#         test_data = data.iloc[train_length:, :]
#
#         ## 4.process
#         # if discrete_channels is not None:
#         #     train_data = np.delete(train_data, discrete_channels, axis=-1)
#         #     test_data = np.delete(test_data, discrete_channels, axis=-1)
#
#         self.discrete = discrete
#
#         test_label = test_data.loc[:, ["label"]].to_numpy()
#
#         train_label = train_data.loc[:, ["label"]].to_numpy()
#
#
#         if self.discrete:
#             if discrete_cols is None or continuous_cols is None:
#                 # ---------- 连续 / 离散列划分 ----------
#                 self.continuous_cols, self.discrete_cols, self.constant_cols = [], [], []
#                 self.discrete_nums = []
#
#                 for col in data.columns:
#                     if data[col].nunique() == 1:
#                         self.constant_cols.append(col)
#
#                     elif data[col].nunique() <= 5:
#                         self.discrete_cols.append(col)
#                         self.discrete_nums.append(data[col].nunique())
#
#                     else:
#                         self.continuous_cols.append(col)
#
#                 self.n_discrete = len(self.discrete_cols)
#                 self.n_continuous = len(self.continuous_cols)
#             else:
#                 self.discrete_cols = discrete_cols
#                 self.continuous_cols = continuous_cols
#
#             train_discrete = train_data[self.discrete_cols].apply(lambda x: pd.factorize(x)[0])
#             train_data = train_data[self.continuous_cols]
#
#             test_discrete = test_data[self.discrete_cols].apply(lambda x: pd.factorize(x)[0])
#             test_data = test_data[self.continuous_cols]
#
#         test_data = test_data.loc[:, test_data.columns != "label"].to_numpy()
#         train_data = train_data.loc[:, train_data.columns != "label"].to_numpy()
#
#
#         self.scaler = StandardScaler()
#         self.scaler.fit(train_data)
#         train_data = self.scaler.transform(train_data)
#         test_data = self.scaler.transform(test_data)
#
#         if self.discrete:
#             if flag == "init":
#                 self.init = train_data
#                 self.init_label = train_label
#                 self.init_discrete = train_discrete
#             else:
#                 train_end = int(len(train_data) * 0.8)
#                 train_start = int(train_end * (1 - percentage))
#                 self.train = train_data[train_start:train_end]
#                 self.train_label = train_label[train_start:train_end]
#                 self.train_discrete = train_discrete[train_start:train_end]
#                 self.val = train_data[train_end:]
#                 self.val_label = train_label[train_end:]
#                 self.val_discrete = train_discrete[train_end:]
#                 self.test = test_data
#                 self.test_label = test_label
#                 self.test_discrete = test_discrete
#
#         else:
#             if flag == "init":
#                 self.init = train_data
#                 self.init_label = train_label
#             else:
#                 train_end = int(len(train_data) * 0.8)
#                 train_start = int(train_end * (1 - percentage))
#
#                 self.train = train_data[train_start:train_end]
#                 self.train_label = train_label[train_start:train_end]
#
#                 self.val = train_data[train_end:]
#                 self.val_label = train_label[train_end:]
#
#                 self.test = test_data
#                 self.test_label = test_label
#
#
#     def __len__(self):
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif self.flag == "val":
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif self.flag == "test":
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         elif self.flag == "init":
#             return (self.init.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1
#
#
#     def __getitem__(self, index, eps=1):
#         index = index * self.step
#         if not self.discrete:
#             if self.flag == "train":
#                 return np.float32(self.train[index: index + self.win_size]), np.float32(
#                     self.train_label[index: index + self.win_size])
#             elif self.flag == "val":
#                 return np.float32(self.val[index: index + self.win_size]), np.float32(
#                     self.val_label[index: index + self.win_size])
#             elif self.flag == "test":
#                 return np.float32(self.test[index: index + self.win_size]), np.float32(
#                     self.test_label[index: index + self.win_size])
#             elif self.flag == "init":
#                 return np.float32(self.init[index: index + self.win_size]), np.float32(
#                     self.init_label[index: index + self.win_size])
#             else:
#                 return np.float32(self.test[
#                                 index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]), np.float32(
#                     self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])
#         else:
#             if self.flag == "train":
#                 return np.float32(self.train[index: index + self.win_size]), np.float32(self.train_discrete[index: index + self.win_size]), np.float32(
#                     self.train_label[index: index + self.win_size])
#             elif self.flag == "val":
#                 return np.float32(self.val[index: index + self.win_size]), np.float32(self.val_discrete[index: index + self.win_size]), np.float32(
#                     self.val_label[index: index + self.win_size])
#             elif self.flag == "test":
#                 return np.float32(self.test[index: index + self.win_size]), np.float32(self.test_discrete[index: index + self.win_size]), np.float32(
#                     self.test_label[index: index + self.win_size])
#             elif self.flag == "init":
#                 return np.float32(self.init[index: index + self.win_size]), np.float32(self.init_discrete[index: index + self.win_size]), np.float32(
#                     self.init_label[index: index + self.win_size])
#             else:
#                 return np.float32(self.test[
#                                 index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]), np.float32(self.test_discrete[
#                                 index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]), np.float32(
#                     self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])

# 测试集划分
class TrainSegLoader(Dataset):
    """公开评估数据的滑窗数据集。

    输入 CSV 支持 README 中描述的 long format。加载后先转换为宽表，再按
    `train_length` 切分训练段和测试段。连续变量会用训练段拟合的
    `StandardScaler` 标准化；离散变量在 `discrete=True` 时单独 factorize，
    供 STAR 状态感知模块使用。
    """

    def __init__(self, data_path, train_length, win_size, step, flag="train",
                 percentage=0.1, discrete=False, continuous_cols=None, discrete_cols=None):  # 新增：X%比例
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.test_anomaly_ratio = 0.1  # 前X%异常点比例

        # 数据格式转换阶段：
        # read_data 会把 long format 的 `date/value/cols` 转成每个变量一列的
        # wide format，并保留 label 列用于监督评估。
        data = read_data(data_path)

        # 时序段切分阶段：
        # DETECT_META.csv 中的 train_lens 定义正常训练段长度，后半段作为测试段。
        # 这里保持原始时间顺序，不做随机打乱。
        train_data = data.iloc[:train_length, :]

        test_data = data.iloc[train_length:, :]

        self.discrete = discrete

        test_label = test_data.loc[:, ["label"]].to_numpy()
        train_label = train_data.loc[:, ["label"]].to_numpy()

        if self.discrete:
            if discrete_cols is None or continuous_cols is None:
                # 连续/离散变量识别阶段：
                # 默认按列唯一值数量划分变量类型。低基数变量作为状态变量给
                # STAR 建模，高基数变量作为连续数值变量进入主干重构模型。
                self.continuous_cols, self.discrete_cols, self.constant_cols = [], [], []
                self.discrete_nums = []

                for col in data.columns:
                    if data[col].nunique() == 1:
                        self.constant_cols.append(col)
                    elif data[col].nunique() <= 5:
                        self.discrete_cols.append(col)
                        self.discrete_nums.append(data[col].nunique())
                    else:
                        self.continuous_cols.append(col)

                self.n_discrete = len(self.discrete_cols)
                self.n_continuous = len(self.continuous_cols)
            else:
                self.discrete_cols = discrete_cols
                self.continuous_cols = continuous_cols

            # 离散变量编码阶段：
            # factorize 将每个状态变量映射为从 0 开始的类别 ID。连续变量与
            # 离散变量分开返回，避免把类别 ID 当作普通连续信号做重构。
            train_discrete = train_data[self.discrete_cols].apply(lambda x: pd.factorize(x)[0])
            train_data = train_data[self.continuous_cols]

            test_discrete = test_data[self.discrete_cols].apply(lambda x: pd.factorize(x)[0])
            test_data = test_data[self.continuous_cols]

        test_data = test_data.loc[:, test_data.columns != "label"].to_numpy()
        train_data = train_data.loc[:, train_data.columns != "label"].to_numpy()

        # 连续变量标准化阶段：
        # 只用训练段拟合 scaler，再同时变换训练段和测试段，避免测试信息泄漏。
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        # 测试集缓存与顺序子集切分阶段：
        # `test_part1/test_part2` 用于少量异常监督或分段评估。切分依据异常点在
        # 原测试序列中的位置，保持完整时序顺序。
        self.test = test_data
        self.test_label = test_label
        self.test_discrete = test_discrete if self.discrete else None
        self._split_test_sequential()  # 时序切分函数
        # ================================================================================

        if self.discrete:
            if flag == "init":
                self.init = train_data
                self.init_label = train_label
                self.init_discrete = train_discrete
            else:
                train_end = int(len(train_data) * 0.8)
                train_start = int(train_end * (1 - percentage))
                self.train = train_data[train_start:train_end]
                self.train_label = train_label[train_start:train_end]
                self.train_discrete = train_discrete[train_start:train_end]
                self.val = train_data[train_end:]
                self.val_label = train_label[train_end:]
                self.val_discrete = train_discrete[train_end:]

        else:
            if flag == "init":
                self.init = train_data
                self.init_label = train_label
            else:
                train_end = int(len(train_data) * 0.8)
                train_start = int(train_end * (1 - percentage))
                self.train = train_data[train_start:train_end]
                self.train_label = train_label[train_start:train_end]
                self.val = train_data[train_end:]
                self.val_label = train_label[train_end:]

    def _split_test_sequential(self):
        """
        严格按照你的要求：
        1. 不打乱时间顺序
        2. 找到前 X% 异常点的最后一个索引
        3. 以此索引为切分点，将测试集切成两段
        4. 第一段包含 X% 异常点，第二段包含 (1-X)% 异常点
        """
        # 异常位置定位阶段：
        # 先找到测试集中所有异常点的时序索引，再取前 test_anomaly_ratio 比例
        # 异常点的最后位置作为切分点。
        labels = self.test_label.squeeze()  # 展平标签
        anomaly_indices = np.where(labels == 1)[0]  # 所有异常点索引（时序顺序）
        total_anomaly = len(anomaly_indices)

        if total_anomaly == 0:
            # 无异常点，直接均分
            split_idx = len(self.test) // 2
        else:
            # 计算前 X% 异常点的数量
            take_num = max(1, int(total_anomaly * self.test_anomaly_ratio))
            # 前 X% 异常点的最后一个位置
            last_anomaly_idx = anomaly_indices[take_num - 1]
            # 以此为切分点
            split_idx = last_anomaly_idx

        # 顺序切分阶段：
        # 不抽样、不重排，只把原测试序列按 split_idx 切成两段。
        self.test_part1 = self.test[:split_idx + 1]
        self.test_part1_label = self.test_label[:split_idx + 1]

        self.test_part2 = self.test[split_idx + 1:]
        self.test_part2_label = self.test_label[split_idx + 1:]

        # 离散特征同步切分
        if self.discrete:
            self.test_part1_discrete = self.test_discrete.iloc[:split_idx + 1].to_numpy()
            self.test_part2_discrete = self.test_discrete.iloc[split_idx + 1:].to_numpy()

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test_part1":  # 前X%异常点子集
            return (self.test_part1.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test_part2":  # 后(1-X)%异常点子集
            return (self.test_part2.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "init":
            return (self.init.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index, eps=1):
        """返回一个连续滑窗样本。

        非离散模式返回 `(x, label)`；离散模式返回 `(x_continuous,
        x_discrete, label)`。外部训练循环据此决定是否启用 STAR 状态感知路径。
        """
        index = index * self.step
        if not self.discrete:
            if self.flag == "train":
                return np.float32(self.train[index: index + self.win_size]), np.float32(
                    self.train_label[index: index + self.win_size])
            elif self.flag == "val":
                return np.float32(self.val[index: index + self.win_size]), np.float32(
                    self.val_label[index: index + self.win_size])
            elif self.flag == "test":
                return np.float32(self.test[index: index + self.win_size]), np.float32(
                    self.test_label[index: index + self.win_size])
            elif self.flag == "test_part1":
                return np.float32(self.test_part1[index: index + self.win_size]), np.float32(
                    self.test_part1_label[index: index + self.win_size])
            elif self.flag == "test_part2":
                return np.float32(self.test_part2[index: index + self.win_size]), np.float32(
                    self.test_part2_label[index: index + self.win_size])
            elif self.flag == "init":
                return np.float32(self.init[index: index + self.win_size]), np.float32(
                    self.init_label[index: index + self.win_size])
            else:
                return np.float32(self.test[
                                index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]), np.float32(
                    self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])
        else:
            if self.flag == "train":
                return np.float32(self.train[index: index + self.win_size]), np.float32(self.train_discrete[index: index + self.win_size]), np.float32(
                    self.train_label[index: index + self.win_size])
            elif self.flag == "val":
                return np.float32(self.val[index: index + self.win_size]), np.float32(self.val_discrete[index: index + self.win_size]), np.float32(
                    self.val_label[index: index + self.win_size])
            elif self.flag == "test":
                return np.float32(self.test[index: index + self.win_size]), np.float32(self.test_discrete[index: index + self.win_size]), np.float32(
                    self.test_label[index: index + self.win_size])
            elif self.flag == "test_part1":
                return np.float32(self.test_part1[index: index + self.win_size]), np.float32(self.test_part1_discrete[index: index + self.win_size]), np.float32(
                    self.test_part1_label[index: index + self.win_size])
            elif self.flag == "test_part2":
                return np.float32(self.test_part2[index: index + self.win_size]), np.float32(self.test_part2_discrete[index: index + self.win_size]), np.float32(
                    self.test_part2_label[index: index + self.win_size])
            elif self.flag == "init":
                return np.float32(self.init[index: index + self.win_size]), np.float32(self.init_discrete[index: index + self.win_size]), np.float32(
                    self.init_label[index: index + self.win_size])
            else:
                return np.float32(self.test[
                                index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]), np.float32(self.test_discrete[
                                index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]), np.float32(
                    self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])

def read_data(path: str, nrows=None) -> pd.DataFrame:
    """将异常检测 CSV 从 long format 转为模型使用的 wide format。

    输入通常包含 `date`、数值列和 `cols` 三列，其中同一变量的所有时间点
    连续堆叠。该函数根据每个变量的点数恢复为“每行一个时间点、每列一个变量”
    的宽表；如果 `cols` 中包含 label，则把最后一列重命名为 `label`。
    """
    data = pd.read_csv(path)
    label_exists = "label" in data["cols"].values
    all_points = data.shape[0]
    columns = data.columns
    # 形状推断阶段：
    # long format 中每个变量按时间顺序堆叠，因此出现次数最多的 cols 计数
    # 可作为单变量时间长度 n_points。
    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()
    is_univariate = n_points == all_points
    n_cols = all_points // n_points
    df = pd.DataFrame()
    cols_name = data["cols"].unique()
    if columns[0] == "date" and not is_univariate:
        # 多变量带时间索引：恢复 date index，并把每个 cols 分组展开为一列。
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif columns[0] != "date" and not is_univariate:
        # 多变量无显式时间索引：只做变量展开，行号即时间顺序。
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
    elif columns[0] == "date" and is_univariate:
        # 单变量带时间索引。
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    else:
        df[cols_name[0]] = data.iloc[:, 0]
    if label_exists:
        last_col_name = df.columns[-1]
        df.rename(columns={last_col_name: "label"}, inplace=True)
    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]
    return df

def read_meta(root_path, dataset):
    """从 DETECT_META.csv 中定位数据文件和训练段长度。"""
    meta_path = root_path + "/DETECT_META.csv"
    meta = pd.read_csv(meta_path)
    meta = meta.query(f'file_name.str.contains("{dataset}")', engine="python")
    file_paths = root_path + f"/data/{meta.file_name.values[0]}"
    train_lens = meta.train_lens.values[0]
    return file_paths, train_lens
