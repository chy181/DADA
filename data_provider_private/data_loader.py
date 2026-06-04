import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from ._read_data import read_data
import os


class TrainSegLoader(Dataset):
    """私有数据配置下的标准滑窗数据集。

    与公开评估 loader 类似，本类先通过 `_read_data.read_data` 把 long format
    CSV 转为宽表，再按 `train_length` 划分训练/测试。`discrete_channels`
    用于在预训练或基础模型训练时移除指定离散通道，避免把类别状态直接作为
    连续重构目标。
    """

    # test ok
    def __init__(self, data_path, train_length, win_size, step, mode="train", percentage=0.1, discrete_channels=None):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        # 长表到宽表转换阶段。
        data = read_data(data_path)
        # 训练/测试时序段切分阶段。
        train_data = data.iloc[:train_length, :]
        train_data, train_label =  (
            train_data.loc[:, train_data.columns != "label"].to_numpy(),
            train_data.loc[:, ["label"]].to_numpy(),
        )
        test_data = data.iloc[train_length:, :]
        test_data, test_label =  (
            test_data.loc[:, test_data.columns != "label"].to_numpy(),
            test_data.loc[:, ["label"]].to_numpy(),
        )
        if discrete_channels is not None:
            # 离散通道移除阶段：
            # 这里用于不启用 STAR 的预训练/基线场景，避免离散类别值进入连续
            # 标准化和重构目标。
            train_data = np.delete(train_data, discrete_channels, axis=-1)
            test_data = np.delete(test_data, discrete_channels, axis=-1)

        # 标准化阶段：
        # 只使用训练段拟合 scaler，随后变换训练、验证和测试数据。
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        if mode == "pretrain":
            self.train = train_data
            self.train_label = train_label
        elif mode == "init":
            self.init = train_data
            self.init_label = train_label
        else:
            # 训练/验证切分阶段：
            # 先取训练段后 20% 为验证集，再由 percentage 控制实际参与训练的
            # 尾部训练比例，便于少样本微调实验。
            train_end = int(len(train_data) * 0.8)
            train_start = int(train_end*(1-percentage))
            self.train = train_data[train_start:train_end]
            self.train_label = train_label[train_start:train_end]
            self.val = train_data[train_end:]
            self.val_label = train_label[train_end:]
            self.test = test_data
            self.test_label = test_label

    def __len__(self):
        if self.mode == "train" or self.mode == "pretrain":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "init":
            return (self.init.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index, eps=1):
        """按滑动窗口返回 `(window, label_window)`。"""
        index = index * self.step
        if self.mode == "train" or self.mode == "pretrain":           
            return np.float32(self.train[index: index + self.win_size]), np.float32(self.train_label[index: index + self.win_size])
        elif self.mode == "val":
            return np.float32(self.val[index: index + self.win_size]), np.float32(self.val_label[index: index + self.win_size])
        elif self.mode == "test":
            return np.float32(self.test[index: index + self.win_size]), np.float32(self.test_label[index: index + self.win_size])
        elif self.mode == "init":
            return np.float32(self.init[index: index + self.win_size]), np.float32(self.init_label[index: index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size: index// self.step * self.win_size+ self.win_size]), np.float32(self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])
        
    
class TrainSampleLoader(Dataset):
    """读取已离线切分的预训练窗口样本。

    样本目录按 `Norm/Anorm/{win_size}_{step}` 组织，每个 `.npy` 文件直接包含
    一个窗口样本及其标签。该 loader 用于跨数据集预训练，避免训练时重复做
    CSV 解析和滑窗。
    """

    def __init__(self, data_path, datasets, win_size, step, type="Norm", nums=-1):
        datasets = datasets.split(",")
        self.samples_list = []
        for dataset in datasets:
            print(f"loading {dataset}({type})...", end=" ")
            step = 50
            # 样本目录解析阶段：
            # 不同来源数据集的离线样本目录结构略有差异，这里按数据集名称
            # 选择对应根目录，再收集所有窗口文件路径。
            file_path = f"{data_path}/AnomalyDatasets_TestSets/{dataset}/{type}/{win_size}_{step}"
            if dataset=="Monash":
                step = 50
                file_path = f"{data_path}/MonashSamples/{type}/{win_size}_{step}"
            if dataset=="Forecast_800M" or dataset=="AnomalyDatasets":
                step = 50
                _path = f"{data_path}/{dataset}/"
                file_paths = os.listdir(_path)
                for file_path in file_paths:
                    file_path = os.path.join(_path, file_path)
                    file_path = f"{file_path}/{type}/{win_size}_{step}"
                    filenames = os.listdir(file_path)
                    samplenames = [os.path.join(file_path, filename) for filename in filenames]
                    self.samples_list.extend(samplenames)
                print("done!")
                continue
            filenames = os.listdir(file_path)
            samplenames = [os.path.join(file_path, filename) for filename in filenames]
            self.samples_list.extend(samplenames)
            print("done!")
        
        if nums != -1:
            nums =  min(len(self.samples_list), nums)
            self.samples_list = self.samples_list[:nums]

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):      
        """返回一个离线窗口样本。"""
        data = np.load(self.samples_list[index])
        return data[0], data[1]


class TrainSampleLoader_Monash(Dataset):
    """Monash CSV 数据的预训练样本读取器。

    Monash 文件长度和列格式不完全一致，因此读取时会先选取数值列、填补缺失，
    再统一截断或填充到全数据集最短序列长度，保证 batch 内张量形状一致。
    """

    def __init__(self, data_path,datasets, win_size, step, type="Norm", nums=-1):
        self.cache = {}
        data_path = 'dataset/monash_csv_downsmp'
        datasets = 'Monash'
        self.samples_list = []
        for dataset in datasets:
            print(f"loading {dataset}({type})...", end=" ")
            file_path = f"{data_path}"
            filenames = os.listdir(file_path)
            samplenames = [os.path.join(file_path, filename) for filename in filenames]
            self.samples_list.extend(samplenames)
            print("done!")

        if nums != -1:
            nums = min(len(self.samples_list), nums)
            self.samples_list = self.samples_list[:nums]

        # 计算整个数据集的最小序列长度
        self.min_seq_length = self._calculate_min_sequence_length()
        print(f"数据集最小序列长度: {self.min_seq_length}")

    def __len__(self):
        return len(self.samples_list)

    # def __getitem__(self, index):
    #     # data = np.load(self.samples_list[index])
    #
    #     df = read_data_Monash(self.samples_list[index])
    #     # 转换为 PyTorch 张量
    #     data = torch.tensor(df.values)
    #     return data

    def __getitem__(self, index):
        file_path = self.samples_list[index]

        # 如果已缓存，直接返回
        if file_path in self.cache:
            return self.cache[file_path]

        try:
            # 读取与数值化阶段：
            # 只保留可转成数值的列，并把缺失值填 0，防止不同 Monash 文件格式
            # 差异导致张量构造失败。
            df = pd.read_csv(file_path)

            # 预处理数据
            df = self.preprocess_data(df)

            # 转换为张量
            data_tensor = torch.tensor(df.values, dtype=torch.float32)

            # 长度对齐阶段：
            # 为了让 DataLoader 能直接组 batch，所有序列统一到数据集中最短长度。
            data_tensor = self._ensure_sequence_length(data_tensor)

            # 缓存结果
            self.cache[file_path] = data_tensor

            return data_tensor

        except Exception as e:
            print(f"加载 {file_path} 失败: {e}")

    def _calculate_min_sequence_length(self):
        """计算整个数据集中最短的序列长度"""
        min_length = float('inf')

        for file_path in self.samples_list:
            try:
                # 读取文件但不加载全部数据
                with open(file_path, 'r') as f:
                    # 使用行数作为序列长度
                    num_lines = sum(1 for _ in f)

                    # 减去标题行（如果有）
                    if num_lines > 1:  # 假设第一行是标题
                        num_lines -= 1

                    if num_lines < min_length:
                        min_length = num_lines
            except Exception as e:
                print(f"计算 {file_path} 长度失败: {e}")

        # 如果无法计算最小长度，使用默认值
        if min_length == float('inf'):
            min_length = 100  # 默认最小长度
            print(f"使用默认最小序列长度: {min_length}")

        return min_length

    def _ensure_sequence_length(self, tensor):
        """确保张量长度不超过最小序列长度"""
        current_length = tensor.shape[0]

        # 如果序列长度大于最小长度，截取前 min_seq_length 个点
        if current_length > self.min_seq_length:
            return tensor[:self.min_seq_length]

        # 如果序列长度小于最小长度，填充
        elif current_length < self.min_seq_length:
            padding = torch.zeros((self.min_seq_length - current_length, self.n_features), dtype=torch.float32)
            return torch.cat([tensor, padding], dim=0)

        # 长度正好
        return tensor

    def preprocess_data(self, df):
        """预处理数据，确保所有列都是数值类型"""
        # 只选择数值列
        numeric_df = df.select_dtypes(include=[np.number])

        # 如果数值列为空，尝试转换所有列为数值
        if numeric_df.empty:
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            numeric_df = df.select_dtypes(include=[np.number])

        # 填充缺失值
        numeric_df = numeric_df.fillna(0)

        # 确保数据类型
        return numeric_df.astype(np.float32)

def read_data_Monash(path: str, nrows=None) -> pd.DataFrame:
    """兼容 Monash long/wide 混合格式的 CSV 读取函数。"""
    data = pd.read_csv(path)

    # 检查标签是否存在（使用第二列）
    label_exists = "label" in data.iloc[:, 1].values if data.shape[1] > 1 else False

    all_points = data.shape[0]
    columns = data.columns

    # 安全计算 n_points
    if data.shape[1] >= 3:  # 确保有至少3列
        n_points = data.iloc[:, 2].value_counts().max()
    elif data.shape[1] == 2:  # 只有2列
        n_points = data.iloc[:, 1].value_counts().max()
    else:  # 只有1列
        n_points = all_points

    is_univariate = n_points == all_points
    n_cols = all_points // n_points if n_points > 0 else 1

    df = pd.DataFrame()
    cols_name = data.iloc[:, 1].unique() if data.shape[1] > 1 else [0]

    # 处理不同数据格式
    if columns[0] == "date" and not is_univariate and data.shape[1] >= 3:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 1].tolist()
            for j in range(min(n_cols, len(cols_name)))
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate and data.shape[1] >= 2:
        col_data = {
            cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 0].tolist()
            for j in range(min(n_cols, len(cols_name)))
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate and data.shape[1] >= 2:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        # 处理单列数据
        if data.shape[1] == 1:
            df[cols_name[0]] = data.iloc[:, 0]
        else:
            df[cols_name[0]] = data.iloc[:, 0]

    if label_exists and not df.empty:
        last_col_name = df.columns[-1]
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df
