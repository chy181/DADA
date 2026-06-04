import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features
# from data_provider.uea import subsample, interpolate_missing, Normalizer
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
               self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
   index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
   index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
   index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
   index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
   index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# v1, 训练集：80%正常；测试集：20%正常+100%异常
class huaweiCompressor(Dataset):
    def __init__(self, args, root_path='dataset/huaweiCompressor', win_size=400, step=1, flag="train", discrete=True, noraml_percentage_for_train=0.8, abnoraml_percentage_for_train=0.8):
        self.flag = flag
        self.step = step
        self.root_path=root_path
        self.win_size = win_size
        self.noraml_percentage_for_train = noraml_percentage_for_train
        self.abnoraml_percentage_for_train = abnoraml_percentage_for_train
        self.scaler = StandardScaler()
        if os.path.exists(f'{self.root_path}/processed') and os.path.exists(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy') and os.path.exists(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy'):
            train_data = np.load(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy')
            test_data = np.load(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy')
        else:
            train_data, test_data = self.process()
            if not os.path.exists(f'{self.root_path}/processed'):
                os.makedirs(f'{self.root_path}/processed')
            np.save(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy', train_data)
            np.save(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy', test_data)

        # num_samples num_points num_channels
        # 取出label
        labels = test_data[:, :, -1:]!=0
        init_labels = train_data[:, :, -1:]!=0

        init_data = train_data[:, :, :-1]
        train_data = train_data[:, :, :-1]
        test_data = test_data[:, :, :-1]

        # 删除离散数据
        # train_data = train_data[:, :, :-7]
        # test_data = test_data[:, :, :-7]
        # init_data = init_data[:, :, :-7]
        dims = train_data.shape[-1]

        # self.discrete = discrete
        # if self.discrete:
        #     if discrete_cols is None or continuous_cols is None:
        #         # ---------- 连续 / 离散列划分 ----------
        #         self.continuous_cols, self.discrete_cols, self.constant_cols = [], [], []
        #         self.discrete_nums = []
        #         for col in data.columns:
        #             if data[col].nunique() == 1:
        #                 self.constant_cols.append(col)
        #             elif data[col].nunique() <= 5:
        #                 self.discrete_cols.append(col)
        #                 self.discrete_nums.append(data[col].nunique())
        #             else:
        #                 self.continuous_cols.append(col)
        #         self.n_discrete = len(self.discrete_cols)
        #         self.n_continuous = len(self.continuous_cols)
        #     else:
        #         self.discrete_cols = discrete_cols
        #         self.continuous_cols = continuous_cols
        #     train_discrete = train_data[self.discrete_cols].apply(lambda x: pd.factorize(x)[0])
        #     train_data = train_data[self.continuous_cols]
        #     test_discrete = test_data[self.discrete_cols].apply(lambda x: pd.factorize(x)[0])
        #     test_data = test_data[self.continuous_cols]
        # test_data = test_data.loc[:, test_data.columns != "label"].to_numpy()
        # train_data = train_data.loc[:, train_data.columns != "label"].to_numpy()

        self.discrete = discrete
        if self.discrete:
            # 1. 初始化索引列表 (代替列名)
            self.continuous_cols = []
            self.discrete_cols = []
            self.constant_cols = []
            self.discrete_nums = []
            self.n_discrete = 0
            self.n_continuous = 0
            # 我们需要把数据展平来统计唯一值，或者在每个窗口上统计？
            # 这里为了模拟 pandas 的 nunique，我们把所有样本和时间步拼在一起看
            # data_reshaped shape: (16*400, 18)
            data_reshaped = train_data.reshape(-1, train_data.shape[-1])
            n_features = data_reshaped.shape[1]
            for col_idx in range(n_features):
                # 获取当前特征的所有值
                col_values = data_reshaped[:, col_idx]
                # 计算唯一值数量 (代替 data[col].nunique())
                unique_vals = np.unique(col_values)
                n_unique = len(unique_vals)
                if n_unique == 1:
                    self.constant_cols.append(col_idx)
                elif n_unique <= 5:
                    self.discrete_cols.append(col_idx)
                    self.discrete_nums.append(n_unique)
                else:
                    self.continuous_cols.append(col_idx)
            self.n_discrete = len(self.discrete_cols)
            self.n_continuous = len(self.continuous_cols)
            # 2. 数据分离 (Discrete vs Continuous)
            # 提取离散特征和连续特征
            # train_discrete shape: (16, 400, n_discrete)
            train_discrete = train_data[:, :, self.discrete_cols]
            # train_data shape: (16, 400, n_continuous)
            train_data = train_data[:, :, self.continuous_cols]
            test_discrete = test_data[:, :, self.discrete_cols]
            test_data = test_data[:, :, self.continuous_cols]


        # 标准化
        train_data = train_data.reshape(-1, self.n_continuous)
        test_data = test_data.reshape(-1, self.n_continuous)
        train_discrete = train_discrete.reshape(-1, self.n_discrete)
        test_discrete = test_discrete.reshape(-1, self.n_discrete)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data).reshape(-1, win_size, self.n_continuous)
        test_data = self.scaler.transform(test_data).reshape(-1, win_size, self.n_continuous)
        self.scaler.fit(train_discrete)
        train_discrete = self.scaler.transform(train_discrete).reshape(-1, win_size, self.n_discrete)
        test_discrete = self.scaler.transform(test_discrete).reshape(-1, win_size, self.n_discrete)


        # 分出验证集
        data_len = len(train_data)
        self.test = test_data
        self.test_discrete = test_discrete
        self.init = train_data
        self.init_descrete = train_discrete
        self.train = train_data[:(int)(data_len * 0.8)]
        self.train_discrete = train_discrete[:(int)(data_len * 0.8)]
        self.val = train_data[(int)(data_len * 0.8):]
        self.vali_descrete = train_discrete[(int)(data_len * 0.8):]
        self.test_labels = labels
        self.init_labels = init_labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def sampleing(self, data, rate):
        total_samples = data.size(0)
        num_to_sample = int(total_samples * rate)
        indices = torch.randperm(total_samples)
        sample_indices = indices[:num_to_sample]
        res_indeices = indices[num_to_sample:]
        sampled_data = data[sample_indices]
        res_data = data[res_indeices]
        return sampled_data, res_data

    def process(self):
        data_columns=["制冷剂系统1-冷凝温度","压缩机1-控制状态","压缩机1-运行状态","压缩机1-排气温度","液冷系统-室外温度","冷却液系统-电池侧回水温度","压缩机1-相电流","制冷剂系统1-压缩机吸气过热度","冷却液系统-电池侧供水温度","电子膨胀阀1-控制状态","电子膨胀阀1-运行状态","制冷剂系统1-冷凝器出口压力","制冷剂系统1-压缩机吸气压力","制冷剂系统1-压缩机排气压力","制冷剂系统1-压缩机吸气温度","制冷剂系统1-冷凝器出口温度","压缩机1-排气过热度","label"]
        samples=[]
        for filename in tqdm(os.listdir(self.root_path)):
            # 检查文件是否以 .csv 结尾
            if filename.endswith('.csv') and 'luna' not in filename:
                df = pd.read_csv(f'{self.root_path}/{filename}')
                tensor = torch.tensor(df[data_columns].values)
                length = tensor.size(0)//self.win_size * self.win_size
                tensor = tensor[:length,:]
                # 在 tensor.unfold(...) 之前添加
                if tensor.size(0) < self.win_size:
                    print(f"[Warning] 跳过当前片段，数据长度({tensor.size(0)}) < 窗口大小({self.win_size})")
                    # 选择1：如果数据太短，直接返回空或者不处理这个片段
                    continue  # 或者根据你的逻辑返回 None
                tensor = tensor.unfold(dimension=0,size=self.win_size,step=self.win_size).permute(0,2,1)
                samples.append(tensor)

            # if filename.endswith('.csv') and 'luna' not in filename:
            #     df = pd.read_csv(f'{self.root_path}/{filename}')
            #     df = df[data_columns]
            #
            #     # 降采样
            #     # print("df shape before nan:", df.shape)
            #     df = group(df, factor=5)
            #     # print("df shape after nan:", df.shape)
            #
            #     tensor = torch.tensor(df.values)
            #     length = tensor.size(0)//self.win_size * self.win_size
            #     tensor = tensor[:length,:]
            #
            #     # 检查数据长度是否足够
            #     if tensor.size(0) < self.win_size:
            #         print(f"警告: 文件 {filename} 数据长度不足 ({tensor.size(0)} < {self.win_size})，跳过")
            #         continue
            #
            #     tensor = tensor.unfold(dimension=0,size=self.win_size,step=self.win_size).permute(0,2,1)
            #     samples.append(tensor)

        all_data = torch.cat(samples,dim=0)
        abnoraml_mask = ((all_data[:,:,-1]!=0).sum(-1)!=0)
        abnoraml = all_data[abnoraml_mask]
        noraml = all_data[~abnoraml_mask]

        # 按比例组合训练集验证集
        train_noraml, test_noraml = self.sampleing(noraml,self.noraml_percentage_for_train)
        train_abnoraml, test_abnoraml = self.sampleing(abnoraml,self.abnoraml_percentage_for_train)
        # train_valid = torch.cat([train_noraml, train_abnoraml], dim=0)
        train_valid = train_noraml
        test = torch.cat([test_noraml, test_abnoraml], dim=0)
        return train_valid.numpy(), test.numpy()
        # return train_valid, test

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return self.train.shape[0]
        elif (self.flag == 'val'):
            return self.val.shape[0]
        elif (self.flag == 'test'):
            return self.test.shape[0]
        elif (self.flag == 'init'):
            return self.init.shape[0]

    def __getitem__(self, index):
        index = index * self.step
        if not self.discrete:
            if self.flag == "train":
                return np.float32(self.train[index]), np.float32(self.test_labels[index])
            elif (self.flag == 'val'):
                return np.float32(self.val[index]), np.float32(self.test_labels[index])
            elif (self.flag == 'test'):
                return np.float32(self.test[index]), np.float32(self.test_labels[index])
            elif (self.flag == 'init'):
                 return np.float32(self.init[index]), np.float32(self.init_labels[index])
        else:
            if self.flag == "train":
                return np.float32(self.train[index]), np.float32(self.train_discrete[index]), np.float32(self.test_labels[index])
            elif (self.flag == 'val'):
                return np.float32(self.val[index]), np.float32(self.vali_descrete[index]), np.float32(self.test_labels[index])
            elif (self.flag == 'test'):
                return np.float32(self.test[index]), np.float32(self.test_discrete[index]), np.float32(self.test_labels[index])
            elif (self.flag == 'init'):
                 return np.float32(self.init[index]), np.float32(self.init_descrete[index]), np.float32(self.init_labels[index])


# v2, 训练集:测试集=8:2，即训练集：80%正常+80%异常；测试集：20%正常+20%异常, train: normal+abnormal, train_norm: only normal, train_abnorm: only abnormal
# class huaweiCompressor(Dataset):
#     def __init__(self, args, root_path='dataset/huaweiCompressor', win_size=100, step=1, flag="train", noraml_percentage_for_train=0.8, abnoraml_percentage_for_train=0.8):
#         self.flag = flag
#         self.step = step
#         self.root_path=root_path
#         self.win_size = win_size
#         self.noraml_percentage_for_train = noraml_percentage_for_train
#         self.abnoraml_percentage_for_train = abnoraml_percentage_for_train
#         self.scaler = StandardScaler()
#         if os.path.exists(f'{self.root_path}/processed') and os.path.exists(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy') and os.path.exists(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy'):
#             train_data = np.load(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy')
#             test_data = np.load(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy')
#             train_data_norm = np.load(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train_norm.npy')
#             train_data_abnorm = np.load(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train_abnorm.npy')
#         else:
#             train_data, test_data, train_data_norm, train_data_abnorm = self.process()
#             if not os.path.exists(f'{self.root_path}/processed'):
#                 os.makedirs(f'{self.root_path}/processed')
#             np.save(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy', train_data)
#             np.save(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy', test_data)
#             np.save(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train_norm.npy', train_data_norm)
#             np.save(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train_abnorm.npy', train_data_abnorm)
#
#         # num_samples num_points num_channels
#
#         # 取出label
#         train_label = train_data[:, :, -1:] != 0
#         test_label = test_data[:, :, -1:] !=0
#         train_label_norm = train_data_norm[:, :, -1:] != 0
#         train_label_abnorm = train_data_abnorm[:, :, -1:] != 0
#
#         train_data = train_data[:, :, :-1]
#         test_data = test_data[:, :, :-1]
#         train_data_norm = train_data_norm[:, :, :-1]
#         train_data_abnorm = train_data_abnorm[:, :, :-1]
#
#         # 删除离散数据
#         train_data = train_data[:, :, :-7]
#         test_data = test_data[:, :, :-7]
#         train_data_norm = train_data_norm[:, :, :-7]
#         train_data_abnorm = train_data_abnorm[:, :, :-7]
#         dims = train_data.shape[-1]
#
#         # 标准化
#         train_data = train_data.reshape(-1, dims)
#         test_data = test_data.reshape(-1, dims)
#         train_data_norm = train_data_norm.reshape(-1, dims)
#         train_data_abnorm = train_data_abnorm.reshape(-1, dims)
#         self.scaler.fit(train_data)
#         train_data = self.scaler.transform(train_data).reshape(-1, win_size, dims)
#         test_data = self.scaler.transform(test_data).reshape(-1, win_size, dims)
#         train_data_norm = self.scaler.transform(train_data_norm).reshape(-1, win_size, dims)
#         train_data_abnorm = self.scaler.transform(train_data_abnorm).reshape(-1, win_size, dims)
#
#         # 分出验证集
#         if flag == "init":
#             self.init = train_data
#             self.init_label = train_label
#         else:
#             train_end = int(len(train_data) * 0.8)
#             self.train = train_data[:train_end]
#             self.train_label = train_label[:train_end]
#             self.val = train_data[train_end:]
#             self.val_label = train_label[train_end:]
#             self.test = test_data
#             self.test_label = test_label
#             print("train:", self.train.shape)
#             print("test:", self.test.shape)
#
#             train_end_norm = int(len(train_data_norm) * 0.8)
#             self.train_norm = train_data_norm[:train_end_norm]
#             self.train_label_norm = train_label_norm[:train_end_norm]
#             self.val_norm = train_data_norm[train_end_norm:]
#             self.val_label_norm = train_label_norm[train_end_norm:]
#             train_end_abnorm = int(len(train_data_abnorm) * 0.8)
#             self.train_abnorm = train_data_abnorm[:train_end_abnorm]
#             self.train_label_abnorm = train_label_abnorm[:train_end_abnorm]
#             self.val_abnorm = train_data_abnorm[train_end_abnorm:]
#             self.val_label_abnorm = train_label_abnorm[train_end_abnorm:]
#
#     def sampleing(self, data, rate):
#         total_samples = data.size(0)
#         num_to_sample = int(total_samples * rate)
#         indices = torch.randperm(total_samples)
#         sample_indices = indices[:num_to_sample]
#         res_indeices = indices[num_to_sample:]
#         sampled_data = data[sample_indices]
#         res_data = data[res_indeices]
#         return sampled_data, res_data
#
#     def process(self):
#         data_columns=["制冷剂系统1-冷凝温度","压缩机1-控制状态","压缩机1-运行状态","压缩机1-排气温度","液冷系统-室外温度","冷却液系统-电池侧回水温度","压缩机1-相电流","制冷剂系统1-压缩机吸气过热度","冷却液系统-电池侧供水温度","电子膨胀阀1-控制状态","电子膨胀阀1-运行状态","制冷剂系统1-冷凝器出口压力","制冷剂系统1-压缩机吸气压力","制冷剂系统1-压缩机排气压力","制冷剂系统1-压缩机吸气温度","制冷剂系统1-冷凝器出口温度","压缩机1-排气过热度","label"]
#         samples=[]
#         for filename in tqdm(os.listdir(self.root_path)):
#             # 检查文件是否以 .csv 结尾
#             # if filename.endswith('.csv') and 'luna' not in filename:
#             #     df = pd.read_csv(f'{self.root_path}/{filename}')
#             #     tensor = torch.tensor(df[data_columns].values)
#             #     length = tensor.size(0)//self.win_size * self.win_size
#             #     tensor = tensor[:length,:]
#             #     tensor = tensor.unfold(dimension=0,size=self.win_size,step=self.win_size).permute(0,2,1)
#             #     samples.append(tensor)
#
#             if filename.endswith('.csv') and 'luna' not in filename:
#                 df = pd.read_csv(f'{self.root_path}/{filename}')
#                 df = df[data_columns]
#                 # print("df shape before nan:", df.shape)
#                 df = group(df, factor=3)
#                 # print("df shape after nan:", df.shape)
#                 tensor = torch.tensor(df.values)
#                 length = tensor.size(0)//self.win_size * self.win_size
#                 tensor = tensor[:length,:]
#                 tensor = tensor.unfold(dimension=0,size=self.win_size,step=self.win_size).permute(0,2,1)
#                 samples.append(tensor)
#
#         all_data = torch.cat(samples,dim=0)
#         abnoraml_mask = ((all_data[:,:,-1]!=0).sum(-1)!=0)
#         abnoraml = all_data[abnoraml_mask]
#         noraml = all_data[~abnoraml_mask]
#
#         # 按比例组合训练集验证集
#         train_noraml, test_noraml = self.sampleing(noraml,self.noraml_percentage_for_train)
#         train_abnoraml, test_abnoraml = self.sampleing(abnoraml,self.abnoraml_percentage_for_train)
#         train_valid_norm = train_noraml
#         train_valid_abnorm = train_abnoraml
#         # 打乱正常和异常顺序
#         train_valid =torch.cat([train_noraml, train_abnoraml], dim=0)
#         test = torch.cat([test_noraml, test_abnoraml], dim=0)
#         train_valid, _ = self.sampleing(train_valid,1.0)
#         test, _ = self.sampleing(test,1.0)
#         return train_valid.numpy(), test.numpy(), train_valid_norm.numpy(), train_valid_abnorm.numpy()
#
#     def __len__(self):
#         """
#         Number of images in the object dataset.
#         """
#         if self.flag == "train":
#             return self.train.shape[0]
#         elif (self.flag == 'val'):
#             return self.val.shape[0]
#         elif (self.flag == 'test'):
#             return self.test.shape[0]
#         elif (self.flag == 'init'):
#             return self.init.shape[0]
#         elif self.flag == "train_norm":
#             return self.train_norm.shape[0]
#         elif (self.flag == 'val_norm'):
#             return self.val_norm.shape[0]
#         elif self.flag == "train_abnorm":
#             return self.train_abnorm.shape[0]
#         elif (self.flag == 'val_abnorm'):
#             return self.val_abnorm.shape[0]
#
#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index]), np.float32(self.train_label[index])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index]), np.float32(self.val_label[index])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index]), np.float32(self.test_label[index])
#         elif (self.flag == 'init'):
#              return np.float32(self.init[index]), np.float32(self.init_label[index])
#         elif self.flag == "train_norm":
#             return np.float32(self.train_norm[index]), np.float32(self.train_label_norm[index])
#         elif (self.flag == 'val_norm'):
#             return np.float32(self.val_norm[index]), np.float32(self.val_label_norm[index])
#         elif self.flag == "train_abnorm":
#             return np.float32(self.train_abnorm[index]), np.float32(self.train_label_abnorm[index])
#         elif (self.flag == 'val_abnorm'):
#             return np.float32(self.val_abnorm[index]), np.float32(self.val_label_abnorm[index])


# v1, 训练集：80%正常；测试集：20%正常+100%异

class huaweiLeakageLoader(Dataset):
    def __init__(self, args, root_path='dataset/huaweiLeakage', win_size=1600, step=1, flag="train", discrete=True, noraml_percentage_for_train=0.8, abnoraml_percentage_for_train=0.8):
        self.flag = flag
        self.step = step
        self.root_path=root_path
        self.win_size = win_size
        self.noraml_percentage_for_train = noraml_percentage_for_train
        self.abnoraml_percentage_for_train = abnoraml_percentage_for_train
        self.scaler = StandardScaler()
        if os.path.exists(f'{self.root_path}/processed') and os.path.exists(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy') and os.path.exists(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy'):
            train_data = np.load(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy')
            test_data = np.load(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy')
        else:
            train_data, test_data = self.process()
            if not os.path.exists(f'{self.root_path}/processed'):
               os.makedirs(f'{self.root_path}/processed')
            np.save(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy', train_data)
            np.save(f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy', test_data)

        # num_samples num_points num_channels

        # 取出label
        labels = test_data[:, :, -1:]!=0
        init_labels = train_data[:, :, -1:]!=0

        init_data = train_data[:, :, :-1]
        train_data = train_data[:, :, :-1]
        test_data = test_data[:, :, :-1]

        # 删除离散数据
        # train_data = train_data[:, :, :-7]
        # test_data = test_data[:, :, :-7]
        # init_data = init_data[:, :, :-7]
        dims = train_data.shape[-1]

        self.discrete = discrete
        if self.discrete:
            # 1. 初始化索引列表 (代替列名)
            self.continuous_cols = []
            self.discrete_cols = []
            self.constant_cols = []
            self.discrete_nums = []
            self.n_discrete = 0  # 关键：初始化 n_discrete
            self.n_continuous = 0
            # 我们需要把数据展平来统计唯一值，或者在每个窗口上统计？
            # 这里为了模拟 pandas 的 nunique，我们把所有样本和时间步拼在一起看
            # data_reshaped shape: (16*400, 18)
            data_reshaped = train_data.reshape(-1, train_data.shape[-1])
            n_features = data_reshaped.shape[1]
            for col_idx in range(n_features):
                # 获取当前特征的所有值
                col_values = data_reshaped[:, col_idx]
                # 计算唯一值数量 (代替 data[col].nunique())
                unique_vals = np.unique(col_values)
                n_unique = len(unique_vals)
                if n_unique == 1:
                    self.constant_cols.append(col_idx)
                elif n_unique <= 5:
                    self.discrete_cols.append(col_idx)
                    self.discrete_nums.append(n_unique)
                else:
                    self.continuous_cols.append(col_idx)
            self.n_discrete = len(self.discrete_cols)
            self.n_continuous = len(self.continuous_cols)
            # 2. 数据分离 (Discrete vs Continuous)
            # 提取离散特征和连续特征
            # train_discrete shape: (16, 400, n_discrete)
            train_discrete = train_data[:, :, self.discrete_cols]
            # train_data shape: (16, 400, n_continuous)
            train_data = train_data[:, :, self.continuous_cols]
            test_discrete = test_data[:, :, self.discrete_cols]
            test_data = test_data[:, :, self.continuous_cols]

        # 标准化
        train_data = train_data.reshape(-1, self.n_continuous)
        test_data = test_data.reshape(-1, self.n_continuous)
        train_discrete = train_discrete.reshape(-1, self.n_discrete)
        test_discrete = test_discrete.reshape(-1, self.n_discrete)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data).reshape(-1, win_size, self.n_continuous)
        test_data = self.scaler.transform(test_data).reshape(-1, win_size, self.n_continuous)
        self.scaler.fit(train_discrete)
        train_discrete = self.scaler.transform(train_discrete).reshape(-1, win_size, self.n_discrete)
        test_discrete = self.scaler.transform(test_discrete).reshape(-1, win_size, self.n_discrete)

        # 分出验证集
        data_len = len(train_data)
        self.test = test_data
        self.test_discrete = test_discrete
        self.init = train_data
        self.init_descrete = train_discrete
        self.train = train_data[:(int)(data_len * 0.8)]
        self.train_discrete = train_discrete[:(int)(data_len * 0.8)]
        self.val = train_data[(int)(data_len * 0.8):]
        self.vali_descrete = train_discrete[(int)(data_len * 0.8):]
        self.test_labels = labels
        self.init_labels = init_labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def sampleing(self, data, rate):
        total_samples = data.size(0)
        num_to_sample = int(total_samples * rate)
        indices = torch.randperm(total_samples)
        sample_indices = indices[:num_to_sample]
        res_indeices = indices[num_to_sample:]
        sampled_data = data[sample_indices]
        res_data = data[res_indeices]
        return sampled_data, res_data

    def process(self):
        data_columns=['二次侧系统:1-二次侧供液温度','二次侧系统:1-二次侧回液温度','二次侧系统:1-二次侧供液压力',
                      '二次侧系统:1-二次侧回液压力','二次侧系统:1-二次侧过滤器入口压力','二次侧系统:1-二次侧供液压力1',
                      '二次侧系统:1-二次侧供液压力2','二次侧系统:1-二次侧回液压力1','二次侧系统:1-二次侧回液压力2',
                      '二次侧系统:1-循环水泵出口压力','二次侧流量计:1-流量','膨胀罐:1-气腔压力',
                      'leak','tank1','tank2','steady','workstate','location']
        samples=[]
        for folder_path in ['updated_files_steady','updated_files_unsteady']:
            if 'steady' in folder_path:
                steady=1
            parent_path = f'{self.root_path}/{folder_path}'
            for filename in tqdm(os.listdir(parent_path)):
                # 检查文件是否以 .csv 结尾
                if filename.endswith('.csv') and 'luna' not in filename:
                    df = pd.read_csv(f'{self.root_path}/{folder_path}/{filename}')
                    df['steady'] = steady
                    tensor = torch.tensor(df[data_columns + ['steady'] + ['label']].values)
                    length = tensor.size(0) // self.win_size * self.win_size
                    tensor = tensor[:length, :]
                    # 在 tensor.unfold(...) 之前添加
                    if tensor.size(0) < self.win_size:
                        print(f"[Warning] 跳过当前片段，数据长度({tensor.size(0)}) < 窗口大小({self.win_size})")
                        # 选择1：如果数据太短，直接返回空或者不处理这个片段
                        continue  # 或者根据你的逻辑返回 None
                    tensor = tensor.unfold(dimension=0, size=self.win_size, step=self.win_size).permute(0, 2, 1)
                    samples.append(tensor)


                # if filename.endswith('.csv') and 'luna' not in filename:
                #     df = pd.read_csv(f'{self.root_path}/{folder_path}/{filename}')
                #     # df['steady'] = steady
                #     df = df[data_columns + ['label']]
                #
                #     # 降采样
                #     # print("df shape before nan:", df.shape)
                #     df = group(df, factor=5)
                #     # print("df shape after nan:", df.shape)
                #
                #     tensor = torch.tensor(df.values)
                #     length = tensor.size(0) // self.win_size * self.win_size
                #     tensor = tensor[:length, :]
                #
                #     # 检查数据长度是否足够
                #     if tensor.size(0) < self.win_size:
                #         print(f"警告: 文件 {filename} 数据长度不足 ({tensor.size(0)} < {self.win_size})，跳过")
                #         continue
                #
                #     tensor = tensor.unfold(dimension=0, size=self.win_size, step=self.win_size).permute(0, 2, 1)
                #     samples.append(tensor)

        all_data = torch.cat(samples,dim=0)
        abnoraml_mask = ((all_data[:,:,-1]!=0).sum(-1)!=0)
        abnoraml = all_data[abnoraml_mask]
        noraml = all_data[~abnoraml_mask]

        # 按比例组合训练集验证集
        train_noraml, test_noraml = self.sampleing(noraml,self.noraml_percentage_for_train)
        train_abnoraml, test_abnoraml = self.sampleing(abnoraml,self.abnoraml_percentage_for_train)
        # train_valid = torch.cat([train_noraml, train_abnoraml], dim=0)
        train_valid = train_noraml
        test = torch.cat([test_noraml, test_abnoraml], dim=0)
        return train_valid.numpy(), test.numpy()

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return self.train.shape[0]
        elif (self.flag == 'val'):
            return self.val.shape[0]
        elif (self.flag == 'test'):
            return self.test.shape[0]
        elif (self.flag == 'init'):
             return self.init.shape[0]

    def __getitem__(self, index):
        index = index * self.step
        if not self.discrete:
            if self.flag == "train":
                return np.float32(self.train[index]), np.float32(self.test_labels[index])
            elif (self.flag == 'val'):
                return np.float32(self.val[index]), np.float32(self.test_labels[index])
            elif (self.flag == 'test'):
                return np.float32(self.test[index]), np.float32(self.test_labels[index])
            elif (self.flag == 'init'):
                return np.float32(self.init[index]), np.float32(self.init_labels[index])
        else:
            if self.flag == "train":
                return np.float32(self.train[index]), np.float32(self.train_discrete[index]), np.float32(
                    self.test_labels[index])
            elif (self.flag == 'val'):
                return np.float32(self.val[index]), np.float32(self.vali_descrete[index]), np.float32(
                    self.test_labels[index])
            elif (self.flag == 'test'):
                return np.float32(self.test[index]), np.float32(self.test_discrete[index]), np.float32(
                    self.test_labels[index])
            elif (self.flag == 'init'):
                return np.float32(self.init[index]), np.float32(self.init_descrete[index]), np.float32(
                    self.init_labels[index])


# v2, 训练集:测试集=8:2，即训练集：80%正常+80%异常；测试集：20%正常+20%异常, train: normal+abnormal, train_norm: only normal, train_abnorm: only abnormal
# class huaweiLeakageLoader(Dataset):
#     def __init__(self, args, root_path='dataset/huaweiLeakage', win_size=100, step=1, flag="train",
#                  noraml_percentage_for_train=0.8, abnoraml_percentage_for_train=0.8):
#         self.flag = flag
#         self.step = step
#         self.root_path = root_path
#         self.win_size = win_size
#         self.noraml_percentage_for_train = noraml_percentage_for_train
#         self.abnoraml_percentage_for_train = abnoraml_percentage_for_train
#         self.scaler = StandardScaler()
#         if os.path.exists(f'{self.root_path}/processed') and os.path.exists(
#                 f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy') and os.path.exists(
#                 f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy'):
#             train_data = np.load(
#                 f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy')
#             test_data = np.load(
#                 f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy')
#             train_data_norm = np.load(
#                 f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train_norm.npy')
#             train_data_abnorm = np.load(
#                 f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train_abnorm.npy')
#         else:
#             train_data, test_data, train_data_norm, train_data_abnorm = self.process()
#             if not os.path.exists(f'{self.root_path}/processed'):
#                 os.makedirs(f'{self.root_path}/processed')
#             np.save(
#                 f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train.npy',
#                 train_data)
#             np.save(
#                 f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_test.npy',
#                 test_data)
#             np.save(
#                 f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train_norm.npy',
#                 train_data_norm)
#             np.save(
#                 f'{self.root_path}/processed/win_size_{win_size}_noraml_percentage_for_train_{noraml_percentage_for_train}_abnoraml_percentage_for_train_{abnoraml_percentage_for_train}_train_abnorm.npy',
#                 train_data_abnorm)
#
#         # num_samples num_points num_channels
#
#         # 取出label
#         train_label = train_data[:, :, -1:] != 0
#         test_label = test_data[:, :, -1:] != 0
#         train_label_norm = train_data_norm[:, :, -1:] != 0
#         train_label_abnorm = train_data_abnorm[:, :, -1:] != 0
#
#         train_data = train_data[:, :, :-1]
#         test_data = test_data[:, :, :-1]
#         train_data_norm = train_data_norm[:, :, :-1]
#         train_data_abnorm = train_data_abnorm[:, :, :-1]
#
#         # 删除离散数据
#         train_data = train_data[:, :, :-7]
#         test_data = test_data[:, :, :-7]
#         train_data_norm = train_data_norm[:, :, :-7]
#         train_data_abnorm = train_data_abnorm[:, :, :-7]
#         dims = train_data.shape[-1]
#
#         # 标准化
#         train_data = train_data.reshape(-1, dims)
#         test_data = test_data.reshape(-1, dims)
#         train_data_norm = train_data_norm.reshape(-1, dims)
#         train_data_abnorm = train_data_abnorm.reshape(-1, dims)
#         self.scaler.fit(train_data)
#         train_data = self.scaler.transform(train_data).reshape(-1, win_size, dims)
#         test_data = self.scaler.transform(test_data).reshape(-1, win_size, dims)
#         train_data_norm = self.scaler.transform(train_data_norm).reshape(-1, win_size, dims)
#         train_data_abnorm = self.scaler.transform(train_data_abnorm).reshape(-1, win_size, dims)
#
#         # 分出验证集
#         if flag == "init":
#             self.init = train_data
#             self.init_label = train_label
#         else:
#             train_end = int(len(train_data) * 0.8)
#             self.train = train_data[:train_end]
#             self.train_label = train_label[:train_end]
#             self.val = train_data[train_end:]
#             self.val_label = train_label[train_end:]
#             self.test = test_data
#             self.test_label = test_label
#             print("train:", self.train.shape)
#             print("test:", self.test.shape)
#
#             train_end_norm = int(len(train_data_norm) * 0.8)
#             self.train_norm = train_data_norm[:train_end_norm]
#             self.train_label_norm = train_label_norm[:train_end_norm]
#             self.val_norm = train_data_norm[train_end_norm:]
#             self.val_label_norm = train_label_norm[train_end_norm:]
#             train_end_abnorm = int(len(train_data_abnorm) * 0.8)
#             self.train_abnorm = train_data_abnorm[:train_end_abnorm]
#             self.train_label_abnorm = train_label_abnorm[:train_end_abnorm]
#             self.val_abnorm = train_data_abnorm[train_end_abnorm:]
#             self.val_label_abnorm = train_label_abnorm[train_end_abnorm:]
#
#     def sampleing(self, data, rate):
#         total_samples = data.size(0)
#         num_to_sample = int(total_samples * rate)
#         indices = torch.randperm(total_samples)
#         sample_indices = indices[:num_to_sample]
#         res_indeices = indices[num_to_sample:]
#         sampled_data = data[sample_indices]
#         res_data = data[res_indeices]
#         return sampled_data, res_data
#
#     def process(self):
#         data_columns = ['二次侧系统:1-二次侧供液温度', '二次侧系统:1-二次侧回液温度', '二次侧系统:1-二次侧供液压力',
#                         '二次侧系统:1-二次侧回液压力', '二次侧系统:1-二次侧过滤器入口压力',
#                         '二次侧系统:1-二次侧供液压力1', '二次侧系统:1-二次侧供液压力2', '二次侧系统:1-二次侧回液压力1',
#                         '二次侧系统:1-二次侧回液压力2', '二次侧系统:1-循环水泵出口压力', '二次侧流量计:1-流量',
#                         '膨胀罐:1-气腔压力', 'leak', 'tank1', 'tank2', 'steady', 'workstate', 'location']
#         samples = []
#         for folder_path in ['updated_files_steady', 'updated_files_unsteady']:
#             if 'steady' in folder_path:
#                 steady = 1
#             parent_path = f'{self.root_path}/{folder_path}'
#             for filename in tqdm(os.listdir(parent_path)):
#                 # 检查文件是否以 .csv 结尾
#                 # if filename.endswith('.csv') and 'luna' not in filename:
#                 #     df = pd.read_csv(f'{self.root_path}/{folder_path}/{filename}')
#                 #     df['steady'] = steady
#                 #     tensor = torch.tensor(df[data_columns + ['steady'] + ['label']].values)
#                 #     length = tensor.size(0) // self.win_size * self.win_size
#                 #     tensor = tensor[:length, :]
#                 #     tensor = tensor.unfold(dimension=0, size=self.win_size, step=self.win_size).permute(0, 2, 1)
#                 #     samples.append(tensor)
#
#                 if filename.endswith('.csv') and 'luna' not in filename:
#                     df = pd.read_csv(f'{self.root_path}/{folder_path}/{filename}')
#                     # df['steady'] = steady
#                     df = df[data_columns + ['label']]
#                     # print("df shape before nan:", df.shape)
#                     df = group(df, factor=5)
#                     # print("df shape after nan:", df.shape)
#                     tensor = torch.tensor(df.values)
#                     length = tensor.size(0) // self.win_size * self.win_size
#                     tensor = tensor[:length, :]
#                     tensor = tensor.unfold(dimension=0, size=self.win_size, step=self.win_size).permute(0, 2, 1)
#                     samples.append(tensor)
#
#         all_data = torch.cat(samples, dim=0)
#         abnoraml_mask = ((all_data[:, :, -1] != 0).sum(-1) != 0)
#         abnoraml = all_data[abnoraml_mask]
#         noraml = all_data[~abnoraml_mask]
#
#         # 按比例组合训练集验证集
#         train_noraml, test_noraml = self.sampleing(noraml, self.noraml_percentage_for_train)
#         train_abnoraml, test_abnoraml = self.sampleing(abnoraml, self.abnoraml_percentage_for_train)
#         train_valid_norm = train_noraml
#         train_valid_abnorm = train_abnoraml
#         # 打乱正常和异常顺序
#         train_valid = torch.cat([train_noraml, train_abnoraml], dim=0)
#         test = torch.cat([test_noraml, test_abnoraml], dim=0)
#         train_valid, _ = self.sampleing(train_valid, 1.0)
#         test, _ = self.sampleing(test, 1.0)
#         return train_valid.numpy(), test.numpy(), train_valid_norm.numpy(), train_valid_abnorm.numpy()
#
#     def __len__(self):
#         """
#         Number of images in the object dataset.
#         """
#         if self.flag == "train":
#             return self.train.shape[0]
#         elif (self.flag == 'val'):
#             return self.val.shape[0]
#         elif (self.flag == 'test'):
#             return self.test.shape[0]
#         elif (self.flag == 'init'):
#             return self.init.shape[0]
#         elif self.flag == "train_norm":
#             return self.train_norm.shape[0]
#         elif (self.flag == 'val_norm'):
#             return self.val_norm.shape[0]
#         elif self.flag == "train_abnorm":
#             return self.train_abnorm.shape[0]
#         elif (self.flag == 'val_abnorm'):
#             return self.val_abnorm.shape[0]
#
#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index]), np.float32(self.train_label[index])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index]), np.float32(self.val_label[index])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index]), np.float32(self.test_label[index])
#         elif (self.flag == 'init'):
#             return np.float32(self.init[index]), np.float32(self.init_label[index])
#         elif self.flag == "train_norm":
#             return np.float32(self.train_norm[index]), np.float32(self.train_label_norm[index])
#         elif (self.flag == 'val_norm'):
#             return np.float32(self.val_norm[index]), np.float32(self.val_label_norm[index])
#         elif self.flag == "train_abnorm":
#             return np.float32(self.train_abnorm[index]), np.float32(self.train_label_abnorm[index])
#         elif (self.flag == 'val_abnorm'):
#             return np.float32(self.val_abnorm[index]), np.float32(self.val_label_abnorm[index])


# 辅助函数
def get_mode(series):
    # .mode() 可能返回多个值，我们取第一个
    # 如果 series 为空，则返回一个默认值，例如 0
    return series.mode()[0] if not series.empty else 0

def group(df, factor, method="mean"):
    grouper = np.arange(len(df)) // factor
    # 1. 获取除 'label' 外的所有列名
    feature_columns = df.columns.drop('label')
    # 2. 使用字典推导式为所有特征列设置聚合函数为 'mean'
    agg_dict = {col: method for col in feature_columns}
    agg_dict['label'] = get_mode
    df_downsampled_agg = df.groupby(grouper).agg(agg_dict)
    return df_downsampled_agg

