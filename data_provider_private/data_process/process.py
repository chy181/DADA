import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ._injection import inject_anomalies
from sklearn.preprocessing import StandardScaler
from data_provider._read_data import data_info, read_data
import time

class  Process_AnomalyDataset:
    def __init__(self,
                 datasets,
                 root_path = "/workspace/dataset/dataset/",
                 sample_save_path = "/workspace/dataset/AnomalyDatasets",
                ):
        self.datasets = datasets.split(",")
        self.root_path = root_path
        self.sample_save_path = sample_save_path
        self.norm_count = 0
        self.anorm_count = 0

    def __call__(self, win_size, step, threshold=0.1):
        for dataset in self.datasets:
            print(f"process {dataset}...")
            self.norm_save_path = f"{self.sample_save_path}/{dataset}/Norm/{win_size}_{step}"
            self.anorm_save_path = f"{self.sample_save_path}/{dataset}/Anorm/{win_size}_{step}"
            if not os.path.exists(self.norm_save_path):
                os.makedirs(self.norm_save_path)
            if not os.path.exists(self.anorm_save_path):
                os.makedirs(self.anorm_save_path) 
            filenames, train_lens, file_nums, discrete_channels = data_info(root_path=self.root_path, dataset=dataset)
            for i in tqdm(range(file_nums)):
                data_path = os.path.join(self.root_path, filenames[i])
                # 1.read data
                data = read_data(data_path)
                # 2.train
                series, label =  ( 
                    data.loc[:, data.columns != "label"].to_numpy(),
                    data.loc[:, ["label"]].to_numpy(),
                )
                scale = StandardScaler()
                series = scale.fit_transform(series) # t x c
                C = series.shape[1]
                for dim_index in range(C):
                    self._generate_sample(series[:train_lens[i], dim_index:dim_index+1], label[:train_lens[i]], dim_index=dim_index, win_size=win_size, step=step, mode=0)
                    self._generate_sample(series[train_lens[i]:, dim_index:dim_index+1], label[train_lens[i]:], dim_index=dim_index, win_size=win_size, step=step, mode=1, threshold=threshold)
            
            print("done!")
    
    def _generate_sample(self, series, series_label, dim_index, win_size, step, mode=0, threshold=0.1):
        n_samples = (len(series) - win_size) // step + 1
        threshold = win_size*threshold
        for i in range(n_samples):
            index = i * step
            sample = series[index: index + win_size]
            sample_label = series_label[index: index + win_size]
            sample = np.expand_dims(sample, axis=0)
            sample_label = np.expand_dims(sample_label, axis=0)
            data = np.concatenate((sample, sample_label), axis=0)
            if mode: # from test data
                if i==0: # init
                    n_anomalies_in_sample = sample_label.sum()
                else:
                    for t in range(1, step+1):
                        n_anomalies_in_sample = n_anomalies_in_sample - series_label[index-t] + series_label[index+win_size-t]
                if n_anomalies_in_sample > threshold:
                    np.save(f"{self.anorm_save_path}/Test_c{dim_index}_{self.anorm_count}.npy", data)
                    self.anorm_count += 1
                elif n_anomalies_in_sample==0:
                    np.save(f"{self.norm_save_path}/Test_c{dim_index}_{self.norm_count}.npy", data)
                    self.norm_count += 1
            else: # from train data
                np.save(f"{self.norm_save_path}/c{dim_index}_{self.norm_count}.npy", data)
                self.norm_count += 1


class Process:
    def __init__(self,
                 data_source = "/workspace/dataset/dataset/dataset_800M",
                 sample_save_path = "/workspace/dataset/Forecast_800M",
                 max_anomaly_ratio = 0.10,
                 rng = None,
                ):
        self.data_source = data_source
        self.sample_save_path = sample_save_path
        self.max_anomaly_ratio = max_anomaly_ratio
        self.rng = rng
        self.norm_count = 0
        self.anorm_count = 0

    def __call__(self, folder_name, win_size, step):
        data_source = os.path.join(self.data_source, folder_name)
        self.norm_save_path = f"{self.sample_save_path}/{folder_name}/Norm/{win_size}_{step}"
        self.anorm_save_path = f"{self.sample_save_path}/{folder_name}/Anorm/{win_size}_{step}"
        if not os.path.exists(self.norm_save_path):
            os.makedirs(self.norm_save_path)
        if not os.path.exists(self.anorm_save_path):
            os.makedirs(self.anorm_save_path) 
        files = os.listdir(data_source)
        for name in tqdm(files):
            data_name = os.path.join(data_source, name)
            data =  pd.read_csv(data_name)
            series = data.iloc[:, 1].values
            scale = StandardScaler()
            series, series_label = inject_anomalies(series, max_anomaly_ratio=self.max_anomaly_ratio, rng=self.rng)
            series = series.reshape(-1, 1)
            # series.dtype = np.float64
            series_label = series_label.reshape(-1, 1)
            # series_label.dtype = np.uint8
            series = scale.fit_transform(series)
            self._generate_sample(series, series_label, win_size=win_size, step=step)
        print(f"norm samples: {self.norm_count}\tanorm samples: {self.anorm_count}")


    def _generate_sample(self, series, series_label, win_size, step):
        n_samples = (len(series) - win_size) // step + 1
        for i in range(n_samples):
            index = i * step
            sample = series[index: index + win_size]
            sample_label = series_label[index: index + win_size]
            if i==0: # init
                n_anomalies_in_sample = sample_label.sum()
            else:
                for t in range(1, step+1):
                    n_anomalies_in_sample = n_anomalies_in_sample - series_label[index-t] + series_label[index+win_size-t]
                # n_anomalies_in_sample = n_anomalies_in_sample - series_label[index-step:index].sum() + series_label[index+win_size-step:index+win_size].sum()
            sample = np.expand_dims(sample, axis=0)
            sample_label = np.expand_dims(sample_label, axis=0)
            data = np.concatenate((sample, sample_label), axis=0)
            if n_anomalies_in_sample:
                np.save(f"{self.anorm_save_path}/{self.anorm_count}.npy", data)
                self.anorm_count += 1
            else:
                np.save(f"{self.norm_save_path}/{self.norm_count}.npy", data)
                self.norm_count += 1


if __name__ == "__main__":
    rng = np.random.default_rng(2024)
    np.random.seed(2024)
    process = Process(rng=rng)
    process(folder_name="test", win_size=100, step=1)


