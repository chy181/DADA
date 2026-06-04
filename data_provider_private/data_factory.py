from torch.utils.data import ConcatDataset, DataLoader
from .data_loader import TrainSegLoader, TrainSampleLoader, TrainSampleLoader_Monash
from ._batch_scheduler import BatchSchedulerSampler
from ._read_data import data_info
import os


from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    """带后台预取的数据加载器。

    预训练样本通常来自多个数据集和大量 `.npy` 文件，后台预取可以减少
    Python 数据读取对训练 step 的阻塞。
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# def PretrainDataProvider(root_path, datasets, batch_size, win_size, step, nums):
#     """
#     return DataLoader,

#     Args:
#         root_path (_type_): _description_
#         datasets (_type_): _description_
#         batch_size (_type_): _description_
#         win_size (_type_): _description_
#         step (_type_): _description_
#         nums (_type_): _description_
#     Returns:
#         concat_dataset, data_loader
#     """
#     datasets = datasets.split(",")
#     concat_dataset = []
#     for dataset in datasets:
#         print(f"loading {dataset}...", end="")
#         filenames, train_lens, file_nums, discrete_channels = data_info(root_path=root_path, dataset=dataset)
#         for i in range(file_nums):
#             data_path = os.path.join(root_path, filenames[i])
#             data_set = TrainSegLoader(data_path, train_lens[i], win_size, step, mode="pretrain", discrete_channels=discrete_channels)
#             concat_dataset.append(data_set)
#         print("done!")
#     concat_dataset = ConcatDataset(concat_dataset)
#     data_loader = DataLoaderX(
#         dataset=concat_dataset,
#         batch_size=batch_size,
#         num_workers=8,
#         drop_last=False,
#         sampler=BatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size, nums=nums),
#     )

#     return concat_dataset, data_loader


def DataProvider(root_path, datasets, batch_size, win_size, step, mode="train", percentage=0.1):
    """构造单个私有/预训练数据集的滑窗 DataLoader。

    `data_info` 根据数据集名称在 `DETECT_META.csv` 中定位文件、训练段长度和
    需要移除的离散通道。`TrainSegLoader` 再负责 read_data、标准化和滑窗。
    """
    if mode == "train":
        shuffle = True
    else: shuffle = False
    print(f"loading {datasets}({mode})...", end="")
    filenames, train_lens, file_nums, discrete_channels = data_info(root_path=root_path, dataset=datasets)
    assert file_nums == 1
    data_path = os.path.join(root_path, filenames[0])
    data_set = TrainSegLoader(data_path, train_lens[0], win_size, step, mode, percentage, discrete_channels)
    data_loader = DataLoaderX(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=8, drop_last=False)
    print("done!")
    return data_set, data_loader


def AnormDataProvider(root_path, datasets, batch_size, win_size, step, nums=-1):
    """加载预切分的异常样本窗口。

    这些样本用于异常增强或预训练阶段的异常分支，不再经过在线滑窗切分。
    """
    anorm_dataset = TrainSampleLoader(root_path, datasets, win_size, step, type="Anorm", nums=nums)
    anorm_loader = DataLoaderX(
        dataset=anorm_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=True,
    )
    return anorm_dataset, anorm_loader


def PretrainDataProvider(root_path, datasets, batch_size, win_size, step, nums=-1):
    """分别构造正常样本和异常样本的预训练 DataLoader。

    返回两个 loader，训练代码可按需要对正常重构目标和异常对抗/判别目标
    进行不同采样。
    """
    norm_dataset = TrainSampleLoader(root_path, datasets, win_size, step, type="Norm", nums=nums)
    anorm_dataset = TrainSampleLoader(root_path, datasets, win_size, step, type="Anorm", nums=nums)
    norm_loader = DataLoaderX(
        dataset=norm_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=True,
    )
    anorm_loader = DataLoaderX(
        dataset=anorm_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=True,
    )
    return norm_loader, anorm_loader


def PretrainDataProvider_Monash(root_path, datasets, batch_size, win_size, step, nums=-1):
    """构造 Monash CSV 预训练样本 DataLoader。"""
    norm_dataset = TrainSampleLoader_Monash(root_path, datasets, win_size, step, type="Norm", nums=nums)
    norm_loader = DataLoaderX(
        dataset=norm_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=True,
    )
    return norm_loader
