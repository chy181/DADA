import pandas as pd

dataset_name = {
    "ASD": "Multi",
    "Exathlon_61-32": "Multi",
    "Exathlon_63-64": "Multi",
    "Exathlon_64-63": "Multi",
    "KPI": "tsb_uad_IOPS",
    "SKAB": "Multi",
    "GHL": "tsb_uad_GHL",
    "Genesis": "Multi",
    "SWAT": "Multi",
    "SMAP": "Multi",
    "Calit2": "Multi",
    "OPPORTUNITY": "tsb_uad_OPPORTUNITY",
    "NYC": "Multi",
    "GAIA_CPoint": "GAIA-Companion_Data-changepoint_data",
    "GAIA_Concept": "GAIA-Companion_Data-concept_drift_data",
    "GAIA_Linear": "GAIA-Companion_Data-linear_data",
    "GAIA_LSignal": "GAIA-Companion_Data-low_signal-to-noise_ratio_data",
    "GAIA_PStationary": "GAIA-Companion_Data-partially_stationary_data",
    "GAIA_Periodic": "GAIA-Companion_Data-periodic_data",
    "GAIA_Staircase": "GAIA-Companion_Data-staircase_data",
    "MGAB": "tsb_uad_MGAB",
    "ECG": "tsb_uad_ECG",
    "MITDB": "tsb_uad_MITDB",
    "SVDB": "tsb_uad_SVDB",
    "YAHOO": "tsb_uad_YAHOO",
    "UCR": "tsb_uad_KDD21",
    "PSM": "Multi",
    "SMD": "Multi",
    "PUMP": "Multi",
    "MSL": "Multi",
    "daphnet_S01R02": "Multi",
    "daphnet_S02R01": "Multi",
    "daphnet_S03R01": "Multi",
    "daphnet_S03R02": "Multi",
    "daphnet_S07R01": "Multi",
    "daphnet_S07R02": "Multi",
    "daphnet_S08R01": "Multi",
    "CICIDS": "Multi",
    "NIPS_TS_GECCO": "Multi",
    "NIPS_TS_Creditcard": "Multi",
    "NIPS_TS_SWAN": "Multi",
}
file_name_re = {
    "ASD": "ASD_dataset",
    "Exathlon_61-32": "Exathlon_4_1_100000_61-32",
    "Exathlon_63-64": "Exathlon_5_1_100000_63-64",
    "Exathlon_64-63": "Exathlon_5_1_100000_64-63",
    "KPI": "KPI",
    "SKAB": "SKAB",
    "GHL": "",
    "Genesis": "Genesis",
    "SWAT": "swat",
    "SMAP": "SMAP",
    "Calit2": "CalIt2",
    "OPPORTUNITY": "",
    "NYC": "nyc_taxi",
    "GAIA_CPoint": "",
    "GAIA_Concept": "",
    "GAIA_Linear": "",
    "GAIA_LSignal": "",
    "GAIA_PStationary": "",
    "GAIA_Periodic": "",
    "GAIA_Staircase": "",
    "MGAB": "",
    "ECG": "",
    "MITDB": "",
    "SVDB": "",
    "YAHOO": "",
    "UCR": "",
    "PSM": "PSM",
    "SMD": "SMD",
    "PUMP": "PUMP",
    "MSL": "MSL",
    "daphnet_S01R02": "daphnet_S01R02",
    "daphnet_S02R01": "daphnet_S02R01",
    "daphnet_S03R01": "daphnet_S03R01",
    "daphnet_S03R02": "daphnet_S03R02",
    "daphnet_S07R01": "daphnet_S07R01",
    "daphnet_S07R02": "daphnet_S07R02",
    "daphnet_S08R01": "daphnet_S08R01",
    "CICIDS": "web_attack",
    "NIPS_TS_GECCO": "water_quality",
    "NIPS_TS_Creditcard": "creditcard",
    "NIPS_TS_SWAN": "swan_sf",
}
discrete_channels_dict = {
    "ASD": None,
    "Exathlon_61-32": None,
    "Exathlon_63-64": None,
    "Exathlon_64-63": None,
    "KPI": None,
    "SKAB": None,
    "GHL": None,
    "Genesis": None, # remove
    "SWAT": None,
    "SMAP": range(1, 25),
    "Calit2": None, # remove
    "OPPORTUNITY": None,
    "NYC": None,
    "GAIA_CPoint": None, # remove
    "GAIA_Concept": None,
    "GAIA_Linear": None,
    "GAIA_LSignal": None,
    "GAIA_PStationary": None,
    "GAIA_Periodic": None,
    "GAIA_Staircase": None,
    "MGAB": None,
    "ECG": None,
    "MITDB": None,
    "SVDB": None,
    "YAHOO": None,
    "UCR": None,
    "PSM": None,
    "SMD": None,
    "PUMP": None,
    "MSL": range(1, 55),
    "daphnet_S01R02": None,
    "daphnet_S02R01": None,
    "daphnet_S03R01": None,
    "daphnet_S03R02": None,
    "daphnet_S07R01": None,
    "daphnet_S07R02": None,
    "daphnet_S08R01": None,
    "CICIDS": None,
    "NIPS_TS_GECCO": None,
    "NIPS_TS_Creditcard": None,
    "NIPS_TS_SWAN": None,
}

def read_data(path: str, nrows=None) -> pd.DataFrame:
    """将预训练/私有数据 CSV 从 long format 恢复为 wide format。

    文件通常由 `value/cols` 或 `date/value/cols` 组成，同一变量的所有时间点
    连续堆叠。该函数根据每个变量出现次数推断单条序列长度，并把不同变量
    展开为列；如果存在 label，则统一重命名为 `label`。
    """
    data = pd.read_csv(path)
    label_exists = "label" in data["cols"].values
    all_points = data.shape[0]
    columns = data.columns
    # 序列形状推断阶段：
    # 每个变量在 long format 中占据连续的 n_points 行，因此通过 cols 频数
    # 恢复变量数和每个变量的时间长度。
    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()
    is_univariate = n_points == all_points
    n_cols = all_points // n_points
    df = pd.DataFrame()
    cols_name = data["cols"].unique()
    if columns[0] == "date" and not is_univariate:
        # 多变量带时间列：恢复时间索引并按变量展开。
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif columns[0] != "date" and not is_univariate:
        # 多变量无时间列：直接按变量展开，行顺序即时间顺序。
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
    elif columns[0] == "date" and is_univariate:
        # 单变量带时间列。
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

def data_info(root_path, dataset):
    """根据数据集别名读取元信息。

    返回文件名列表、训练段长度、文件数量，以及需要从连续重构输入中移除的
    离散通道索引。该函数是 `data_factory.DataProvider` 的元数据入口。
    """
    meta_path = root_path + "/DETECT_META.csv"
    meta = pd.read_csv(meta_path)
    meta = meta.query(f'dataset_name.str.contains("{dataset_name[dataset]}") & file_name.str.contains("{file_name_re[dataset]}")', engine="python")
    file_names = meta.file_name.values
    train_lens = meta.train_lens.values
    file_nums = len(file_names)
    discrete_channels = discrete_channels_dict[dataset]
    return file_names, train_lens, file_nums, discrete_channels
