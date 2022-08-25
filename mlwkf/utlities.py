import ray
import numpy as np
import pandas as pd
from mlwkf.evaluation_metrics import mean_squared_error_scorer, mean_absolute_error_scorer, r2_scorer, adjusted_r2_scorer, rmse_scorer
import copy
import torch
from pathlib import Path
from collections.abc import Iterable


def get_csv_columns(path_to_csv_file):
    return list(pd.read_csv(path_to_csv_file, nrows=1).columns)


def read_dataframe_from_csv(path_to_csv_file):
    df = pd.read_csv(path_to_csv_file)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_config_file(config, config_file_path, output_folder):
    # update config file
    with open(config_file_path, 'w') as configfile:
        config.write(configfile)

    copy_of_config_file = output_folder / Path(config_file_path).name
    with open(copy_of_config_file, 'w') as configfile:
        config.write(configfile)


def flatten(x):
    """

    :param x: list of lists:
    :return flat list:

    usage:
    flatten( [[3, 4], [[5, 6], 6]])
    output: [3, 4, 5, 6, 6]
    """
    return sum((flatten(i) for i in x), []) if isinstance(x, Iterable) else [x]





def create_chunked_target(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def get_no_of_cpus():
    return int(ray.cluster_resources()["CPU"] - 4)


def infer_trial_resources():
    '''Infer the resources_per_trial for ray from spec'''
    num_cpus = int(ray.cluster_resources()["CPU"])
    num_gpus = int(torch.cuda.device_count() if torch.cuda.is_available() else 0)
    return {'cpu': num_cpus, 'gpu': num_gpus}


def get_formated_dataframe(df):
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
