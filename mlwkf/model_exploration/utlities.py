import ray
import numpy as np
import pandas as pd
from mlwkf.evaluation_metrics import *
from mlwkf.models.bootstrapped_models import *
from mlwkf.models.standard_models import *
from mlwkf.models.ensemble_models import *

from ray.tune.sample import (function, sample_from, uniform, quniform, choice,
                             randint, lograndint, qrandint, qlograndint, randn,
                             qrandn, loguniform, qloguniform)

import copy
import torch
from pathlib import Path


def get_out_of_sample_score(training_dataset, oos_dataset, selected_features, model, scoring_functions, output_folder):

    df = pd.read_csv(training_dataset)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    label_train = df['target']
    data_train = df.drop(["target", "x", "y"], axis=1, errors='ignore')
    data_train = data_train[selected_features]

    df = pd.read_csv(oos_dataset)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    label_oos = df['target']
    data_oos = df.drop(["target", "x", "y"], axis=1, errors='ignore')
    data_oos = data_oos[selected_features]
    data_oos = data_oos.reindex(list(data_train.columns), axis=1)

    model.fit(data_train, label_train)

    model_file_path = output_folder / Path("oos_model.bin")
    model.save(model_file_path)

    label_pred = model.predict(data_oos)
    return {
        "oos_"
        + scoring_function.__name__: scoring_function(
            label_oos, label_pred, len(data_train.columns)
        )
        for scoring_function in scoring_functions
    }


def get_cross_validation_score(training_dataset, n_splits, selected_features, model, scoring_functions, output_folder):
    df = pd.read_csv(training_dataset).astype('float32')
    df = df[selected_features + ['target']]

    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    split_dataset = get_split_dataset(df, n_splits)

    results = {
        scoring_function.__name__: {"each_split_score": [], "mean_score": None}
        for scoring_function in scoring_functions
    }

    for i in range(n_splits):
        train_dataset = copy.deepcopy(split_dataset)
        test_dataset = train_dataset.pop(i)
        train_dataset = pd.concat(train_dataset)

        y_train = train_dataset['target']
        X_train = train_dataset[selected_features]

        y_test = test_dataset['target']
        X_test = test_dataset[selected_features]

        model.fit(X_train, y_train)

        model_file_path = output_folder / Path(f"{str(i)}_cv_model.bin")
        model.save(model_file_path)

        y_pred = model.predict(X_test)

        for scoring_function in scoring_functions:
            results[scoring_function.__name__]["each_split_score"].append(scoring_function(y_test.values, y_pred, len(selected_features)))

    for scoring_function in scoring_functions:
        results[scoring_function.__name__]["mean_score"] = np.mean(results[scoring_function.__name__]["each_split_score"])

    return {
        "cv_"
        + scoring_function.__name__: results[scoring_function.__name__][
            "mean_score"
        ]
        for scoring_function in scoring_functions
    }


def get_split_dataset(dataset, n_spits):
    shuffled_dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    return np.array_split(shuffled_dataset, n_spits)


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
