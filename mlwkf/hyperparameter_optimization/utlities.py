import ray
import numpy as np
import pandas as pd
import copy
import torch
from pathlib import Path
from mlwkf.constants import NON_COVARIATES_FIELDS
from collections.abc import Iterable
from mlwkf.evaluation_metrics import mean_squared_error_scorer, mean_absolute_error_scorer, r2_scorer, adjusted_r2_scorer, rmse_scorer
from random import shuffle
import pickle5 as pickle



def get_group_cv_split_dataset(training_dataset_path, n_splits):

    df = pd.read_csv(training_dataset_path)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    grouped = df.groupby(df.groupcv_class)

    split_dataset = []
    for i in range(n_splits):
        df_new = grouped.get_group(i)
        split_dataset.append(df_new)

    return split_dataset


def get_group_cross_validation_score(training_dataset, selected_features, model, scoring_functions, n_splits):

    df = pd.read_csv(training_dataset)
    if "groupcv" not in df.columns.values:
        scores = {
            "groupcv_" + scoring_function.__name__: 0
            for scoring_function in scoring_functions
        }

        return scores

    split_dataset = get_group_cv_split_dataset(training_dataset, n_splits)

    results = {
        scoring_function.__name__: {"each_split_score": [], "mean_score": None}
        for scoring_function in scoring_functions
    }

    n_splits = len(split_dataset)
    for i in range(n_splits):
        train_dataset = copy.deepcopy(split_dataset)
        test_dataset = train_dataset.pop(i)
        train_dataset = pd.concat(train_dataset)

        y_train = train_dataset['target']
        X_train = train_dataset.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')
        X_train = X_train[selected_features]

        y_test = test_dataset['target']
        X_test = test_dataset.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')
        X_test = X_test[selected_features]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for scoring_function in scoring_functions:
            results[scoring_function.__name__]["each_split_score"].append(scoring_function(y_test.values, y_pred, len(selected_features)))

    for scoring_function in scoring_functions:
        results[scoring_function.__name__]["mean_score"] = np.mean(results[scoring_function.__name__]["each_split_score"])

    scores = {
        "groupcv_"
        + scoring_function.__name__: results[scoring_function.__name__][
            "mean_score"
        ]
        for scoring_function in scoring_functions
    }

    return scores


def get_split_dataset(dataset, n_spits):
    shuffled_dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    return np.array_split(shuffled_dataset, n_spits)


def get_cross_validation_score(training_dataset, n_splits, selected_features, model, scoring_functions):
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
        X_train = train_dataset.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')
        X_train = X_train[selected_features]

        y_test = test_dataset['target']
        X_test = test_dataset.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')
        X_test = X_test[selected_features]

        model.fit(X_train, y_train)
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



def get_out_of_sample_score(training_dataset, oos_dataset, selected_features, model, scoring_functions):
    df = pd.read_csv(training_dataset)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    label_train = df['target']
    data_train = df.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')
    data_train = data_train[selected_features]

    df = pd.read_csv(oos_dataset)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    label_oos = df['target']
    data_oos = df.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')
    data_oos = data_oos[selected_features]
    data_oos = data_oos.reindex(list(data_train.columns), axis=1)

    model.fit(data_train, label_train)

    model_file_path = Path(ray.tune.get_trial_dir()) / "model.bin"
    model.save(model_file_path)

    label_pred = model.predict(data_oos)
    return {
        "oos_"
        + scoring_function.__name__: scoring_function(
            label_oos, label_pred, len(data_train.columns)
        )
        for scoring_function in scoring_functions
    }
