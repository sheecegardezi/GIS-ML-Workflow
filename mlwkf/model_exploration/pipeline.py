import sys
import logging
import configparser
import numpy as np
import pandas as pd
from pathlib import Path
import altair as alt
import copy
import json

from ray.tune.sample import (function, sample_from, uniform, quniform, choice,
                             randint, lograndint, qrandint, qlograndint, randn,
                             qrandn, loguniform, qloguniform)
from mlwkf.constants import NON_COVARIATES_FIELDS
from mlwkf.evaluation_metrics import mean_squared_error_scorer, mean_absolute_error_scorer, r2_scorer, rmse_scorer, adjusted_r2_scorer
from mlwkf.models.bootstrapped_models import *
from mlwkf.models.standard_models import *
from mlwkf.models.ensemble_models import *


def create_scatter_plot(y_pred, y_oos, output_scatter_plot):
    data_point = []
    label = []
    iteration = []
    for i in range(len(y_pred)):
        data_point.append(float(y_pred[i]))
        data_point.append(float(y_oos[i]))

        label.append("y_pred")
        label.append("y_oos")

        iteration.append(i)
        iteration.append(i)

    df = pd.DataFrame({'data_point': data_point, 'label': label, 'iteration':iteration}, columns=['data_point', 'label', 'iteration'])

    alt.Chart(df).mark_circle(size=30).encode(
        x='iteration',
        y='data_point',
        color='label',
        tooltip=['label', 'data_point']
    ).save(str(output_scatter_plot))


def create_output_csv(y_pred, y_true, output_csv_file, x, y):
    df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true,"x":x, "y":y}, columns=['y_pred', 'y_true','x','y'])
    df.to_csv(output_csv_file, index = False, header=True)


def create_scatter_plot_pred_vs_real(y_pred, y_true, output_scatter_plot):
    df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true}, columns=['y_pred', 'y_true'])
    alt.Chart(df).mark_circle(size=20, color="black").encode(
        x='y_pred',
        y='y_true',
        tooltip=['y_true', 'y_pred']
    ).save(str(output_scatter_plot))

# def create_predicted_geotif(model, covariates, output_predicted_geotiff_path):
#
#     print("model", model)
#     print("covariates", covariates)
#     print("output_predicted_geotiff_path", output_predicted_geotiff_path)


def get_split_dataset(dataset, n_spits):
    shuffled_dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    split_dataset = np.array_split(shuffled_dataset, n_spits)
    return split_dataset


def run_model_exploration_pipeline(config_file_path):

    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(config_file_path)

    n_splits = config.getint('ModelExploration', 'n_splits')
    model_function = eval(config.get('Model', 'model_function'))
    model_function_parameters = eval(config.get('Model', 'parameters', fallback="{}"))
    scoring_functions = eval(config.get('ModelExploration', 'scoring_functions'))

    training_dataset = Path(config.get('Intermediate', 'training_dataset', fallback=None))
    oos_dataset = Path(config.get('Intermediate', 'oos_dataset', fallback=None))
    selected_features = eval(config.get('Intermediate', 'selected_features', fallback=None))

    path_to_trained_model = Path(config.get('ModelExploration', 'path_to_trained_model', fallback=None))
    print("path_to_trained_model",path_to_trained_model)
    output_folder = Path(list(config['OutputFolder'].keys())[0])

    model = model_function(model_function_parameters)
    if path_to_trained_model.exists():
        model.load(path_to_trained_model)

    # read training dataset
    df = pd.read_csv(training_dataset)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    label_train = df['target']
    data_train = df.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')

    # read oos dataset
    df = pd.read_csv(oos_dataset)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    label_oos = df['target']
    data_oos = df.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')
    data_oos = data_oos.reindex(list(data_train.columns), axis=1)

    # train the model
    model.fit(data_train, label_train)

    # oos testing
    label_pred = model.predict(data_oos)

    oos_results = {}
    for scoring_function in scoring_functions:
        score = scoring_function(label_oos.values, label_pred, len(list(data_train.columns)))
        oos_results["oos_"+scoring_function.__name__] = score

    oos_results_path = output_folder / Path("oos_results.txt")
    with open(oos_results_path, 'w', encoding='utf-8') as f:
        json.dump(oos_results, f, ensure_ascii=False, indent=4)

    oos_results_df = pd.DataFrame({
        "label_oos": label_oos,
        "label_pred": label_pred
    })
    oos_results_pred_vs_real_path = output_folder / Path("oos_results_pred_vs_real.csv")
    oos_results_df.to_csv(oos_results_pred_vs_real_path, index=False, header=list(oos_results_df.columns.values))

    # cv testing
    df = pd.read_csv(training_dataset)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    split_dataset = get_split_dataset(df, n_splits)

    scores = {}
    for scoring_function in scoring_functions:
        scores[scoring_function.__name__] = []

    actual_labels = []
    predicted_labels = []
    for i, ith_dataset in enumerate(split_dataset):
        train_dataset = copy.deepcopy(split_dataset)
        test_dataset = train_dataset.pop(i)
        train_dataset = pd.concat(train_dataset)

        label_train = train_dataset['target']
        data_train = train_dataset.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')

        label_test = test_dataset['target']
        data_test = test_dataset.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')

        model.fit(data_train, label_train)
        label_pred = model.predict(data_test)

        actual_labels.extend(label_test.values)
        predicted_labels.extend(label_pred)

        for scoring_function in scoring_functions:
            score = scoring_function(label_test.values, label_pred, len(list(data_train.columns)))
            scores[scoring_function.__name__].append(score)

    cv_results = {}
    for scoring_function in scoring_functions:
        cv_results["cv_"+scoring_function.__name__+"_scores"] = scores[scoring_function.__name__]
        cv_results["cv_"+scoring_function.__name__+"_mean"] = np.mean(scores[scoring_function.__name__])

    cv_results_path = output_folder / Path("cv_results.txt")
    with open(cv_results_path, 'w', encoding='utf-8') as f:
        json.dump(cv_results, f, ensure_ascii=False, indent=4)

    cv_results_df = pd.DataFrame({
        "actual_labels": actual_labels,
        "predicted_labels": predicted_labels
    })
    cv_results_pred_vs_real_path = output_folder / Path("cv_results_pred_vs_real.csv")
    cv_results_df.to_csv(cv_results_pred_vs_real_path, index=False, header=list(cv_results_df.columns.values))
    

    
    # X_train = X_train.values
    # y_train_pred = model.predict(X_train)
    # output_scatter_plot = output_folder / "train_scatter_plot.html"
    # create_scatter_plot_pred_vs_real(y_train_pred, y_train, output_scatter_plot)
    #
    # train_r2_score = r2_score(y_train, y_train_pred)
    #
    # y_oos_pred = model.predict(X_oos)
    # output_scatter_plot = output_folder / "oos_scatter_plot.html"
    # create_scatter_plot_pred_vs_real(y_oos_pred, y_oos, output_scatter_plot)
    #
    # output_csv_file = output_folder / "train_values.csv"
    # create_output_csv(y_train_pred, y_train, output_csv_file, x_cord_train, y_cord_train)
    #
    # output_csv_file = output_folder / "oos_values.csv"
    # create_output_csv(y_oos_pred, y_oos, output_csv_file, x_cord_oos, y_cord_oos)
    #
    # oos_r2_score = r2_score(y_oos, y_oos_pred)
    # train_cv_mae_score, train_cv_r2_score = get_cross_validation_score(training_dataset, n_splits, selected_features, model)
    #
    # output_scatter_plot = output_folder / "train_distribution_scatter_plot.html"
    # create_scatter_plot(y_train_pred, y_train, output_scatter_plot)
    #
    # output_scatter_plot = output_folder / "oos_distribution_scatter_plot.html"
    # create_scatter_plot(y_oos_pred, y_oos, output_scatter_plot)
    #
    # print(selected_features)
    # print("train_r2_score", train_r2_score)
    # print("train_cv_mae_score", train_cv_mae_score)
    # print("train_cv_r2_score", train_cv_r2_score)
    # print("oos_r2_score", oos_r2_score)
    #
    # return train_r2_score, train_cv_mae_score, train_cv_r2_score, oos_r2_score

    # update config parameters
    config['Workflow']['ModelExploration'] = "False"

    # update config file
    with open(config_file_path, 'w') as configfile:
        config.write(configfile)
