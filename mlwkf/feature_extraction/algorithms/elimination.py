"""
Algorithm for feature elimination:

Repeat N times, where n is the total number of features
-For a feature in the list_of_features:
--train a model (fixed parameters) using all the features in the list_of_features except feature
--get performance score for the trained model
-remove the feature with least significance to the performance of the model
--list_of_features.pop(feature)

"""
import time
import json
import ray
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from mlwkf.feature_extraction.utlities import get_cross_validation_score, get_out_of_sample_score


def find_least_important_feature_cv(data, label, oos_dataset, model_function, n_splits, scoring_function, cpus_per_job, gpu_per_job):

    work = [get_cross_validation_score.remote(data, label, n_splits, feature_name, model_function, scoring_function) for feature_name in data.columns.values]
    return ray.get(work)

def find_least_important_feature_oos(data, label, oos_dataset, model_function, n_splits, scoring_function, cpus_per_job, gpu_per_job):

    work = [get_out_of_sample_score.remote(data, label, oos_dataset, feature_name, model_function, scoring_function) for feature_name in data.columns.values]
    return ray.get(work)


def get_lowest_scoring_feature(result_list):
    # result = [ {"score": mean_score, "feature": feature_name} ]
    feature_names, scores = [], []

    for result in result_list:
        feature_name = result["feature"]
        score = result["score"]
        feature_names.append(feature_name)
        scores.append(score)

    logging.warning(
        f"{str(feature_names[scores.index(max(scores))])} {str(max(scores))}"
    )

    return feature_names[scores.index(max(scores))]


def get_ranked_features(results):
    selected_features, feature_ranks = [] , []
    total_iterations = len(results)

    for i in range(total_iterations):
        result_list = results[total_iterations-i-1]

        feature_names, scores = [], []

        for result in result_list:
            feature_name = result["feature"]
            score = result["score"]
            feature_names.append(feature_name)
            scores.append(score)

        selected_features.append(feature_names[scores.index(max(scores))])
        feature_ranks.append(max(scores))

    return selected_features, feature_ranks


def calculate_feature_ranking_by_elimination(training_dataset, oos_dataset, model_function, n_splits, scoring_function, cpus_per_job, gpu_per_job, output_folder, use_oos):
    df = pd.read_csv(training_dataset)

    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    label_train = df['target']
    data_train = df.drop(["target", "x", "y"], axis=1, errors='ignore')

    start_time = time.time()
    ray.init(address='auto', _redis_password='5241590000000000', ignore_reinit_error=True, local_mode=False)
    iteration_counter = 0
    results = {
        iteration_counter: find_least_important_feature_oos(
            data_train,
            label_train,
            oos_dataset,
            model_function,
            n_splits,
            scoring_function,
            cpus_per_job,
            gpu_per_job,
        )
        if use_oos
        else find_least_important_feature_cv(
            data_train,
            label_train,
            oos_dataset,
            model_function,
            n_splits,
            scoring_function,
            cpus_per_job,
            gpu_per_job,
        )
    }


    lowest_feature = get_lowest_scoring_feature(results[iteration_counter])

    current_X = data_train.drop(lowest_feature, axis=1)
    current_total_feature = current_X.shape[1]

    while current_total_feature > 1:
        logging.warning(f"features remaining: {str(current_total_feature)}")
        iteration_counter = iteration_counter + 1
        if use_oos:
            results[iteration_counter] = find_least_important_feature_oos(current_X, label_train, oos_dataset, model_function, n_splits, scoring_function, cpus_per_job, gpu_per_job)
        else:
            results[iteration_counter] = find_least_important_feature_cv(current_X, label_train, oos_dataset, model_function, n_splits, scoring_function, cpus_per_job, gpu_per_job)

        lowest_feature = get_lowest_scoring_feature(results[iteration_counter])

        current_X = current_X.drop(lowest_feature, axis=1)
        current_total_feature = current_X.shape[1]

    ray.shutdown()

    output_results_file_path = output_folder / Path('feature_ranking_elimination_results.txt')
    with open(output_results_file_path, 'w') as f:
        print(json.dumps(results, sort_keys=False, indent=4), file=f)


    logging.warning(results)
    logging.warning(f"time take: {str(time.time() - start_time)}")

    selected_features, feature_ranks = get_ranked_features(results)
    return selected_features, feature_ranks
