"""
Algorithm for feature elimination:

Repeat N times, where n is the total number of features
-For a feature in the list_of_features:
--train a model (fixed parameters) using all the features in the list_of_features except feature
--get performance score for the trained model
-remove the feature with least significance to the performance of the model
--list_of_features.pop(feature)

"""
import copy
import time
import json
import ray
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from mlwkf.constants import NON_COVARIATES_FIELDS
import pickle5 as pickle



def get_split_dataset(dataset, n_spits):
    shuffled_dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    split_dataset = np.array_split(shuffled_dataset, n_spits)
    return split_dataset


@ray.remote
def get_model_score(i, path_to_split_dataset, model_function, scoring_function, model_function_parameters):
    model = model_function(model_function_parameters)
    logging.warning("cv iteration: "+str(i))

    with open(path_to_split_dataset, 'rb') as handle:
        split_dataset = pickle.load(handle)

    test_dataset = split_dataset[i]
    train_dataset = pd.concat(split_dataset[:i] + split_dataset[i+1:])

    y_train = train_dataset['target']
    del train_dataset['target']
    X_train = train_dataset

    y_test = test_dataset['target']
    del test_dataset['target']
    X_test = test_dataset

    model.fit(data=X_train, label=y_train)
    y_pred = model.predict(X_test)

    del train_dataset
    return scoring_function(y_test.values, y_pred, len(list(test_dataset.columns)))


def find_least_important_feature_cv(training_dataset_path, current_features, model_function, n_splits, scoring_function, cpus_per_job, gpu_per_job, output_folder, model_function_parameters):

    intermediate_current_features = copy.deepcopy(current_features)
    intermediate_current_features.append("target")
    df = pd.read_csv(training_dataset_path, usecols=intermediate_current_features)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    shuffled_dataset = df.sample(frac=1, random_state=1).reset_index(drop=True)
    split_dataset = np.array_split(shuffled_dataset, n_splits)

    path_to_split_dataset = output_folder/ Path("split_dataset.pkl")
    with open(path_to_split_dataset, 'wb') as handle:
        pickle.dump(split_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del df
    del split_dataset

    result_ids = []
    feature_list = []
    for feature_name in current_features:
        fields_to_use = copy.deepcopy(intermediate_current_features)
        if feature_name in fields_to_use:
            fields_to_use.remove(feature_name)

        for i in range(n_splits):
            result_ids.append(get_model_score.options(num_cpus=cpus_per_job, num_gpus=gpu_per_job).remote(i, path_to_split_dataset, model_function, scoring_function, model_function_parameters))
            feature_list.append(feature_name)

    feature_scores = ray.get(result_ids)
    results_dic = {}

    for i, score in enumerate(feature_scores):
        if feature_list[i] not in results_dic:
            results_dic[feature_list[i]] = score
        else:
            results_dic[feature_list[i]] = results_dic[feature_list[i]] + score

    results = []
    for feature in results_dic:
        results_dic[feature] = results_dic[feature] / n_splits
        results.append({"score": results_dic[feature], "feature": feature})

    return results


def get_lowest_scoring_feature(result_list):
    # result = [ {"score": mean_score, "feature": feature_name} ]
    feature_names, scores = [], []

    for result in result_list:
        feature_name = result["feature"]
        score = result["score"]
        feature_names.append(feature_name)
        scores.append(score)

    logging.warning(str(feature_names[scores.index(max(scores))])+" "+str(max(scores)))
    return feature_names[scores.index(max(scores))]


def get_ranked_features(results):
    features_selected = []
    features_rank = []
    features_score = []

    total_iterations = len(results)

    for i in range(total_iterations):
        result_list = results[i]

        feature_names, scores = [], []

        for result in result_list:
            feature_name = result["feature"]
            score = result["score"]
            feature_names.append(feature_name)
            scores.append(score)

        features_selected.append(feature_names[scores.index(max(scores))])
        features_score.append(max(scores))
        features_rank.append(i)

    return features_selected, features_rank, features_score


def get_list_of_covariates(training_dataset):
    df = pd.read_csv(training_dataset, nrows=1)
    df = df.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')
    return list(df.columns)

import sys
def calculate_feature_ranking_by_cv_elimination(training_dataset, model_function, n_splits, scoring_function, cpus_per_job, gpu_per_job, output_folder, model_function_parameters):

    current_features = get_list_of_covariates(training_dataset)

    start_time = time.time()
    ray.init(address='auto', _redis_password='5241590000000000', ignore_reinit_error=True, local_mode=False)
    iteration_counter = 0
    results = {iteration_counter: find_least_important_feature_cv(training_dataset, current_features, model_function, n_splits, scoring_function, cpus_per_job, gpu_per_job, output_folder, model_function_parameters)}
    lowest_feature = get_lowest_scoring_feature(results[iteration_counter])

    current_features.remove(lowest_feature)
    current_total_feature = len(current_features)

    while current_total_feature > 1:
        logging.warning("features remaining: "+str(current_total_feature))
        iteration_counter = iteration_counter + 1
        results[iteration_counter] = find_least_important_feature_cv(training_dataset, current_features, model_function, n_splits, scoring_function, cpus_per_job, gpu_per_job, output_folder, model_function_parameters)
        lowest_feature = get_lowest_scoring_feature(results[iteration_counter])
        print("lowest_feature", lowest_feature)
        print("current_features", current_features)
        current_features.remove(lowest_feature)
        current_total_feature = len(current_features)

    # handle edge case only one feature remains
    iteration_counter = iteration_counter + 1
    results[iteration_counter] = [{"feature": current_features[0], "score": 0}]
    ray.shutdown()

    output_results_file_path = output_folder / Path('feature_ranking_elimination_results.txt')
    with open(output_results_file_path, 'w') as f:
        print(json.dumps(results, sort_keys=False, indent=4), file=f)

    logging.warning(results)
    logging.warning("time take: "+str(time.time()-start_time))
    print(results)
    features_selected, features_rank, features_score = get_ranked_features(results)
    return features_selected, features_rank, features_score
