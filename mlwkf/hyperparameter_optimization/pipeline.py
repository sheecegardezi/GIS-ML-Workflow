import logging
import configparser
from pathlib import Path

from mlwkf.hyperparameter_optimization.algorithms.grid_search import run_grid_search_algorithm
from mlwkf.hyperparameter_optimization.algorithms.bayesian_search import run_bayesian_search_algorithm
from mlwkf.hyperparameter_optimization.algorithms.hyperopt_search import run_hyperopt_search_algorithm

from mlwkf.evaluation_metrics import *
from mlwkf.models.bootstrapped_models import *
from mlwkf.models.standard_models import *
from mlwkf.models.ensemble_models import *

from ray.tune.sample import (function, sample_from, uniform, quniform, choice,
                             randint, lograndint, qrandint, qlograndint, randn,
                             qrandn, loguniform, qloguniform)


def fillter_out_extra_fields(best_result, scoring_functions):
    keys_to_drop = []
    for key in best_result:
        flag = any(
            scoring_function.__name__ in key
            for scoring_function in scoring_functions
        )

        if not flag:
            keys_to_drop.append(key)
    for key in keys_to_drop:
        best_result.pop(key)
    return best_result


def run_hyper_parameter_optimization_pipeline(config_file_path):

    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(config_file_path)

    cpus_per_job = config.getint('Control', 'cpus_per_job')
    gpu_per_job = config.getint('Control', 'gpu_per_job')

    output_folder = Path(list(config['OutputFolder'].keys())[0])
    training_dataset = Path(config.get('Intermediate', 'training_dataset'))
    oos_dataset = Path(config.get('Intermediate', 'oos_dataset'))
    selected_features = eval(config.get('Intermediate', 'selected_features'))

    model_function = eval(config.get('Model', 'model_function'))

    algorithm_name = config.get('HyperParameterOptimization', 'algorithm')
    scoring_functions = eval(config.get('HyperParameterOptimization', 'scoring_functions'))
    hyper_parameters = eval(config.get('HyperParameterOptimization', 'hyper_parameters'))
    n_iteration = config.getint('HyperParameterOptimization', 'n_iteration')
    n_splits = config.getint('HyperParameterOptimization', 'n_splits')
    scoring_function_to_use_for_evaluation = config.get('HyperParameterOptimization', 'scoring_function_to_use_for_evaluation')

    logging.warning(f"Modeling function being used:{str(model_function)}")

    if not training_dataset.exists():
        raise Exception("Please provide valid input training dataset file.")
    if not oos_dataset.exists():
        raise Exception("Please provide valid input oos dataset file.")

    best_estimator_prams = None
    path_to_trained_model = None
    best_result = None

    if algorithm_name in "BayesianOptimization":
        best_estimator_prams, max_trail_id, best_result = run_bayesian_search_algorithm(training_dataset, oos_dataset, selected_features, model_function, hyper_parameters, scoring_functions, n_iteration, n_splits, scoring_function_to_use_for_evaluation, output_folder, cpus_per_job, gpu_per_job)
        path_to_trained_model = Path(output_folder) / Path(
            f"bayesian_optimization_results/{str(max_trail_id)}/model.bin"
        )

        config['Results']['path_to_hyper_parameter_search_results'] = str(Path(output_folder) / Path("bayesian_optimization_results.csv"))

    elif algorithm_name in "GridSearch":
        best_estimator_prams, max_trail_id, best_result = run_grid_search_algorithm(training_dataset, oos_dataset, selected_features, model_function, hyper_parameters, scoring_functions, n_iteration, n_splits, scoring_function_to_use_for_evaluation, output_folder, cpus_per_job, gpu_per_job)
        path_to_trained_model = Path(output_folder) / Path(
            f"grid_search_results/{str(max_trail_id)}/model.bin"
        )

        config['Results']['path_to_hyper_parameter_search_results'] = str(Path(output_folder) / Path("grid_search_results.csv"))

    elif algorithm_name in "HyperOptSearch":
        best_estimator_prams, max_trail_id, best_result = run_hyperopt_search_algorithm(training_dataset, oos_dataset, selected_features, model_function, hyper_parameters, scoring_functions, n_iteration, n_splits, scoring_function_to_use_for_evaluation, output_folder, cpus_per_job, gpu_per_job)
        path_to_trained_model = Path(output_folder) / Path(
            f"hyperopt_optimization_results/{str(max_trail_id)}/model.bin"
        )

        config['Results']['path_to_hyper_parameter_search_results'] = str(Path(output_folder) / Path("hyperopt_optimization_results.csv"))
    else:
        raise Exception("Algorithm not implemented.")

    # update config parameters
    config['Workflow']['HyperParameterOptimization'] = "False"

    config['ModelExploration']['path_to_trained_model'] = str(path_to_trained_model)
    config['PredictionMapping']['path_to_trained_model'] = str(path_to_trained_model)
    config['Results']['best_path_to_trained_model'] = str(path_to_trained_model)

    config['Results']['best_estimator_prams'] = str(best_estimator_prams)
    config['ModelExploration']['default_parameters'] = str(best_estimator_prams)

    best_result = fillter_out_extra_fields(best_result, scoring_functions)
    config['Results']['best_estimator_scores'] = str(best_result)

    # update config file
    with open(config_file_path, 'w') as configfile:
        config.write(configfile)
