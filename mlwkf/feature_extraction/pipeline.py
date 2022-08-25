import configparser
from pathlib import Path, PosixPath
import numpy as np
import pandas as pd
import logging
from mlwkf.feature_extraction.algorithms.cv_elimination import calculate_feature_ranking_by_cv_elimination
from mlwkf.feature_extraction.algorithms.oos_elimination import calculate_feature_ranking_by_oos_elimination
from mlwkf.feature_extraction.algorithms.randomness import calculate_feature_ranking_by_randomness
from mlwkf.feature_extraction.algorithms.shap import calculate_feature_ranking_by_shap
from mlwkf.feature_extraction.algorithms.groupcv import calculate_feature_ranking_by_groupcv


from mlwkf.feature_extraction.utlities import create_feature_ranking_graph, output_results
from mlwkf.evaluation_metrics import mean_squared_error_scorer, mean_absolute_error_scorer, r2_scorer, rmse_scorer, adjusted_r2_scorer
from mlwkf.models.bootstrapped_models import *
from mlwkf.models.standard_models import *
from mlwkf.models.ensemble_models import *



def run_feature_extraction_pipeline(config_file_path):

    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(config_file_path)

    cpus_per_job = config.getint('Control', 'cpus_per_job')
    gpu_per_job = config.getint('Control', 'gpu_per_job')
    output_folder = Path(list(config['OutputFolder'].keys())[0])
    training_dataset = Path(config.get('Intermediate', 'training_dataset'))
    oos_dataset = Path(config.get('Intermediate', 'oos_dataset'))
    print("output_folder", output_folder)
    covariates = [
        Path(covariate_path)
        for covariate_path in eval(config.get('Intermediate', 'covariates'))
    ]

    algorithm = config.get('FeatureExtraction', 'algorithm')
    no_features_to_select = config.getint('FeatureExtraction', 'no_features_to_select', fallback=-1)
    no_features_to_select = len(covariates) if no_features_to_select == -1 else no_features_to_select

    if algorithm == "FeatureRankingByRandomness":

        logging.warning("Running FeatureRankingByRandomness")
        features_selected, features_rank, features_score = calculate_feature_ranking_by_randomness(training_dataset, no_features_to_select)
        path_to_output_chart = create_feature_ranking_graph(features_selected, features_rank, features_score, output_folder)
        features_selected = str(features_selected)

    elif algorithm == "FeatureRankingByEliminationCV":

        logging.warning("Running FeatureRankingByEliminationCV")
        n_splits = config.getint('FeatureExtraction', 'n_splits')
        model_function = eval(config.get('Model', 'model_function'))
        model_function_parameters = eval(config.get('Model', 'parameters', fallback="{}"))
        logging.warning(f"Modeling function being used:{str(model_function)}")
        scoring_function = eval(config.get('FeatureExtraction', 'scoring_function'))
        features_selected, features_rank, features_score = calculate_feature_ranking_by_cv_elimination(training_dataset, model_function, n_splits, scoring_function, cpus_per_job, gpu_per_job, output_folder, model_function_parameters)
        path_to_output_chart = create_feature_ranking_graph(features_selected, features_rank, features_score, output_folder)
        features_selected = features_selected[len(features_selected)-no_features_to_select:]
        features_selected = str(features_selected)

    elif algorithm == "FeatureRankingByEliminationOOS":

        logging.warning("Running FeatureRankingByEliminationOOS")
        model_function = eval(config.get('Model', 'model_function', fallback="XGBRegressor"))
        model_function_parameters = eval(config.get('Model', 'parameters', fallback="{}"))
        scoring_function = eval(config.get('FeatureExtraction', 'scoring_function', fallback=mean_absolute_error_scorer))
        features_selected, features_rank, features_score = calculate_feature_ranking_by_oos_elimination(training_dataset, oos_dataset, model_function, scoring_function, cpus_per_job, gpu_per_job, output_folder, model_function_parameters)
        path_to_output_chart = create_feature_ranking_graph(features_selected, features_rank, features_score, output_folder)
        features_selected = features_selected[len(features_selected)-no_features_to_select:]
        features_selected = str(features_selected)

    elif algorithm == "FeatureRankingByShap":

        logging.warning("Running FeatureRankingByShap")
        model_function = eval(config.get('Model', 'model_function', fallback="XGBRegressor"))
        model_function_parameters = eval(config.get('Model', 'parameters', fallback="{}"))
        features_selected, features_rank, features_score = calculate_feature_ranking_by_shap(training_dataset, oos_dataset, model_function, no_features_to_select, output_folder, model_function_parameters, cpus_per_job, gpu_per_job)
        path_to_output_chart = create_feature_ranking_graph(features_selected, features_rank, features_score, output_folder)
        features_selected = features_selected[:no_features_to_select]
        features_selected = str(features_selected)

    elif algorithm == "FeatureRankingByGroupCV":

        target_groupcv = config.get('Target', 'target_groupcv', fallback=None)
        if target_groupcv is None:
            raise Exception("To run FeatureRankingByGroupCV please provide target_groupcv property.")
        model_function = eval(config.get('Model', 'model_function', fallback="XGBRegressor"))
        model_function_parameters = eval(config.get('Model', 'parameters', fallback="{}"))
        scoring_function = eval(config.get('FeatureExtraction', 'scoring_function'))
        n_splits = config.getint('Target', 'target_groupcv_n_splits', fallback=None)
        features_selected, features_rank, features_score = calculate_feature_ranking_by_groupcv(training_dataset, model_function, scoring_function, cpus_per_job, gpu_per_job, output_folder, model_function_parameters, n_splits)
        path_to_output_chart = create_feature_ranking_graph(features_selected, features_rank, features_score, output_folder)
        features_selected = features_selected[len(features_selected)-no_features_to_select:]
        features_selected = str(features_selected)

    else:
        raise Exception("Algorithm not implemented.")

    # update config parameters
    logging.warning(f"selected_features: {str(features_selected)}")
    logging.warning(f"features_rank: {str(features_rank)}")
    logging.warning(f"features_score: {str(features_score)}")

    config['Intermediate']['selected_features'] = features_selected
    config['Workflow']['FeatureExtraction'] = "False"
    config['Results']['path_to_feature_ranking_results'] = str(path_to_output_chart)

    # update config file
    with open(config_file_path, 'w') as configfile:
        config.write(configfile)

