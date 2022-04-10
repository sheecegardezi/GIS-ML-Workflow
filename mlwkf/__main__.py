import argparse
import configparser
import time
import logging
import os

from pathlib import Path

from mlwkf.data_preparation.pipeline import run_data_preparation_pipeline
from mlwkf.data_exploration.pipeline import run_data_exploration_pipeline
from mlwkf.feature_extraction.pipeline import run_feature_extraction_pipeline
from mlwkf.hyperparameter_optimization.pipeline import run_hyper_parameter_optimization_pipeline
from mlwkf.prediction_mapping.pipeline import run_prediction_pipeline
from mlwkf.model_exploration.pipeline import run_model_exploration_pipeline
from mlwkf.covariates_drift.pipeline import run_covariate_drift_pipeline


def validate_file(f):
    if not os.path.isfile(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


def main():

    parser = argparse.ArgumentParser(description='Uncover Machine Learning')
    parser.add_argument("-c", "--config", type=validate_file, help="input configfile", metavar="FILE", required=True)
    parser.add_argument("-l", "--log", type=str.upper, default="INFO", dest="logLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level", required=False)

    args = parser.parse_args()

    # Debugging
    # args.config = "INFO"
    # args.logLevel = "configurations_examples/reference_configuration.ini"

    if args.logLevel:
        logging.basicConfig(level=getattr(logging, args.logLevel))

    logging.warning("Running ML Workflow")
    config_file_path = Path(args.config)
    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(config_file_path)

    # create output directory
    output_folder = Path(list(config['OutputFolder'].keys())[0])
    logging.warning("Creating output folder if it dose not exist: "+str(output_folder))
    output_folder.mkdir(parents=True, exist_ok=True)

    start_experiment_time = time.time()
    if config.getboolean('Workflow', 'DataPreparation'):
        start_data_preparation_time = time.time()
        logging.warning("Preforming DataPreparation")
        run_data_preparation_pipeline(config_file_path)
        logging.warning("Time taken DataPreparation: %s seconds" % (time.time() - start_data_preparation_time))
        logging.warning("Finished DataPreparation")

    if config.getboolean('Workflow', 'DataExploration'):
        logging.warning("Running DataExploration")
        start_data_exploration_time = time.time()
        run_data_exploration_pipeline(config_file_path)
        logging.warning("Time taken DataExploration: %s seconds" % (time.time() - start_data_exploration_time))
        logging.warning("Finished DataExploration")

    if config.getboolean('Workflow', 'FeatureExtraction'):
        logging.warning("Running FeatureExtraction")
        start_feature_extraction_time = time.time()
        run_feature_extraction_pipeline(config_file_path)
        logging.warning("Time taken FeatureExtraction: %s seconds" % (time.time() - start_feature_extraction_time))
        logging.warning("Finished FeatureExtraction")

    if config.getboolean('Workflow', 'HyperParameterOptimization'):
        logging.warning("Running HyperParameterOptimization")
        start_hyperparameter_optimization_time = time.time()
        run_hyper_parameter_optimization_pipeline(config_file_path)
        logging.warning("Time taken HyperParameterOptimization: %s seconds" % (time.time() - start_hyperparameter_optimization_time))
        logging.warning("Finished HyperParameterOptimization")

    if config.getboolean('Workflow', 'ModelExploration'):
        logging.warning("TODO: Running ModelExploration")
        start_model_exploration_time = time.time()
        run_model_exploration_pipeline(config_file_path)
        logging.warning("Time taken ModelExploration: %s seconds" % (time.time() - start_model_exploration_time))
        logging.warning("TODO: Finished ModelExploration")

    if config.getboolean('Workflow', 'PredictionMapping'):
        logging.warning("Running Prediction")
        start_prediction_mapping_time = time.time()
        run_prediction_pipeline(config_file_path)
        logging.warning("Time taken PredictionMapping: %s seconds" % (time.time() - start_prediction_mapping_time))
        logging.warning("Finished Prediction")

    if config.getboolean('Workflow', 'CovariateDrift'):
        logging.warning("Running Covariate Drift")
        start_prediction_mapping_time = time.time()
        run_covariate_drift_pipeline(config_file_path)
        logging.warning("Time taken PredictionMapping: %s seconds" % (time.time() - start_prediction_mapping_time))
        logging.warning("Finished CovariateDrift")

    logging.warning("Time take to complete experiment: %s seconds" % (time.time() - start_experiment_time))
    logging.warning("Finished ML Workflow")


if __name__ == '__main__':
    main()
