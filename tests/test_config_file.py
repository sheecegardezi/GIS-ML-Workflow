import configparser
import pathlib
import pytest
from .tests_constants import RELATIVE_PATH_TO_CONFIG_FILE, REQUIRED_CONFIG_SECTIONS, \
    REQUIRED_CONFIG_WORKFLOW_SUBSECTIONS, REQUIRED_CONFIG_OUTPUTFOLDER_SUBSECTIONS, REQUIRED_CONFIG_MODEL_SUBSECTIONS, \
    REQUIRED_CONFIG_TARGET_SUBSECTIONS, REQUIRED_CONFIG_COVARIATES_SUBSECTIONS, REQUIRED_CONFIG_COVARIATEDRIFT_SUBSECTIONS,\
    REQUIRED_CONFIG_FEATUREEXTRACTION_SUBSECTIONS, REQUIRED_CONFIG_HYPERPARAMETEROPTIMIZATION_SUBSECTIONS, \
    REQUIRED_CONFIG_PREDICTIONMAPPING_SUBSECTIONS, REQUIRED_CONFIG_MODELEXPLORATION_SUBSECTIONS, \
    REQUIRED_CONFIG_CONTROL_SUBSECTIONS, REQUIRED_CONFIG_INTERMEDIATE_SUBSECTIONS, REQUIRED_CONFIG_RESULTS_SUBSECTIONS
from .tests_constants import LIST_OF_SUPPORTED_MODELS, LIST_OF_SUPPORTED_EVALUATION_METRICS, \
    LIST_OF_SUPPORTED_HPO_ALGORITHMS, LIST_OF_SUPPORTED_SCORING_FUNCTION_TO_USE, LIST_OF_SUPPORTED_BINARY_FUNCTION

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from mlwkf.models.bootstrapped_models import *
from mlwkf.models.standard_models import *
from mlwkf.models.ensemble_models import *
from mlwkf.evaluation_metrics import *
from ray.tune.sample import (function, sample_from, uniform, quniform, choice,
                             randint, lograndint, qrandint, qlograndint, randn,
                             qrandn, loguniform, qloguniform)


def test_config_file_exists():
    assert Path(RELATIVE_PATH_TO_CONFIG_FILE).exists()


@pytest.fixture
def config():
    config_file_path = Path(RELATIVE_PATH_TO_CONFIG_FILE)
    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(config_file_path)
    return config


@pytest.mark.incremental
class TestConfigFileSections:

    def test_required_config_sections(self, config):
        for required_config_section in REQUIRED_CONFIG_SECTIONS:
            if required_config_section not in config:
                raise Exception(
                    f'A required configuration section: "{required_config_section}" is missing.'
                )

    def test_required_config_subsection_workflow(self, config):
        for required_config_subsection in REQUIRED_CONFIG_WORKFLOW_SUBSECTIONS:
            if required_config_subsection not in config["Workflow"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section Workflow is missing.'
                )

    def test_required_config_subsection_outputfolder(self, config):
        for required_config_subsection in REQUIRED_CONFIG_OUTPUTFOLDER_SUBSECTIONS:
            if required_config_subsection not in config["OutputFolder"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section OutputFolder is missing.'
                )

    def test_required_config_subsection_model(self, config):
        for required_config_subsection in REQUIRED_CONFIG_MODEL_SUBSECTIONS:
            if required_config_subsection not in config["Model"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section Model is missing.'
                )

    def test_required_config_subsection_target(self, config):
        for required_config_subsection in REQUIRED_CONFIG_TARGET_SUBSECTIONS:
            if required_config_subsection not in config["Target"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section Target is missing.'
                )

    def test_required_config_subsection_covariates(self, config):
        for required_config_subsection in REQUIRED_CONFIG_COVARIATES_SUBSECTIONS:
            if required_config_subsection not in config["Covariates"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section Covariates is missing.'
                )

    def test_required_config_subsection_featureextraction(self, config):
        for required_config_subsection in REQUIRED_CONFIG_FEATUREEXTRACTION_SUBSECTIONS:
            if required_config_subsection not in config["FeatureExtraction"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section FeatureExtraction is missing.'
                )

    def test_required_config_subsection_hyperparameteroptimization(self, config):
        for required_config_subsection in REQUIRED_CONFIG_HYPERPARAMETEROPTIMIZATION_SUBSECTIONS:
            if required_config_subsection not in config["HyperParameterOptimization"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section HyperParameterOptimization is missing.'
                )

    def test_required_config_subsection_predictionmapping(self, config):
        for required_config_subsection in REQUIRED_CONFIG_PREDICTIONMAPPING_SUBSECTIONS:
            if required_config_subsection not in config["PredictionMapping"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section PredictionMapping is missing.'
                )

    def test_required_config_subsection_modelexploration(self, config):
        for required_config_subsection in REQUIRED_CONFIG_MODELEXPLORATION_SUBSECTIONS:
            if required_config_subsection not in config["ModelExploration"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section ModelExploration is missing.'
                )

    def test_required_config_subsection_covariatedrift(self, config):
        for required_config_subsection in REQUIRED_CONFIG_COVARIATEDRIFT_SUBSECTIONS:
            if required_config_subsection not in config["CovariateDrift"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section CovariateDrift is missing.'
                )

    def test_required_config_subsection_control(self, config):
        for required_config_subsection in REQUIRED_CONFIG_CONTROL_SUBSECTIONS:
            if required_config_subsection not in config["Control"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section Control is missing.'
                )

    def test_required_config_subsection_intermediate(self, config):
        for required_config_subsection in REQUIRED_CONFIG_INTERMEDIATE_SUBSECTIONS:
            if required_config_subsection not in config["Intermediate"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section Intermediate is missing.'
                )

    def test_required_config_subsection_results(self, config):
        for required_config_subsection in REQUIRED_CONFIG_RESULTS_SUBSECTIONS:
            if required_config_subsection not in config["Results"]:
                raise Exception(
                    f'A required configuration subsection: "{required_config_subsection}" for section Results is missing.'
                )


def dict_contains_dict(small: dict, big: dict):
    return (big | small) == big


@pytest.mark.incremental
class TestConfigFileDataTypes:

    def test_workflow_parameter_types(self, config):
        for parameter in config['Workflow'].keys():
            assert isinstance(config.getboolean('Workflow', parameter), bool)

    def test_output_folder_parameter_types(self, config):
        output_folder = list(config['OutputFolder'].keys())[0]
        if output_folder is None:
            raise Exception(f'Output folder path is missing.')
        assert Path(list(config['OutputFolder'].keys())[0]).exists()

    def test_model_parameter_types(self, config):

        model_function = eval(config.get('Model', 'model_function'))
        assert model_function in LIST_OF_SUPPORTED_MODELS

        model_function_parameters = eval(config.get('Model', 'parameters'))
        assert type(model_function_parameters) is dict
    # TODO create a list of all accepted parameters within each model class and use that for testing
    #     model_function_parameters = eval(config.get('Model', 'parameters', fallback="{}"))
    #     print(set(model_function_parameters.items()).issubset(model_function().param.items()))
    #     print("model_function_parameters.items()", model_function_parameters.items())
    #     print("model_function().param.items()", model_function().param.items())
    #     output_folder = list(config['OutputFolder'].keys())[0]
    #     if output_folder is None:
    #         raise Exception(f'Output folder path is missing.')
    #     assert Path(list(config['OutputFolder'].keys())[0]).exists()

    def test_target_parameter_types(self, config):
        assert type(config.get('Target', 'target_property')) is str
        assert type(config.get('Target', 'target_groupcv')) is str
        assert type(config.getint('Target', 'target_groupcv_n_splits')) is int
        assert type(Path(config.get('Target', 'target_path'))) in [pathlib.Path, pathlib.PosixPath]
        assert type(config.getint('Target', 'percentage_oos')) is int
        # assert type(Path(config.get('Target', 'oos_path'))) in [pathlib.Path, pathlib.PosixPath]

    def test_covariates_parameter_types(self, config):
        for covariate_path in list(config['Covariates'].keys()):
            assert type(Path(covariate_path)) in [pathlib.Path, pathlib.PosixPath]

    def test_feature_extraction_parameter_types(self, config):
        feature_extraction_algorithm = ["FeatureRankingByRandomness", "FeatureRankingByEliminationCV", "FeatureRankingByEliminationOOS", "FeatureRankingByShap", "FeatureRankingByGroupCV"]
        algorithm = config.get('FeatureExtraction', 'algorithm')
        assert algorithm in feature_extraction_algorithm

    def test_no_features_to_select_parameter_types(self, config):
        no_features_to_select = config.getint('FeatureExtraction', 'no_features_to_select')
        assert no_features_to_select >= -1 and no_features_to_select != 0

    def test_scoring_function_parameter_types(self, config):
        scoring_function = eval(config.get('FeatureExtraction', 'scoring_function'))
        assert scoring_function in LIST_OF_SUPPORTED_EVALUATION_METRICS


@pytest.mark.incremental
class TestHPODataTypes:

    def test_algorithm_parameter_types(self, config):

        algorithm = config.get('HyperParameterOptimization', 'algorithm')
        assert algorithm in LIST_OF_SUPPORTED_HPO_ALGORITHMS

    def test_hyper_parameters_parameter_types(self, config):

        hyper_parameters = eval(config.get('HyperParameterOptimization', 'hyper_parameters'))
        assert type(hyper_parameters) is dict

    def test_n_iteration_parameter_types(self, config):

        n_iteration = config.getint('HyperParameterOptimization', 'n_iteration')
        assert type(n_iteration) is int

    def test_scoring_functions_parameter_types(self, config):
        scoring_functions = eval(config.get('HyperParameterOptimization', 'scoring_functions'))
        for scoring_function in scoring_functions:
            assert scoring_function in LIST_OF_SUPPORTED_EVALUATION_METRICS

    def test_scoring_function_to_use_for_evaluation_parameter_types(self, config):

        scoring_function_to_use_for_evaluation = config.get('HyperParameterOptimization', 'scoring_function_to_use_for_evaluation')
        assert scoring_function_to_use_for_evaluation in LIST_OF_SUPPORTED_SCORING_FUNCTION_TO_USE

    def test_n_splits_parameter_types(self, config):
        n_splits = config.getint('HyperParameterOptimization', 'n_splits')
        assert type(n_splits) is int


@pytest.mark.incremental
class TestPredictionMappingTypes:

    def test_algorithm_parameter_types(self, config):
        assert type(Path(config.get('PredictionMapping', 'path_to_trained_model'))) in [pathlib.Path, pathlib.PosixPath]


@pytest.mark.incremental
class TestModelExplorationTypes:

    def test_n_splits_parameter_types(self, config):
        assert type(config.getint('ModelExploration', 'n_splits')) is int

    def test_path_to_trained_model_parameter_types(self, config):
        assert type(Path(config.get('ModelExploration', 'path_to_trained_model'))) in [pathlib.Path, pathlib.PosixPath]

    def test_scoring_functions_parameter_types(self, config):
        scoring_functions = eval(config.get('ModelExploration', 'scoring_functions'))
        for scoring_function in scoring_functions:
            assert scoring_function in LIST_OF_SUPPORTED_EVALUATION_METRICS


@pytest.mark.incremental
class TestCovariateDriftDataTypes:
    def test_modeling_function_parameter_types(self, config):
        modeling_function = eval(config.get('CovariateDrift', 'modeling_function'))
        assert modeling_function in LIST_OF_SUPPORTED_BINARY_FUNCTION


@pytest.mark.incremental
class TestControlDataTypes:

    def test_cpus_per_job_parameter_types(self, config):
        cpus_per_job = config.getint('Control', 'cpus_per_job')
        assert cpus_per_job > 0

    def test_gpu_per_job_parameter_types(self, config):
        gpu_per_job = config.getint('Control', 'gpu_per_job')
        assert gpu_per_job >= 0


@pytest.mark.incremental
class TestIntermediateDataTypes:

    def test_training_dataset_parameter_types(self, config):
        assert type(config.get('Intermediate', 'training_dataset')) is str

    def test_oos_dataset_parameter_types(self, config):
        assert type(config.get('Intermediate', 'oos_dataset')) is str

    def test_selected_features_parameter_types(self, config):
        assert type(config.get('Intermediate', 'selected_features')) is str

    def test_covariates_parameter_types(self, config):
        assert type(config.get('Intermediate', 'covariates')) is str

    def test_area_of_interest_parameter_types(self, config):
        assert type(config.get('Intermediate', 'area_of_interest')) is str

    def test_drift_dataset_parameter_types(self, config):
        assert type(config.get('Intermediate', 'drift_dataset')) is str

    def test_drift_vector_dataset_parameter_types(self, config):
        assert type(config.get('Intermediate', 'drift_vector_dataset')) is str


@pytest.mark.incremental
class TestResultsDataTypes:

    def test_path_to_feature_ranking_results_parameter_types(self, config):
        assert type(config.get('Results', 'path_to_feature_ranking_results')) is str

    def test_best_estimator_prams_parameter_types(self, config):
        assert type(config.get('Results', 'best_estimator_prams')) is str

    def test_best_path_to_trained_model_parameter_types(self, config):
        assert type(config.get('Results', 'best_path_to_trained_model')) is str

    def test_path_to_hyper_parameter_search_results_parameter_types(self, config):
        assert type(config.get('Results', 'path_to_hyper_parameter_search_results')) is str

    def test_best_estimator_scores_parameter_types(self, config):
        assert type(config.get('Results', 'best_estimator_scores')) is str

    def test_path_to_predicted_geotiff_parameter_types(self, config):
        assert type(config.get('Results', 'path_to_predicted_geotiff')) is str
