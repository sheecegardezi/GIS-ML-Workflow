import pathlib
from mlwkf.models.bootstrapped_models import *
from mlwkf.models.standard_models import *
from mlwkf.models.ensemble_models import *
from mlwkf.evaluation_metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT_FOLDER = pathlib.Path(__file__).parent.parent
RELATIVE_PATH_TO_CONFIG_FILE = PROJECT_ROOT_FOLDER / pathlib.Path('tests/testdata/testing_configuration.ini')
REQUIRED_CONFIG_SECTIONS = [
    "Workflow",
    "OutputFolder",
    "Model",
    "Target",
    "Covariates",
    "FeatureExtraction",
    "HyperParameterOptimization",
    "PredictionMapping",
    "ModelExploration",
    "CovariateDrift",
    "Control",
    "Intermediate",
    "Results"
]
REQUIRED_CONFIG_WORKFLOW_SUBSECTIONS = [
    "DataPreparation",
    "DataExploration",
    "FeatureExtraction",
    "HyperParameterOptimization",
    "ModelExploration",
    "PredictionMapping",
    "CovariateDrift"
]
REQUIRED_CONFIG_OUTPUTFOLDER_SUBSECTIONS = [
]
REQUIRED_CONFIG_MODEL_SUBSECTIONS = [
    "model_function",
    "parameters"
]
REQUIRED_CONFIG_TARGET_SUBSECTIONS = [
    "target_property",
    "target_groupcv",
    "target_groupcv_n_splits",
    "target_path",
    "percentage_oos"
]
REQUIRED_CONFIG_COVARIATES_SUBSECTIONS = [
]

REQUIRED_CONFIG_FEATUREEXTRACTION_SUBSECTIONS = [
    "algorithm",
    "no_features_to_select",
    "scoring_function"
]

REQUIRED_CONFIG_HYPERPARAMETEROPTIMIZATION_SUBSECTIONS = [
    "algorithm",
    "hyper_parameters",
    "n_iteration",
    "scoring_functions",
    "scoring_function_to_use_for_evaluation",
    "n_splits"
]

REQUIRED_CONFIG_PREDICTIONMAPPING_SUBSECTIONS = [
    "path_to_trained_model"
]

REQUIRED_CONFIG_MODELEXPLORATION_SUBSECTIONS = [
    "n_splits",
    "path_to_trained_model",
    "scoring_functions"
]

REQUIRED_CONFIG_COVARIATEDRIFT_SUBSECTIONS = [
    "modeling_function"
]

REQUIRED_CONFIG_CONTROL_SUBSECTIONS = [
    "cpus_per_job",
    "gpu_per_job"
]

REQUIRED_CONFIG_INTERMEDIATE_SUBSECTIONS = [
    "training_dataset",
    "oos_dataset",
    "selected_features",
    "covariates",
    "area_of_interest",
    "drift_dataset",
    "drift_vector_dataset"
]

REQUIRED_CONFIG_RESULTS_SUBSECTIONS = [
    "path_to_feature_ranking_results",
    "best_estimator_prams",
    "best_path_to_trained_model",
    "path_to_hyper_parameter_search_results",
    "best_estimator_scores",
    "path_to_predicted_geotiff"
]

LIST_OF_SUPPORTED_MODELS = (
    XGBRegressor,
    CatBoostRegressor,
    LightGBMRegressor,
    RandomForestRegressor,
    SVMRegressor,
    BootstrappedXGBRegressor,
    BootstrappedSVMRegressor
)

LIST_OF_SUPPORTED_EVALUATION_METRICS = [
    mean_squared_error_scorer,
    mean_absolute_error_scorer,
    r2_scorer,
    rmse_scorer,
    adjusted_r2_scorer
]

LIST_OF_SUPPORTED_HPO_ALGORITHMS = [
    "BayesianOptimization",
    "GridSearch",
    "HyperOptSearch"
]

LIST_OF_SUPPORTED_SCORING_FUNCTION_TO_USE = [
    "groupcv_mean_squared_error_scorer",
    "groupcv_mean_absolute_error_scorer",
    "groupcv_r2_scorer",
    "groupcv_rmse_scorer",
    "groupcv_adjusted_r2_scorer",
    "cv_mean_squared_error_scorer",
    "cv_mean_absolute_error_scorer",
    "cv_r2_scorer",
    "cv_rmse_scorer",
    "cv_adjusted_r2_scorer",
    "oos_mean_squared_error_scorer",
    "oos_mean_absolute_error_scorer",
    "oos_r2_scorer",
    "oos_rmse_scorer",
    "oos_adjusted_r2_scorer"
]

LIST_OF_SUPPORTED_BINARY_FUNCTION = [
    LogisticRegression,
    RandomForestClassifier,
    GaussianNB,
    KNeighborsClassifier
]
