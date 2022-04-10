import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import time
import ray
import configparser
from pathlib import Path, PosixPath

from mlwkf.evaluation_metrics import *
from mlwkf.models.bootstrapped_models import *
from mlwkf.models.standard_models import *
from mlwkf.models.ensemble_models import *

from ray.tune.sample import (function, sample_from, uniform, quniform, choice,
                             randint, lograndint, qrandint, qlograndint, randn,
                             qrandn, loguniform, qloguniform)
from mlwkf.prediction_mapping.utlities import get_extent_coordinates, create_predicted_geotiff


def run_prediction_pipeline(config_file_path):

    os.environ["MODIN_ENGINE"] = "ray"
    time.sleep(2.0)

    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(config_file_path)

    cpus_per_job = config.getint('Control', 'cpus_per_job')
    gpu_per_job = config.getfloat('Control', 'gpu_per_job')

    output_folder = Path(list(config['OutputFolder'].keys())[0])
    output_folder.mkdir(parents=True, exist_ok=True)

    line_geotiff_folder = output_folder / "line_geotiff"
    line_geotiff_folder.mkdir(parents=True, exist_ok=True)

    merged_geotiff_folder = output_folder / "merged_geotiff"
    merged_geotiff_folder.mkdir(parents=True, exist_ok=True)

    selected_features = eval(config.get('Intermediate', 'selected_features'))
    path_to_trained_model = Path(config.get('PredictionMapping', 'path_to_trained_model'))
    area_of_interest = Path(config.get('Target', 'area_of_interest'))
    model_function = eval(config.get('Model', 'model_function'))

    if not path_to_trained_model.exists():
        print(path_to_trained_model)
        raise Exception("Please provide valid path to trained model.")

    if not area_of_interest.exists():
        print(area_of_interest)
        raise Exception("Please provide valid path to area of interest.")

    # TODO update logic for checking feature names
    covariates = []
    for selected_feature in selected_features:
        for covariate in eval(config.get('Intermediate', 'covariates')):
            if selected_feature in str(covariate):
                covariates.append(covariate)
                break

    path_to_predicted_geotiff = create_predicted_geotiff(area_of_interest, covariates, path_to_trained_model, line_geotiff_folder, merged_geotiff_folder, output_folder, model_function, cpus_per_job, gpu_per_job)

    config['Workflow']['PredictionMapping'] = "False"
    config['Results']['path_to_predicted_geotiff'] = str(path_to_predicted_geotiff)

    print("update config file")
    with open(config_file_path, 'w') as configfile:
        config.write(configfile)
