import os
import time
import random
import logging
import configparser
import rasterio
import ray
import json
import pandas as pd
import numpy as np
from pathlib import Path, PosixPath

from mlwkf.data_preparation.utlities import check_projection_in_epsg3577
from mlwkf.data_preparation.pipeline import  create_vector_csv
from mlwkf.utlities import save_config_file
from mlwkf.constants import NON_COVARIATES_FIELDS
from mlwkf.data_preparation.pipeline import create_vrt_datasets
from mlwkf.utlities import read_dataframe_from_csv
from mlwkf.covariates_drift.utlities import get_extent_coordinates, create_predicted_geotiff
from mlwkf.models.bootstrapped_models import *
from mlwkf.models.standard_models import *
from mlwkf.models.ensemble_models import *


from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def df_to_geojson(df, properties, lat='latitude', lon='longitude'):
    # create a new python dict to contain our geojson data, using geojson format
    geojson = {
        'type': 'FeatureCollection',
        "name": "covariate_drift",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3577"}},
        'features': []
    }

    # loop through each row in the dataframe and convert each row to geojson format
    for _, row in df.iterrows():
        # create a feature template to fill in
        feature = {'type': 'Feature',
                   'properties': {},
                   'geometry': {'type': 'Point',
                                'coordinates': []}}

        # fill in the coordinates
        feature['geometry']['coordinates'] = [row[lon], row[lat]]

        # for each column, get the value and add it as a new feature property
        for prop in properties:
            feature['properties'][prop] = row[prop]

        # add this feature (aka, converted dataframe row) to the list of features inside our dict
        geojson['features'].append(feature)

    return geojson


def create_random_point_vector_dataset(area_of_interest, training_dataset, drift_property, output_folder):

    output_vector_dataset = {
        'type': 'FeatureCollection',
        "name": "covariate_drift",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3577"}},
        'features': []
    }
    training_df = read_dataframe_from_csv(training_dataset)
    no_of_points_in_target_dataset = int(len(training_df.index) * 1.1)  # add a 10% margin

    with rasterio.open(area_of_interest) as dataset:
        left, bottom, right, top = tuple(dataset.bounds)

        while no_of_points_in_target_dataset > 0:

            x = random.uniform(left, right)
            y = random.uniform(bottom, top)
            random_value = next(dataset.sample([[x, y]], indexes=[1]))
            if random_value != 0:
                # create a feature template to fill in
                feature = {
                    'type': 'Feature',
                    'properties': {
                        drift_property: 1
                    },
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [x, y]
                    }
                }

                no_of_points_in_target_dataset -= 1
                output_vector_dataset['features'].append(feature)

    path_drift_dataset = output_folder / Path("drift_vector.geojson")
    with open(path_drift_dataset, 'w') as outfile:
        json.dump(output_vector_dataset, outfile)

        # open up
        # get extent

        # generate random x,y codinate
        random_x = None
        random_y = None
        # check if the value is not none
        # dataset.sample([['x', 'y']], indexes=[1])
        # check if the value is not already part of targets
        # check if the value is not already part of drift
        # add selected point to new geojson

    return path_drift_dataset


def run_covariate_drift_pipeline(config_file_path):

    print("Running covariate_drift")

    os.environ["MODIN_ENGINE"] = "ray"
    time.sleep(2.0)

    ray.init(address='auto', _redis_password='5241590000000000', ignore_reinit_error=True, local_mode=False)
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
    modeling_function = eval(config.get('CovariateDrift', 'modeling_function', fallback=LogisticRegression))


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

    training_dataset = Path(config.get('Intermediate', 'training_dataset', fallback=None))
    drift_property = config.get('Target', 'target_property')
    area_of_interest = ray.get(check_projection_in_epsg3577.remote(area_of_interest, output_folder))
    drift_vector_dataset = create_random_point_vector_dataset(area_of_interest, training_dataset, drift_property, output_folder)
    drift_dataset = create_vector_csv(covariates, drift_vector_dataset, drift_property, output_folder, target_weight=None, target_groupcv=None)

    # read drift dataset
    drift_df = read_dataframe_from_csv(drift_dataset)
    training_df = read_dataframe_from_csv(training_dataset)

    training_df["class"] = 1
    drift_df["class"] = 2

    # Stack the DataFrames on top of each other
    vertical_stack = pd.concat([drift_df, training_df], axis=0)

    y_train = vertical_stack['class']
    X_train = vertical_stack.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')

    scaler = StandardScaler()
    X_train_transformed = scaler.fit_transform(X_train)

    model = modeling_function()
    model_function_name = type(model).__name__

    model.fit(X_train_transformed, y_train)

    y_pred = model.predict_proba(X_train_transformed)
    y_pred = pd.DataFrame(y_pred, columns=["CLASS_1", "CLASS_2"])
    X_train['probability_class_1'] = y_pred['CLASS_1']
    X_train['actual_class'] = y_train

    path_vertical_stack_csv = output_folder / Path(model_function_name + "_drift_output.csv")
    X_train.to_csv(path_vertical_stack_csv, index=None, header=X_train.columns.values)

    ###############################
    # Create drift prediction map #
    ###############################
    output_folder = Path(list(config['OutputFolder'].keys())[0])
    output_folder.mkdir(parents=True, exist_ok=True)

    line_geotiff_folder = output_folder / "line_geotiff"
    line_geotiff_folder.mkdir(parents=True, exist_ok=True)

    merged_geotiff_folder = output_folder / "merged_geotiff"
    merged_geotiff_folder.mkdir(parents=True, exist_ok=True)

    path_to_predicted_drift_geotiff = create_predicted_geotiff(area_of_interest, covariates, path_to_trained_model,
                                                         line_geotiff_folder, merged_geotiff_folder, output_folder,
                                                         model, cpus_per_job, gpu_per_job)

    config['Intermediate']['drift_vector_dataset'] = str(drift_vector_dataset)
    config['Intermediate']['drift_dataset'] = str(drift_dataset)
    config['Intermediate']['path_to_predicted_drift_geotiff'] = str(path_to_predicted_drift_geotiff)
    config['Workflow']['CovariateDrift'] = "False"

    save_config_file(config, config_file_path, output_folder)

    ray.shutdown()
