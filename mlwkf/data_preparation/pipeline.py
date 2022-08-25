"""
## Objectives of Data Preparation

- data should be a 2D array
- input data has to be numbers
- no nan or inf
- columns are scaled to similar ranges (mean=0, variance=1)
- columns should not be collinear (cx1!=k*cx2)
- rows should not be causally dependent
- data should be 100 times larger then the number of columns
"""
import sys
from pathlib import Path
import csv
import time
import numpy as np
import pandas as pd
import fiona
import shutil
import rasterio
import logging
import configparser
from rasterio.shutil import copy
from mlwkf.utlities import create_chunked_target
from mlwkf.data_preparation.utlities import remove_duplicate_covariates, check_covariates_exist, \
    convert_multi_band_to_single_band, check_projection_in_epsg3577, create_oos_dataset, \
    check_projection_in_target_epsg3577
import ray
import os
import traceback
from mlwkf.utlities import flatten, save_config_file
from random import shuffle
import logging


def merge_csv_file(list_of_csv_file_paths, output_csv_file_path):
    """
    Merge multiple csv files into one.
    :param list_of_csv_file_paths: a list of csv file paths
    :param output_csv_file_path: an output csv file path
    :return: file path to merged csv file
    """
    logging.warning("Merge csv files")
    logging.warning("Input csv files:")
    logging.warning(str(list_of_csv_file_paths))

    with open(output_csv_file_path, "wb") as fout:
        # first file:
        with open(list_of_csv_file_paths[0], "rb") as f:
            fout.write(f.read())
        # now the rest:
        for csv_file_path in list_of_csv_file_paths[1:]:
            with open(csv_file_path, "rb") as f:
                next(f)  # skip the header
                fout.write(f.read())

    logging.warning("Deleting chunked csv files.")
    for csv_file_path in list_of_csv_file_paths:
        try:
            os.remove(str(csv_file_path))
        except OSError:
            pass


@ray.remote(num_cpus=8, max_retries=5)
def create_chunked_vector_csv(chunked_target_list, target_property, i, covariates, chunked_csv_folder, target_weight, target_groupcv):
    # logging.basicConfig(level=getattr(logging, "INFO"))
    logging.warning("create chunked vector csv")
    logging.warning("Current time: %s", str(time.ctime()))
    # create iterators to inputs feature dataset and target dataset
    datasets = [rasterio.open(covariate) for covariate in covariates]
    # create the first row containing col names
    head_row = ['target']
    head_row.extend(covariate.stem for covariate in covariates)
    head_row.extend(("x", "y"))
    if target_weight is not None:
        head_row.append("weight")
    if target_groupcv is not None:
        head_row.append("groupcv")
    csv_rowlist = [head_row]

    chunked_csv_path = chunked_csv_folder / Path(f"{str(i)}_dataset.csv")
    with open(chunked_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        # Iterate through list of targets
        logging.warning("Creating chunked csv file ...... ")
        for target in chunked_target_list:

            try:
                value = 0 if target_property is None else target['properties'][target_property]
                if target_weight is not None:
                    target_weight_value = target['properties'][target_weight]
                if target_groupcv is not None:
                    target_groupcv_value = target['properties'][target_groupcv]

                # TODO update multiple point logic
                if type(target['geometry']["coordinates"]) is type([]):
                    x, y = target['geometry']['coordinates'][0]
                elif len(target['geometry']['coordinates']) == 3:
                    x, y, z = target['geometry']['coordinates']
                else:
                    x, y = target['geometry']['coordinates']
                new_row = [value]

                # Iterate through list of features
                new_row.extend(next(dataset.sample([(x, y)]))[0] for dataset in datasets)
                new_row.append(x)
                new_row.append(y)
                if target_weight is not None:
                    new_row.append(target_weight_value)
                if target_groupcv is not None:
                    new_row.append(target_groupcv_value)

                csv_rowlist.append(new_row)

            except Exception as err:
                logging.warning(f"Exception: {str(err)}")
                traceback.print_tb(err.__traceback__)

        writer.writerows(csv_rowlist)

    # close iterators
    for dataset in datasets:
        dataset.close()

    df = pd.read_csv(chunked_csv_path)
    df = df.astype('float32')

    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(chunked_csv_path, index=None, header=df.columns.values)

    return chunked_csv_path


def create_vector_csv(covariates, target_path, target_property, output_folder, target_weight, target_groupcv):
    """
    read target dataset and get number of points
    read mask get spatial properties
    create a new dataset set with random points of interest from mask

    :param covariates:
    :param target_path:
    :param target_property:
    :param output_folder:
    :param target_weight:
    :param target_groupcv:
    :return:
    """
    logging.warning("create vector csv")
    logging.warning(f"target_path: {str(target_path)}")

    target_handle = fiona.open(target_path)
    n = 5000
    chunked_target_lists = create_chunked_target(list(target_handle), n)
    target_handle.close()

    chunked_csv_folder = output_folder / Path("chunked_csv_folder")
    chunked_csv_folder.mkdir(parents=True, exist_ok=True)

    result_ids = [
        create_chunked_vector_csv.remote(
            chunked_target_list,
            target_property,
            i,
            covariates,
            chunked_csv_folder,
            target_weight,
            target_groupcv,
        )
        for i, chunked_target_list in enumerate(chunked_target_lists)
    ]

    chunked_csv_paths = ray.get(result_ids)

    output_csv_file_path = output_folder / Path(target_path.stem + "_dataset.csv")
    merge_csv_file(chunked_csv_paths, output_csv_file_path)

    logging.warning("Output file has been created: %s", str(output_csv_file_path))

    try:
        shutil.rmtree(chunked_csv_folder)
    except OSError as e:
        logging.warning(f"Error: {chunked_csv_folder} : {e.strerror}")

    return output_csv_file_path


def create_vrt_datasets(covariates):
    vrt_datasets = []
    for covariate in covariates:
        vrt_dataset_path = covariate.parent / Path(covariate.stem + ".vrt")

        if not os.path.exists(vrt_dataset_path):
            copy(covariate, vrt_dataset_path, driver='VRT')
        else:
            logging.warning(f"this vrt dataset already exists: {str(vrt_dataset_path)}")
        vrt_datasets.append(vrt_dataset_path)

    return vrt_datasets


def create_groupcv_csv(target_path, output_folder, n_splits):

    df = pd.read_csv(target_path)
    df["groupcv_class"] = None
    df = df.sort_values(
        by="groupcv",
        ascending=True
    )

    total_subgroups = list(df['groupcv'].unique())
    shuffle(total_subgroups)

    for row in df.itertuples():
        groupcv_class = int(df.at[row.Index, 'groupcv'])
        index_split_dataset = total_subgroups.index(groupcv_class) % n_splits
        df.at[row.Index, 'groupcv_class'] = index_split_dataset

    output_csv_file_path = output_folder / Path(target_path.stem + "_groupcv.csv")
    df.to_csv(output_csv_file_path, index=None, header=True)
    return output_csv_file_path


def run_data_preparation_pipeline(config_file_path):
    ray.init(address='auto', _redis_password='5241590000000000', ignore_reinit_error=True, local_mode=False)
    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(config_file_path)

    output_folder = Path(list(config['OutputFolder'].keys())[0])

    covariates = [Path(covariate) for covariate in list(config['Covariates'].keys())]

    logging.warning("removing duplicate covariates")
    covariates = remove_duplicate_covariates(covariates)
    logging.warning("check covariates exist")
    covariates = check_covariates_exist(covariates)

    logging.warning("convert multi band to single band")
    results = ray.get([convert_multi_band_to_single_band.remote(covariate, output_folder) for covariate in covariates])
    covariates = flatten(results)

    logging.warning("check projection in epsg3577")
    results = ray.get([check_projection_in_epsg3577.remote(covariate, output_folder) for covariate in covariates])
    covariates = flatten(results)

    covariates = create_vrt_datasets(covariates)
    for covariate in covariates:
        logging.warning(f"New covariate created: {str(covariate)}")

    target_path = Path(config.get('Target', 'target_path'))
    target_property = config.get('Target', 'target_property')

    target_weight = config.get('Target', 'target_weight', fallback=None)
    target_groupcv = config.get('Target', 'target_groupcv', fallback=None)
    target_groupcv_n_splits = config.getint('Target', 'target_groupcv_n_splits', fallback=None)

    logging.warning("check target projection in epsg3577")
    target_path = check_projection_in_target_epsg3577(target_path, output_folder)
    logging.warning(f"Reprojected target_path file created at: {str(target_path)}")

    area_of_interest = Path(config.get('Target', 'area_of_interest'))
    logging.warning("check area_of_interest projection in epsg3577")
    area_of_interest = ray.get(check_projection_in_epsg3577.remote(area_of_interest, output_folder))
    logging.warning(
        f"Reprojected area_of_interest file created at: {str(area_of_interest)}"
    )


    training_dataset = create_vector_csv(covariates, target_path, target_property, output_folder, target_weight, target_groupcv)
    if target_groupcv is not None:
        training_dataset = create_groupcv_csv(training_dataset, output_folder, target_groupcv_n_splits)

    oos_path = Path(config.get('Target', 'oos_path', fallback="None"))

    percentage_oos = config.getint('Target', 'percentage_oos', fallback=None)

    if percentage_oos is not None:
        training_dataset, oos_dataset = create_oos_dataset(target=training_dataset, percentage_oos=percentage_oos)
    else:
        oos_dataset = create_vector_csv(covariates, oos_path, target_property, output_folder, target_weight, target_groupcv)

    if config.getboolean('Workflow', 'ShapValues'):
        shap_path = Path(config.get('ShapValues', 'shap_path', fallback="None"))
        shap_dataset = create_vector_csv(covariates, shap_path, None, output_folder, target_weight, target_groupcv)
        config['Intermediate']['shap_dataset'] = str(shap_dataset)

    # update config parameters
    logging.warning(f"output training_dataset: {str(training_dataset)}")
    logging.warning(f"output oos_dataset: {str(oos_dataset)}")

    config['Intermediate']['target_path'] = str(target_path)
    config['Intermediate']['training_dataset'] = str(training_dataset)
    config['Intermediate']['oos_dataset'] = str(oos_dataset)
    config['Intermediate']['covariates'] = str(covariates)
    config['Intermediate']['area_of_interest'] = str(area_of_interest)

    config['Workflow']['DataPreparation'] = "False"

    save_config_file(config, config_file_path, output_folder)

    ray.shutdown()
