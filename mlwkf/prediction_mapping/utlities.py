import shutil
import logging
import os
import warnings
warnings.simplefilter("ignore", UserWarning)
import time
import sys
from pathlib import Path

import ray

import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.merge import merge
from mlwkf.evaluation_metrics import *
from mlwkf.models.bootstrapped_models import *
from mlwkf.models.standard_models import *
from mlwkf.models.ensemble_models import *

from mlwkf.constants import NON_COVARIATES_FIELDS
from ray.tune.sample import (function, sample_from, uniform, quniform, choice,
                             randint, lograndint, qrandint, qlograndint, randn,
                             qrandn, loguniform, qloguniform)


def get_extent_coordinates(area_of_interest):
    dataset = rasterio.open(area_of_interest)

    x_min, y_min, x_max, y_max = dataset.bounds
    x_resolution, _, _, _, y_resolution, _, _, _, _ = dataset.transform

    dataset.close()

    global_x_min, global_y_min, global_x_max, global_y_max = round(x_min), round(y_min), round(x_max), round(y_max)
    x_resolution, y_resolution = abs(round(x_resolution)), abs(round(y_resolution))

    x_resolution = 1 if x_resolution < 1 else x_resolution
    x_array = np.arange(global_x_min + 1, global_x_max, x_resolution, dtype=np.int32)

    y_resolution = 1 if y_resolution < 1 else y_resolution
    y_array = np.arange(global_y_min + 1, global_y_max, y_resolution, dtype=np.int32)

    return x_array, y_array


@ray.remote
def create_line_geotiff(y_index, x_array, covariates, path_to_trained_model, line_geotiff_folder, model_function):
    header = [covariate.stem for covariate in covariates]

    df = pd.DataFrame(columns=header, dtype=float)
    df["x"], df["y"] = x_array, y_index
    resolution = df["x"].iloc[1] - df["x"].iloc[0]

    for covariate in covariates:
        with rasterio.open(covariate) as dataset:
            df[covariate.stem] = [sample[0] for sample in dataset.sample(df[['x', 'y']].to_numpy(), indexes=[1])]

    model = model_function()
    model.load(path_to_trained_model)
    df['target'] = model.predict(df.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore'))

    target_array = np.array([df["target"].to_numpy().reshape(1, df["x"].shape[0])])
    transform = from_origin(min(df["x"]), max(df["y"]), resolution, resolution)
    _, height, width = target_array.shape
    dtype = target_array.dtype
    crs = CRS.from_epsg(3577)
    output_geotif_path = line_geotiff_folder / Path("line_" + str(y_index) + ".tif")

    with rasterio.open(
            output_geotif_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=str(dtype),
            crs=crs,
            transform=transform,
    ) as new_dataset:
        new_dataset.write(target_array)

    return str(output_geotif_path)


def divide_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


@ray.remote
def merged_line_geotiffs(geotif_file_paths, index, bounds, merged_geotiff_folder):
    datasets = []
    for geotif_file_path in geotif_file_paths:
        datasets.append(rasterio.open(geotif_file_path, 'r'))
    output_geotif_path = merged_geotiff_folder / Path("merged_" + str(index) + ".tif")
    merge(datasets=datasets, dst_path=output_geotif_path, bounds=bounds)
    for dataset in datasets:
        dataset.close()
    return str(output_geotif_path)


def merged_geotiffs(geotif_file_paths, bounds, output_folder):

    datasets = []
    for geotif_file_path in geotif_file_paths:
        datasets.append(rasterio.open(geotif_file_path, 'r'))
    output_geotif_path = output_folder / Path("predicted_model.tif")
    merge(datasets=datasets, dst_path=output_geotif_path, bounds=bounds)
    for dataset in datasets:
        dataset.close()
    return output_geotif_path


def get_list_of_tifs_to_merge(merged_geotiff_folder):
    list_of_geotifs = []
    paths = merged_geotiff_folder.glob('**/merged*.tif')
    for path in paths:
        # because path is object not string
        path_in_str = str(path)
        list_of_geotifs.append(path_in_str)

    return list_of_geotifs


def create_predicted_geotiff(area_of_interest, covariates, path_to_trained_model, line_geotiff_folder, merged_geotiff_folder, output_folder, model_function, cpus_per_job, gpu_per_job):
    ray.init(address='auto', _redis_password='5241590000000000', ignore_reinit_error=True)
    start_time = time.time()

    x_array, y_array = get_extent_coordinates(area_of_interest)
    x_array_id = ray.put(x_array)

    covariates_id = ray.put(covariates)
    path_to_trained_model_id = ray.put(path_to_trained_model)
    line_geotiff_folder_id = ray.put(line_geotiff_folder)
    model_function_id = ray.put(model_function)

    result_ids = []
    for y_index in y_array:
        result_ids.append(create_line_geotiff.options(num_cpus=cpus_per_job, num_gpus=gpu_per_job).remote(y_index, x_array_id, covariates_id, path_to_trained_model_id, line_geotiff_folder_id, model_function_id))

    print("Wait for the tasks to complete and retrieve the results.")
    line_geotiff_files = ray.get(result_ids)

    print("Release ray store space")
    del x_array_id
    del covariates_id
    del path_to_trained_model_id
    del line_geotiff_folder_id

    print("Merge line GeoTIFFs")
    # 19000 is the limit of open files for rasterio.merge function
    max_number_of_file_to_merge = 5000
    chunked_line_geotiff_files = list(divide_chunks(line_geotiff_files, max_number_of_file_to_merge))

    with rasterio.open(area_of_interest) as dataset:
        bounds = tuple(dataset.bounds)

    bounds_id = ray.put(bounds)
    merged_geotiff_folder_id = ray.put(merged_geotiff_folder)

    result_ids = []
    for index, chunked_line_geotiff_file in enumerate(chunked_line_geotiff_files):
        result_ids.append(merged_line_geotiffs.remote(chunked_line_geotiff_file, index, bounds_id, merged_geotiff_folder_id))

    print("Wait for the tasks to complete and retrieve the results.")
    merged_geotif_paths = ray.get(result_ids)

    ray.shutdown()
    ray.init(address='auto', _redis_password='5241590000000000', ignore_reinit_error=True)
    # debug get file paths
    # merged_geotif_paths = get_list_of_tifs_to_merge(merged_geotiff_folder)

    # Create output predicted geotiff
    predicted_geotiff = merged_geotiffs(merged_geotif_paths, bounds, output_folder)
    print("Predicted_geotiff create at: ", predicted_geotiff)

    print("Duration: ", time.time() - start_time)
    ray.shutdown()

    # remove tmp files
    print("deleting chunked files.")
    for line_geotiff_file in line_geotiff_files:
        try:
            os.remove(str(line_geotiff_file))
        except OSError as e:
            logging.warning("Error: %s : %s" % (line_geotiff_file, e.strerror))

    print("deleting chunked folder.")
    try:
        shutil.rmtree(line_geotiff_folder)
    except OSError as e:
        logging.warning("Error: %s : %s" % (line_geotiff_folder, e.strerror))

    print("deleting chunked files.")
    for merged_geotif_path in merged_geotif_paths:
        try:
            os.remove(str(merged_geotif_path))
        except OSError as e:
            logging.warning("Error: %s : %s" % (merged_geotif_path, e.strerror))

    print("deleting chunked folder.")
    try:
        shutil.rmtree(merged_geotiff_folder)
    except OSError as e:
        logging.warning("Error: %s : %s" % (merged_geotiff_folder, e.strerror))


    # read aoi that has non data mask
    # read generated geotiff
    # overwrite on data values with -9999.0
    with rasterio.open(str(area_of_interest)) as dataset:
        masked_band = dataset.read(1, masked=True)

    with rasterio.open(str(predicted_geotiff)) as input_dataset:
        profile = input_dataset.profile
        new_path_to_predicted_geotiff = str(predicted_geotiff).replace(".tif", "_fixed.tif")
        with rasterio.open(new_path_to_predicted_geotiff, 'w', **profile) as new_dataset:
            new_dataset.nodata = -9999.0
            band1 = input_dataset.read(1)
            band1[masked_band.mask] = -9999.0
            new_dataset.write(band1, 1)

    return new_path_to_predicted_geotiff
