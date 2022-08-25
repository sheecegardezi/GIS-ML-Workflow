import os
from pathlib import Path
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import logging
import ray
import numpy as np
import pandas as pd
import subprocess


def create_chunked_target(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def get_no_of_cpus():
    return int(ray.cluster_resources()["CPU"] - 4)


def remove_duplicate_covariates(covariates):
    temp_covariates = covariates.copy()
    unique_values = list(set(temp_covariates))
    for unique_value in unique_values:
        temp_covariates.remove(unique_value)
    if len(temp_covariates) > 0:
        logging.warning("Following are duplicates:")
        logging.warning("Duplicate covariate: %s", temp_covariates)
    return list(set(covariates))


def check_covariates_exist(covariates):
    for covariate in covariates:
        if not os.path.isfile(covariate):
            covariates.remove(covariate)
            logging.warning("Covariate does not exist: %s", covariate)
    return covariates


@ray.remote
def convert_multi_band_to_single_band(covariate, output_folder):
    # logging.basicConfig(level=getattr(logging, "INFO"))
    logging.warning("convert_multi_band_to_single_band: %s", covariate)
    with rasterio.open(covariate) as dataset:
        covariates = []
        if len(dataset.indexes) > 1:
            logging.warning("Multi-band dataset found: %s", covariate)
            for band in dataset.indexes:
                path_to_new_dataset = output_folder / Path(str(covariate.stem + "_" + str(band) + ".tif"))
                logging.warning("Reading data from band: %s", band)
                new_dataset = rasterio.open(
                    path_to_new_dataset,
                    'w',
                    driver='GTiff',
                    height=dataset.shape[0],
                    width=dataset.shape[1],
                    count=1,
                    dtype=dataset.meta['dtype'],
                    crs=dataset.crs,
                    transform=dataset.transform
                )

                new_dataset.write(dataset.read(band), 1)
                new_dataset.close()
                logging.warning("New dataset added: %s", path_to_new_dataset)
                covariates.append(path_to_new_dataset)
        else:
            covariates = [covariate]

    return covariates


def check_projection_in_target_epsg3577(target_path, output_folder):
    """
    Docs: https://gdal.org/programs/ogr2ogr.html
    """
    new_target_file_path = output_folder / Path(str(target_path.stem + "_reprojected.geojson"))
    cmd = f"ogr2ogr -f GeoJSON -dim XY -t_srs EPSG:3577 {str(new_target_file_path)} {str(target_path)}"

    logging.warning("Running command:")
    logging.warning(cmd)
    subprocess.getstatusoutput(cmd)
    logging.warning(
        f"Reprojected target file created at: {str(new_target_file_path)}"
    )


    return new_target_file_path


@ray.remote
def check_projection_in_epsg3577(covariate, output_folder):
    # logging.basicConfig(level=getattr(logging, "INFO"))
    logging.warning("check_projection_in_epsg3577: %s", covariate)
    crs_epsg3577 = rasterio.crs.CRS.from_string('EPSG:3577')

    with rasterio.open(covariate) as src:
        if crs_epsg3577 != src.crs:
            logging.warning("Converting dataset: %s", str(covariate))
            transform, width, height = calculate_default_transform(
                src.crs,
                crs_epsg3577,
                src.width,
                src.height,
                *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({'crs': crs_epsg3577, 'transform': transform, 'width': width, 'height': height})

            new_covariate_file_path = output_folder / Path(str(covariate.stem + "_reprojected.tif"))
            with rasterio.open(new_covariate_file_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs_epsg3577,
                    resampling=Resampling.nearest)

            logging.warning("Dataset: %s has been reprojected and saved as: %s", covariate, new_covariate_file_path)
        else:
            new_covariate_file_path = covariate

    return new_covariate_file_path


def create_oos_dataset(target, percentage_oos):
    # take a N% of target dataset and save it as oos dataset

    target_df = pd.read_csv(target)

    shuffle = np.random.permutation(len(target_df))
    test_size = int(len(target_df) * (percentage_oos / 100))
    test_aux = shuffle[:test_size]
    train_aux = shuffle[test_size:]

    new_target_df = target_df.iloc[train_aux]
    training_dataset_path = target.parent / Path(target.stem + "_target_dataset.csv")
    new_target_df.reset_index(drop=True, inplace=True)
    new_target_df.to_csv(training_dataset_path, index=None, header=new_target_df.columns.values)

    oos_df = target_df.iloc[test_aux]
    oos_dataset_path = target.parent / Path(target.stem + "_oos_dataset.csv")
    oos_df.reset_index(drop=True, inplace=True)
    oos_df.to_csv(oos_dataset_path, index=None, header=oos_df.columns.values)

    return training_dataset_path, oos_dataset_path
