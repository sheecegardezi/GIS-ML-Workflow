import sys
import pytest
import rasterio
import numpy as np
import argparse
import pathlib
import configparser
import time
import logging
import os
import subprocess
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

from mlwkf.data_preparation.pipeline import run_data_preparation_pipeline
from mlwkf.data_exploration.pipeline import run_data_exploration_pipeline
from mlwkf.feature_extraction.pipeline import run_feature_extraction_pipeline
from mlwkf.hyperparameter_optimization.pipeline import run_hyper_parameter_optimization_pipeline
from mlwkf.prediction_mapping.pipeline import run_prediction_pipeline
from mlwkf.model_exploration.pipeline import run_model_exploration_pipeline
from mlwkf.covariates_drift.pipeline import run_covariate_drift_pipeline

from .tests_constants import RELATIVE_PATH_TO_CONFIG_FILE, PROJECT_ROOT_FOLDER

import logging, select, subprocess

LOG_FILE = "test.log"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,filename=LOG_FILE,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@pytest.mark.incremental
class TestSystemSanity:

    def test_complete_system_run(self):

        cmd = "ray start --head"
        process = Popen(cmd, shell=True)
        process.wait()

        cmd = f"cd {str(PROJECT_ROOT_FOLDER)} && python -m mlwkf -c {str(RELATIVE_PATH_TO_CONFIG_FILE)}"

        process = Popen(cmd, shell=True)
        process.wait()
        print(process)

        cmd = "ray stop"
        process = Popen(cmd, shell=True)
        process.wait()

        predicted_map = PROJECT_ROOT_FOLDER / Path("tests/testdata/output/predicted_model.tif")
        assert predicted_map.exists()

        feature_ranking = PROJECT_ROOT_FOLDER / Path("tests/testdata/output/feature_ranking_graph.html")
        assert feature_ranking.exists()

        data_exploration = PROJECT_ROOT_FOLDER / Path("tests/testdata/output/data_exploration")
        assert data_exploration.exists()

        hyperopt_optimization_results = PROJECT_ROOT_FOLDER / Path("tests/testdata/output/hyperopt_optimization_results")
        assert hyperopt_optimization_results.exists()
