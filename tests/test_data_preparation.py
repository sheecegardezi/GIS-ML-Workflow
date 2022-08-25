import os
import configparser
import pathlib
import pytest
import pandas as pd
from .tests_constants import RELATIVE_PATH_TO_CONFIG_FILE, PROJECT_ROOT_FOLDER

from mlwkf.data_preparation.pipeline import merge_csv_file


def test_config_file_exists():
    assert pathlib.Path(RELATIVE_PATH_TO_CONFIG_FILE).exists()


@pytest.fixture
def config():
    config_file_path = pathlib.Path(RELATIVE_PATH_TO_CONFIG_FILE)
    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(config_file_path)
    return config


@pytest.mark.incremental
class TestDtaPreparationPipeline:

    def test_merge_csv_file(self, config):
        # create multiple csv files
        number_of_records = 5
        file_paths = []
        for i in range(number_of_records):
            file_path = PROJECT_ROOT_FOLDER / pathlib.Path(f"{str(i)}.csv")
            with open(file_path, "w") as f:
                f.write("id,col\n")
                f.write(f"1,{str(i)}" + "\n")
            file_paths.append(file_path)
        output_file_path = PROJECT_ROOT_FOLDER / pathlib.Path("out.csv")
        # merge multiple csv files using the function
        merge_csv_file(file_paths, output_file_path)

        df = pd.read_csv(output_file_path)
        # test that all the records in individual csv exist in the new csv
        assert len(df.index) == number_of_records

        # clean up artifacts
        try:
            os.remove(str(output_file_path))
        except OSError:
            pass
