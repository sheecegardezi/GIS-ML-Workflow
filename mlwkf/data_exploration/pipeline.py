from pathlib import Path
import configparser
import datashader as ds
import pandas as pd
import colorcet
import pathlib
from datashader.utils import export_image
from mlwkf.utlities import save_config_file
import altair as alt
from altair_saver import save
from mlwkf.constants import NON_COVARIATES_FIELDS


def create_bar_plot(path_to_input_csv, output_folder):
    df = pd.read_csv(path_to_input_csv)

    for col in df.columns.values:
        if col not in NON_COVARIATES_FIELDS:
            output_image_file = output_folder / pathlib.Path(col + "_distribution_bar_plot.html")

            alt.Chart(df).mark_bar().encode(
                x=alt.X(col, bin=True, title=col),
                y='count()',
                tooltip=[col, 'count()', 'x', 'y']
            ).properties(
                width=1000,
                height=800
            ).save(str(output_image_file))


def create_scatter_plot(path_to_input_csv, output_folder):
    df = pd.read_csv(path_to_input_csv)

    for col in df.columns.values:
        if col not in NON_COVARIATES_FIELDS:
            output_image_file = output_folder / pathlib.Path("target_vs_" + col + "_scatter_plot.html")
            alt.Chart(df).mark_circle(size=20).encode(
                x=col,
                y='target',
                tooltip=['target', col, "x", "y"]
            ).properties(
                width=1000,
                height=800
            ).save(str(output_image_file))


def create_geo_plot(path_to_input_csv, output_folder):
    df = pd.read_csv(path_to_input_csv)
    cvs = ds.Canvas(plot_width=850, plot_height=500)
    agg = cvs.points(df[["target", "x", "y"]], 'x', 'y')
    img = ds.tf.shade(agg, cmap=colorcet.fire, how='log')
    output_image_file = output_folder / pathlib.Path("target_plot.png")
    export_image(img, str(output_image_file))


# def create_scatter_plot(y_pred, y_oos, output_scatter_plot):
#     data_point = []
#     label = []
#     iteration = []
#     for i in range(len(y_pred)):
#         data_point.append(float(y_pred[i]))
#         data_point.append(float(y_oos[i]))
#
#         label.append("y_pred")
#         label.append("y_oos")
#
#         iteration.append(i)
#         iteration.append(i)
#
#     df = pd.DataFrame({'data_point': data_point, 'label': label, 'iteration':iteration}, columns=['data_point', 'label', 'iteration'])
#
#     alt.Chart(df).mark_circle(size=30).encode(
#         x='iteration',
#         y='data_point',
#         color='label',
#         tooltip=['label', 'data_point']
#     ).save(str(output_scatter_plot))


def create_correlation_plot(path_to_input_csv, output_folder):
    df = pd.read_csv(path_to_input_csv)


def run_data_exploration_pipeline(config_file_path):
    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(config_file_path)

    output_folder = Path(list(config['OutputFolder'].keys())[0])
    training_dataset = Path(config.get('Intermediate', 'training_dataset', fallback=None))
    oos_dataset = Path(config.get('Intermediate', 'oos_dataset', fallback=None))

    data_exploration_output_folder = output_folder / Path("data_exploration")
    data_exploration_output_folder.mkdir(parents=True, exist_ok=True)

    training_data_exploration_output_folder = data_exploration_output_folder / Path("training")
    training_data_exploration_output_folder.mkdir(parents=True, exist_ok=True)

    oos_data_exploration_output_folder = data_exploration_output_folder / Path("oos")
    oos_data_exploration_output_folder.mkdir(parents=True, exist_ok=True)

    create_bar_plot(training_dataset, training_data_exploration_output_folder)
    create_scatter_plot(training_dataset, training_data_exploration_output_folder)
    create_geo_plot(training_dataset, training_data_exploration_output_folder)

    create_bar_plot(oos_dataset, oos_data_exploration_output_folder)
    create_scatter_plot(oos_dataset, oos_data_exploration_output_folder)
    create_geo_plot(oos_dataset, oos_data_exploration_output_folder)

    config['Workflow']['DataExploration'] = "False"
    save_config_file(config, config_file_path, output_folder)
