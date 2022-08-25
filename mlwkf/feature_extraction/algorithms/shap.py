import pandas as pd
import numpy as np
import ray
import shap
from mlwkf.constants import NON_COVARIATES_FIELDS
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_feature_ranking_by_shap(training_dataset, oos_dataset, model_function, no_features_to_select, output_folder, model_function_parameters, cpus_per_job, gpu_per_job):

    df = pd.read_csv(training_dataset)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    label_train = df['target']
    data_train = df.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')

    df = pd.read_csv(oos_dataset)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # df = df.head(100)
    label_oos = df['target']
    data_oos = df.drop(NON_COVARIATES_FIELDS, axis=1, errors='ignore')
    data_oos = data_oos.reindex(list(data_train.columns), axis=1)

    model = model_function(model_function_parameters)
    model.fit(data_train, label_train)

    explainer = shap.KernelExplainer(model=model.predict, data=data_oos)
    shap_values = explainer.shap_values(X=data_oos)

    shap.summary_plot(shap_values, data_oos.iloc[0,:], plot_type="bar", show=False)
    path_shap_bar_chart = output_folder / Path("shap_bar_chart.png")
    plt.savefig(path_shap_bar_chart)

    # get feature importance
    feature_names = list(data_oos.columns.values)
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    feature_importance = feature_importance.reset_index()
    feature_importance.drop(["index"],axis=1,inplace=True)
    print(feature_importance)

    features_selected = []
    features_rank = []
    features_score = []

    for index, row in feature_importance.iterrows():
        features_selected.append(row['col_name'])
        features_rank.append(index)
        features_score.append(row['feature_importance_vals'])

    return features_selected, features_rank, features_score
