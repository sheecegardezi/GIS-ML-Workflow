import pandas as pd
import numpy as np
import random


def calculate_feature_ranking_by_randomness(training_dataset, no_features_to_select):

    df = pd.read_csv(training_dataset)

    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    label_train = df['target']
    data_train = df.drop(["target", "x", "y"], axis=1, errors='ignore')

    features_selected = []
    features_rank = []
    features_score = []
    total_number_of_features = len(data_train.columns)
    random_number_list = random.sample(range(0, total_number_of_features), no_features_to_select)
    for featureIndex, feature_name in enumerate(data_train.columns):
        if featureIndex in random_number_list:
            features_selected.append(feature_name)
            features_rank.append(total_number_of_features-featureIndex)
            features_score.append(featureIndex/10)

    return features_selected, features_rank, features_score
