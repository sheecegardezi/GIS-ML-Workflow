import altair as alt
import pickle5 as pickle
from pathlib import Path
import logging
import csv
import numpy as np
import pandas as pd
import copy
import ray


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def create_feature_ranking_graph(features_selected, features_rank, features_score, output_folder):
    path_to_output_chart = output_folder / Path("feature_ranking_graph.html")
    source = pd.DataFrame({
        'features_selected': features_selected,
        'features_rank': features_rank,
        'features_score': features_score
    })

    bars = alt.Chart(source).mark_bar(opacity=0.5).encode(
        x=alt.X('features_selected', sort=alt.EncodingSortField(field="features_rank", op="max")),
        y=alt.Y('features_score')
    )

    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['features_selected'], empty='none')
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x=alt.X('features_selected', sort=alt.EncodingSortField(field="features_rank", op="max")),
        y=alt.Y('features_score'),
        opacity=alt.value(0)
    ).add_selection(
        nearest
    )
    # Draw line on top of bars
    line = bars.mark_line(color='black').encode(
        x=alt.X('features_selected', sort=alt.EncodingSortField(field="features_rank", op="max")),
        y=alt.Y('features_score')
    )

    # Draw points on the line, and highlight based on selection
    points = bars.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw point data location of the selection
    text_1 = bars.mark_text(align='left', dx=5, dy=-30).encode(
        text=alt.condition(nearest, 'features_selected', alt.value(' '))
    )
    text_2 = bars.mark_text(align='left', dx=5, dy=-15).encode(
        text=alt.condition(nearest, 'features_score', alt.value(' '))
    )
    text_3 = bars.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'features_rank', alt.value(' '))
    )

    # Draw x rule at the location of the selection
    x_rules = alt.Chart(source).mark_rule(color='gray').encode(
        x=alt.X('features_selected', sort=alt.EncodingSortField(field="features_rank", op="max")),
    ).transform_filter(
        nearest
    )
    # Draw y rule at the location of the selection
    y_rules = alt.Chart(source).mark_rule(color='gray').encode(
        y=alt.Y('features_score'),
    ).transform_filter(
        nearest
    )

    # Put the layers into a chart and bind the data
    alt.layer(
        line, bars, selectors, x_rules, y_rules, points, text_1, text_2, text_3
    ).properties(
        title='Ranked Features Chart',
        width=1000,
        height=700
    ).save(str(path_to_output_chart))

    return str(path_to_output_chart)


@ray.remote(num_cpus=8)
def get_out_of_sample_score(data_train, label_train, oos_dataset, feature_name, model_function, scoring_function, model_function_parameters):
    print("get_out_of_sample_score", feature_name)
    if feature_name is not None:
        data_train = data_train.drop(feature_name, axis=1)
    else:
        feature_name = data_train.columns.values[0]

    model = model_function(model_function_parameters)
    df = pd.read_csv(oos_dataset)
    df = df.astype('float32')
    df = df[~df.isin([np.nan, np.inf, -np.inf, -9999.0]).any(1)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    label_oos = df['target']
    data_oos = df.drop(["target", "x", "y"], axis=1, errors='ignore')
    data_oos = data_oos.drop(feature_name, axis=1)
    data_oos = data_oos.reindex(list(data_train.columns), axis=1)

    model.fit(data_train, label_train)

    label_pred = model.predict(data_oos)

    score = scoring_function(label_oos.values, label_pred, len(list(data_train.columns)))
    print(feature_name, score)
    return {"score":score, "feature":feature_name}


def output_results(path_to_raw_results_file, output_folder):
    with open(path_to_raw_results_file, "rb") as rawResultsFile:
        rawResultsObject = pickle.load(rawResultsFile)

    list_of_all_features = []
    for iteration_id in rawResultsObject:
        list_of_all_features = list(
            rawResultsObject[iteration_id]["intermediate_results"].keys())
        break

    # extracted ranked feature list from raw results
    lowest_feature = []
    lowest_score = []
    iteration = []
    for iteration_id in rawResultsObject:
        lowest_feature.append(rawResultsObject[iteration_id]["lowest_feature"])
        lowest_score.append(rawResultsObject[iteration_id]["lowest_score"])
        iteration.append(iteration_id)

    top_features = list(set(list_of_all_features) - set(lowest_feature))

    for feature in top_features:
        lowest_feature.append(feature)
        lowest_score.append(lowest_score[-1] + 0.01)
        iteration.append(iteration[-1])

    rank = [iteration_id - len(top_features) for iteration_id in iteration]
    # save ranked feature list
    path_to_ranked_feature_list = output_folder / Path(path_to_raw_results_file.stem + ".csv")
    with open(path_to_ranked_feature_list, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        line = ["Rank", "Feature", "Score", "Iteration"]
        writer.writerow(line)

        rank.reverse()
        lowest_feature.reverse()
        lowest_score.reverse()
        iteration.reverse()

        for i, feature in enumerate(lowest_feature):
            line = [rank[i], feature, lowest_score[i], iteration[i]]
            writer.writerow(line)
    logging.warning("Feature Ranked List saved at: %s ", path_to_ranked_feature_list)

    # save ranked feature vs score graph
    min_y = min(lowest_score) - 0.01
    max_y = max(lowest_score) + 0.01
    df = pd.DataFrame({'rank': rank, 'feature': lowest_feature, 'score': lowest_score, 'iteration': iteration},
                      columns=['rank', 'feature', 'score', 'iteration'])
    path_to_output_chart = output_folder / Path(
        f"{str(path_to_raw_results_file.stem)}.html"
    )

    logging.warning("Output graph file name: %s", str(path_to_output_chart))
    alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X('iteration', scale=alt.Scale(domain=[max(iteration), min(iteration)]),
                axis=alt.Axis(title='Number of feature used')),
        y=alt.Y('score', scale=alt.Scale(domain=[min_y, max_y])),
        tooltip=['feature', 'score', 'rank']
    ).save(str(path_to_output_chart))
    logging.warning("Feature Ranking Chart saved at: %s ", path_to_output_chart)
