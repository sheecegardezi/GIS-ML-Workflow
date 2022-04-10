import time
import warnings
import ray
import pandas as pd
from ray import tune
from ray.tune import Callback
import pandas as pd
from pathlib import Path
import numpy as np
import ray
import logging

from mlwkf.hyperparameter_optimization.utlities import *
from mlwkf.evaluation_metrics import *
from mlwkf.models.standard_models import *
from ray.tune.sample import (function, sample_from, uniform, quniform, choice,
                             randint, lograndint, qrandint, qlograndint, randn,
                             qrandn, loguniform, qloguniform)

warnings.simplefilter("ignore")


def gird_search_algorithm(training_dataset, oos_dataset, selected_features, model_function, hyper_parameters, n_iteration, n_splits, scoring_functions, scoring_function_to_use_for_evaluation, output_folder, cpus_per_job, gpu_per_job):

    def objective_function(parameters, checkpoint_dir=None):

        model = model_function(parameters)
        oos_scores = get_out_of_sample_score(training_dataset, oos_dataset, selected_features, model, scoring_functions)
        cv_scores = get_cross_validation_score(training_dataset, n_splits, selected_features, model, scoring_functions)
        groupcv_scores = get_group_cross_validation_score(training_dataset, selected_features, model, scoring_functions, n_splits)
        tune.report(**{**oos_scores, **cv_scores, **groupcv_scores})

    def trial_str_creator(trial):
        return trial.trial_id

    def trial_dirname_creator(trial):
        return trial.trial_id

    n_iteration = 1
    analysis = tune.run(
        objective_function,
        num_samples=n_iteration,
        name="grid_search_results",
        config=hyper_parameters,
        resources_per_trial=tune.PlacementGroupFactory([{"CPU": cpus_per_job, "GPU": gpu_per_job}]),
        metric=scoring_function_to_use_for_evaluation,
        mode="max",
        verbose=1,
        max_failures=1,
        raise_on_failed_trial=False,
        local_dir=output_folder,
        trial_name_creator=trial_str_creator,
        trial_dirname_creator=trial_dirname_creator
    )

    return analysis




def run_grid_search_algorithm(training_dataset, oos_dataset, selected_features, model_function, hyper_parameters, scoring_functions, n_iteration, n_splits, scoring_function_to_use_for_evaluation, output_folder, cpus_per_job, gpu_per_job):

    ray.init(address='auto', _redis_password='5241590000000000', ignore_reinit_error=True, local_mode=False)

    analysis = gird_search_algorithm(training_dataset, oos_dataset, selected_features, model_function, hyper_parameters, n_iteration, n_splits, scoring_functions, scoring_function_to_use_for_evaluation, output_folder, cpus_per_job, gpu_per_job)

    # Get a dataframe for analyzing trial results.
    results_path = Path(output_folder) / Path("grid_optimization_results.csv")
    analysis.results_df.to_csv(results_path, header=analysis.results_df.columns.values)

    df = pd.read_csv(results_path)
    max_trail_id = df.loc[df[scoring_function_to_use_for_evaluation] == df[scoring_function_to_use_for_evaluation].max()]["trial_id"].values[0]

    best_estimator_prams = str(analysis.get_best_config(metric=scoring_function_to_use_for_evaluation, mode="max"))

    logging.warning("Best Estimator: %s", best_estimator_prams)

    ray.shutdown()
    return best_estimator_prams, max_trail_id, analysis.best_result
