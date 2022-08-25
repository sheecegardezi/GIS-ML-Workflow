"""
Greater score is always better
"""
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost.utils import eval_metric
# https://catboost.ai/docs/references/eval-metric__supported-metrics.html
# eval_metric(Y_train, train_preds, "R2")[0]

def mean_squared_error_scorer(y_true, y_pred, *args):
    return -1 * mean_squared_error(y_true, y_pred)


def mean_absolute_error_scorer(y_true, y_pred, *args):
    return -1 * mean_absolute_error(y_true, y_pred)


def r2_scorer(y_true, y_pred, *args):
    return r2_score(y_true, y_pred)


def rmse_scorer(y_true, y_pred, *args):
    return -1 * mean_squared_error(y_true, y_pred, squared=False)


def adjusted_r2_scorer(y_true, y_pred, no_of_covariates, *args):

    r2 = r2_score(y_true, y_pred)
    n = len(y_pred)
    p = no_of_covariates
    score = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    score = max(score, 0)
    return score

