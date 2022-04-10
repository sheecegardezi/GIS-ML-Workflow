import xgboost as xgb
from catboost import CatBoost, Pool
import matplotlib.pyplot as plt
import lightgbm as lgb
import chefboost as chef

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import NuSVR
import joblib
import pandas as pd


class XGBRegressor(object):

    def __init__(self, param=None):

        if param is None:
            param = {}
        self.param = {
            'booster': 'gbtree',
            'verbosity': 1,
            'validate_parameters': False,
            'nthread': 8,
            'disable_default_eval_metric': True,
            'learning_rate': 0.3,
            'gamma': 0.11,
            'max_depth': 13,
            'min_child_weight': 4,
            'max_delta_step': 3,
            'subsample': 0.65,
            'sampling_method': 'uniform',
            'colsample_bytree': 1,
            'colsample_bylevel': 0.45,
            'colsample_bynode': 1,
            'lambda': 1,
            'alpha': 0,
            'tree_method': 'hist',
            'sketch_eps': 0.03,
            'scale_pos_weight': 1,
            'refresh_leaf': 1,
            'process_type': 'default',
            'grow_policy': 'depthwise',
            'max_leaves': 0,
            'max_bin': 256,
            "predictor": "cpu_predictor",
            'num_parallel_tree': 1,
            'single_precision_histogram': True,
            # 'deterministic_histogram': False,
            'objective': 'reg:squarederror',
            'seed': 0,
            'num_boost_round': 500,
            'base_score': 0.5,
            'seed_per_iteration': False

        }

        for key in param.keys():
            self.param[key] = param[key]

        self.param['learning_rate'] = float(self.param['learning_rate']) if self.param.get('learning_rate') else 0.3
        self.param['gamma'] = float(self.param['gamma']) if self.param.get('gamma') else 0
        self.param['max_depth'] = int(self.param['max_depth']) if self.param.get('max_depth') else 6
        self.param['max_bin'] = int(self.param['max_bin']) if self.param.get('max_bin') else 256

        self.num_boost_round = int(self.param.pop('num_boost_round', 100))

        self.model = xgb.Booster(self.param)
        self.hyper_parameters = {
            'learning_rate': [0.01, 0.1],
            'gamma': [0, 1],
            'max_depth': [3, 25],
            'min_child_weight': [0, 1],
            'max_delta_step': [0, 1],
            'subsample': [0, 1],
            'colsample_bytree': [0, 1],
            'colsample_bylevel': [0, 1],
            'colsample_bynode': [0, 1],
            'lambda': [0.8, 1],
            'alpha': [0, 0.1],
            'sketch_eps': [0.01, 0.05],
            'scale_pos_weight': [0.1, 0],
            'max_bin': [256, 500],
            'num_boost_round': [300, 900],
            'base_score': [0.2, 0.7]
        }

    def fit(self, data, label, weight=None):

        dtrain = xgb.DMatrix(data, label=label, weight=weight)
        self.model = xgb.train(self.param, dtrain, self.num_boost_round)

    def predict(self, data):

        data = xgb.DMatrix(data, feature_names=self.model.feature_names, nthread=-1)
        return self.model.predict(data)

    def plot_feature_importance(self, output_file_path):

        ax = xgb.plot_importance(self.model)
        ax.figure.tight_layout()
        ax.figure.savefig(output_file_path)

    def save(self, model_file_path):
        self.model.save_model(model_file_path)

    def load(self, model_file_path):
        self.model.load_model(model_file_path)

    def get_model(self):
        return self.model


class CatBoostRegressor(object):
    """
    https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    https://catboost.ai/docs/references/eval-metric__supported-metrics.html
    """

    def __init__(self, param=None):
        if param is None:
            self.param = {
                'num_boost_round': 10,
                'learning_rate': 0.1,
                'random_seed': 0,
                'depth': 15,
                'max_bin': 255,
                'grow_policy': 'SymmetricTree',
                'verbose': 10,
                'task_type': "CPU",
                'iterations': 100,
                'thread_count': -1
            }
        else:
            self.param = param

        self.model = CatBoost(self.param)
        self.search_grid = {
            'num_boost_round': [20, 1000],
            'learning_rate': [0.01, 10],
            'random_seed': [0],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'random_strength': [0, 10],
            'bagging_temperature': [0, 50],
            'max_bin': [128, 255],
            'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
            'min_child_samples': [0, 100],
            'early_stopping_rounds': [20, 50]
        }

    def fit(self, data, label):

        train_data = Pool(data, label)
        self.model.fit(train_data)
        self.model.set_feature_names(data.columns.values)

    def predict(self, data):
        return self.model.predict(data)

    def plot_feature_importance(self, output_file_path):

        df = self.model.get_feature_importance(prettified=True, thread_count=-1, verbose=True)
        y, x = zip(*sorted(zip(df["Importances"], df["Feature Id"])))
        plt.barh(x, y)
        plt.title("Feature Importance")
        plt.xlabel("Feature Score")
        plt.ylabel('Feature Name')
        plt.savefig(output_file_path, bbox_inches='tight')

    def save(self, model_file_path):
        self.model.save_model(model_file_path, format="cbm")

    def load(self, model_file_path):
        self.model.load_model(model_file_path, format='cbm')


class LightGBMRegressor(object):
    """
    TODO
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor
    """

    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)
        self.model = None

    def fit(self, data):
        self.model = lgb.train(data)

    def predict(self, data):
        return self.model.predict(data)

    def plot_feature_importance(self, output_file_path):
        self.model.plot_importance(output_file_path)

    def save(self, output_file_path):
        self.model.save_model(output_file_path)


class RandomForestRegressor(object):
    """
    TODO
    https://sefiks.com/2018/08/28/a-step-by-step-regression-decision-tree-example/
    https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
    https://github.com/kjw0612/awesome-random-forest
    https://github.com/glouppe/phd-thesis

    alternative:
    https://xgboost.readthedocs.io/en/latest/tutorials/rf.html
    """

    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)
        self.model = None

    def fit(self, data):
        params = {'algorithm': 'Regression'}
        self.model = chef.train(data, params)

    def predict(self, data):
        return chef.predict(self.model, data)

    def plot_feature_importance(self, output_file_path):
        self.model.plot_importance(output_file_path)

    def save(self, output_file_path):
        self.model.save_model(output_file_path)


class SVMRegressor(object):

    def __init__(self, iterable=(), **kwargs):
        self.model = NuSVR()
        self.features_names = None

    def fit(self, data, label):
        self.features_names = data.columns.values
        self.model.fit(data, label)

    def predict(self, data):
        return self.model.predict(data)

    def plot_feature_importance(self, output_file_path):

        importance_value, features_names = zip(*sorted(zip(self.model.coef_, self.features_names)))
        plt.barh(range(len(features_names)), importance_value, align='center')
        plt.yticks(range(len(features_names)), features_names)
        plt.title("Feature Importance")
        plt.xlabel("Feature Score")
        plt.ylabel('Feature Name')
        plt.savefig(output_file_path, bbox_inches='tight')

    def save(self, output_file_path):
        joblib.dump(self.model, output_file_path)

    def load(self, model_file_path):
        self.model = joblib.load(model_file_path)

    def get_model(self):
        return self.model
