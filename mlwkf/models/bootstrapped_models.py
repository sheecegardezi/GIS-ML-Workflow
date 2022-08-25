import numpy as np
import xgboost as xgb
from numpy import mean, median
from sklearn.svm import NuSVR


class BootstrappedXGBRegressor(object):

    def __init__(self, param=None):

        if param is None:
            param = {}
        self.param = {
            'bootstrapped_number_of_models': 10,
            'bootstrapped_weight': None,
            'bootstrapped_fraction': 0.9,
            'bootstrapped_replace': False,
            'bootstrapped_random_state': 0,

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
        self.bootstrapped_number_of_models = int(self.param.pop('bootstrapped_number_of_models', 10))
        self.bootstrapped_weight = self.param.pop('bootstrapped_weight', None)
        self.bootstrapped_fraction = float(self.param.pop('bootstrapped_fraction', 0.9))
        self.bootstrapped_replace = bool(self.param.pop('bootstrapped_replace', False))
        self.bootstrapped_random_state = int(self.param.pop('bootstrapped_random_state', 0))

        self.predicted_results = None

        self.models = [
            xgb.Booster(self.param)
            for _ in range(self.bootstrapped_number_of_models)
        ]

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
            'base_score': [0.2, 0.7],
        }

    def fit(self, data, label, weight=None):

        data["label"] = label
        data = data.sample(frac=self.bootstrapped_fraction, replace=self.bootstrapped_replace, random_state=self.bootstrapped_random_state)
        label = data["label"]
        data = data.drop(["label"], axis=1)

        self.models = []
        for i in range(self.bootstrapped_number_of_models):
            dtrain = xgb.DMatrix(data, label=label, weight=weight, nthread=-1)
            self.param['seed'] = i
            self.models.append(xgb.train(self.param, dtrain, self.num_boost_round))

    def predict(self, data):

        data = xgb.DMatrix(data, feature_names=self.models[0].feature_names, nthread=-1)
        predicted_results = [
            self.models[i].predict(data)
            for i in range(self.bootstrapped_number_of_models)
        ]

        predicted_results = np.array(predicted_results)
        self.predicted_results = predicted_results

        return predicted_results.mean(axis=0)

    def plot_feature_importance(self, output_file_path):
        pass

    def save(self, model_file_path):
        # self.model.save_model(model_file_path)
        pass

    def load(self, model_file_path):
        # self.model.load_model(model_file_path)
        pass

    def get_model(self):
        # return self.model
        pass


class BootstrappedSVMRegressor(object):

    def __init__(self, param=None):

        if param is None:
            param = {}
        self.param = {
            'bootstrapped_number_of_models': 10,
            'bootstrapped_weight': None,
            'bootstrapped_fraction': 0.9,
            'bootstrapped_replace': False,
            'bootstrapped_random_state': 0,

            'nu': 0.5,
            'C': 1.0,
            'kernel': 'rbf',
            'degree': 3,
            'coef0': 0.0,
            'shrinking': True,
            'tol': 0.001,
            'cache_size': 200,
            'max_iter': -1,
            'gamma': 'scale'
        }

        for key in param.keys():
            self.param[key] = param[key]

        self.param['nu'] = float(self.param['nu']) if self.param.get('nu') else 0.5
        self.param['C'] = float(self.param['C']) if self.param.get('C') else 1.0
        self.param['kernel'] = self.param['kernel'] if self.param.get('kernel') else "rbf"
        self.param['degree'] = int(self.param['degree']) if self.param.get('degree') else 3
        self.param['coef0'] = float(self.param['coef0']) if self.param.get('coef0') else 0.0
        self.param['shrinking'] = bool(self.param['shrinking']) if self.param.get('shrinking') else True
        self.param['tol'] = float(self.param['tol']) if self.param.get('tol') else 0.001
        self.param['cache_size'] = float(self.param['cache_size']) if self.param.get('cache_size') else 200
        self.param['max_iter'] = int(self.param['max_iter']) if self.param.get('max_iter') else -1
        self.param['gamma'] = self.param['gamma'] if self.param.get('gamma') else "scale"

        self.bootstrapped_number_of_models = int(self.param.pop('bootstrapped_number_of_models', 10))
        self.bootstrapped_weight = self.param.pop('bootstrapped_weight', None)
        self.bootstrapped_fraction = float(self.param.pop('bootstrapped_fraction', 0.9))
        self.bootstrapped_replace = bool(self.param.pop('bootstrapped_replace', False))
        self.bootstrapped_random_state = int(self.param.pop('bootstrapped_random_state', 0))

        self.predicted_results = None

        self.models = [
            NuSVR(**self.param) for _ in range(self.bootstrapped_number_of_models)
        ]

        self.hyper_parameters = {}

    def fit(self, data, label, weight=None):
        print(type(data))
        print(type(label))
        data["label"] = label
        data = data.sample(frac=self.bootstrapped_fraction, replace=self.bootstrapped_replace, random_state=self.bootstrapped_random_state)
        label = data["label"]
        data = data.drop(["label"], axis=1)

        for i in range(self.bootstrapped_number_of_models):
            self.models[i].fit(data, label, sample_weight=None)

    def predict(self, data):

        predicted_results = [
            self.models[i].predict(data)
            for i in range(self.bootstrapped_number_of_models)
        ]

        predicted_results = np.array(predicted_results)
        self.predicted_results = predicted_results

        return predicted_results.mean(axis=0)

    def plot_feature_importance(self, output_file_path):
        pass

    def save(self, output_file_path):
        pass

    def load(self, model_file_path):
        pass

    def get_model(self):
        pass