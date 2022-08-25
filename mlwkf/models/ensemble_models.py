
import pickle
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn.ensemble import VotingRegressor
import catboost as cat
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor


class SuperLearnerRegressor(object):

    def __init__(self, iterable=(), **kwargs):

        modeling_functions = [
            tree.DecisionTreeRegressor(),
            ensemble.RandomForestRegressor(n_estimators=10, random_state=1),
            xgb.XGBRegressor(n_estimators=100, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8),
            ensemble.GradientBoostingRegressor(learning_rate=0.1, n_estimators=100),
            lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=100),
            cat.CatBoostRegressor(iterations=20, learning_rate=0.1, depth=12, l2_leaf_reg=0.2, verbose=False),
            svm.SVR(kernel='rbf'),
            neighbors.KNeighborsRegressor(),
            ensemble.BaggingRegressor(n_estimators=100),
            ensemble.ExtraTreesRegressor(n_estimators=100),
        ]
        self.__dict__.update(iterable, **kwargs)
        estimators = []
        weights = []
        for modeling_function in modeling_functions:
            modeling_function_name = str(modeling_function.__class__).split('.')[-1][:-2]
            estimators.append((modeling_function_name, modeling_function))
            weights.append(1)
        self.model = VotingRegressor(estimators=estimators, weights=weights, n_jobs=1, verbose=False)

    def fit(self, data, label, weight=None):

        self.model.fit(data, label)

    def predict(self, data):
        return self.model.predict(data)

    def plot_feature_importance(self, output_file_path):
        pass

    def save(self, output_file_path):
        pickle.dump(self.model, open(output_file_path, 'wb'))

    def load(self, model_file_path):
        self.model = pickle.load(open(model_file_path, 'rb'))


from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import norm, gamma
import numpy as np
import logging
log = logging.getLogger(__name__)


class QuantileGradientBoostingRegressor(object):

    def __init__(self, param=None):

        if param is None:
            param = {}
        self.param = {
            'upper_alpha': 0.95,
            'lower_alpha': 0.05,
            'loss': 'ls',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 1.0,
            'criterion': 'friedman_mse',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.,
            'max_depth': 3,
            'min_impurity_decrease': 0.,
            'min_impurity_split': None,
            'init': None,
            'random_state': None,
            'max_features': None,
            'alpha': 0.9,
            'verbose': 0,
            'max_leaf_nodes': None,
            'warm_start': False,
            'presort': 'deprecated',
            'validation_fraction': 0.1,
            'n_iter_no_change': None,
            'tol': 1e-4,
            'ccp_alpha': 0.0
        }

        for key in param.keys():
            self.param[key] = param[key]

        self.model = QuantileGradientBoosting(**self.param)

    def fit(self, data, label):

        self.model.fit(data, label)

    def predict(self, data):
        return self.model.predict(data)

    def plot_feature_importance(self, output_file_path):
        pass

    def save(self, output_file_path):
        pickle.dump(self.model, open(output_file_path, 'wb'))

    def load(self, model_file_path):
        self.model = pickle.load(open(model_file_path, 'rb'))


class QuantileGradientBoosting(BaseEstimator, RegressorMixin):
    def __init__(self, upper_alpha=0.95, lower_alpha=0.05,
                 loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='deprecated',
                 validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0
                 ):

        # upper_quantile_params
        self.loss = "quantile"
        self.upper_alpha = upper_alpha
        self.lower_alpha = lower_alpha
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.subsample = subsample
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.presort = presort
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha

        self.gb = GradientBoostingRegressor(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state, alpha=alpha, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha
        )
        self.gb_quantile_upper = GradientBoostingRegressor(
            learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha,
            loss="quantile"
        )
        self.gb_quantile_lower = GradientBoostingRegressor(
            learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha,
        )

    @staticmethod
    def collect_prediction(regressor, X_test):
        return regressor.predict(X_test)

    def fit(self, X, y):
        log.info('Fitting gb base model')
        self.gb.fit(X, y)
        log.info('Fitting gb upper quantile model')
        self.gb_quantile_upper.fit(X, y)
        log.info('Fitting gb lower quantile model')
        self.gb_quantile_lower.fit(X, y)

    def predict(self, X):
        return self.predict_dist(X)[0]

    def predict_dist(self, X, interval=0.95):
        Ey = self.gb.predict(X)

        ql_ = self.collect_prediction(self.gb_quantile_lower, X)
        qu_ = self.collect_prediction(self.gb_quantile_upper, X)
        # divide qu - ql by the normal distribution Z value diff between the quantiles, square for variance
        Vy = ((qu_ - ql_) / (norm.ppf(self.upper_alpha) - norm.ppf(self.lower_alpha))) ** 2

        # to make gbm quantile model consistent with other quantile based models
        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu
