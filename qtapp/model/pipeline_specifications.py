# -*- coding: utf-8 -*-

from __future__ import division

# Models
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, \
                             ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

# Other imports
import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, ExpSineSquared, Matern, RationalQuadratic, RBF
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


try:
    from qtapp.utils import helper
except:
    from utils import helper


__description__ = """Functions to handle pipeline specifications on Tab 2: Train Model"""


def get_model(learner_type, model_name):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    CLASSIFIERS = {"ExtraTrees": ExtraTreesClassifier,
                   "GaussianProcess": GaussianProcessClassifier,
                   "GradientBoostedTrees": GradientBoostingClassifier,
                   "KNearestNeighbors": KNeighborsClassifier,
                   "LinearModel": LogisticRegression,
                   "NeuralNetwork": MLPClassifier,
                   "RandomForest": RandomForestClassifier,
                   "SupportVectorMachine": SVC}

    REGRESSORS = {"ExtraTrees": ExtraTreesRegressor,
                  "GaussianProcess": GaussianProcessRegressor,
                  "GradientBoostedTrees": GradientBoostingRegressor,
                  "KNearestNeighbors": KNeighborsRegressor,
                  "LinearModel": LinearRegression,
                  "NeuralNetwork": MLPRegressor,
                  "RandomForest": RandomForestRegressor,
                  "SupportVectorMachine": SVR}

    return CLASSIFIERS[model_name] if learner_type == "Classifier" else REGRESSORS[model_name]


def standardize_features(X, scaler=None):
    """Standardizes each feature (column) to have mean 0 and variance 1

    Parameters
    ----------
    X : 2d array-like
        ADD

    scaler : sklearn preprocessor
        ADD

    Returns
    -------
    """
    # Apply scaler transformation
    if scaler:
        return scaler.transform(X)

    # Create scaler transformation
    else:
        scaler = StandardScaler().fit(X)
        return scaler.transform(X), scaler


def feature_reduction(X, learner_type, method, y=None, transformer=None):
    """ADD

    Parameters
    ----------

    Returns
    --------
    """
    # Apply feature reduction transformer
    if transformer:
        return transformer.transform(X)

    # Create feature reduction transformer
    else:
        # Using PCA
        if method == "PCA":
            # Fit n_components = X.shape[1] and then determine how many components to keep based on explained variance
            transformer = PCA().fit(X)

            # n_components is set equal to the number of components where the explained variance is ~80%
            threshold = .80
            n_components = np.argmin(np.abs(transformer.explained_variance_ratio_ - threshold)) + 1  # add 1 since 0 indexed

            # NOTE: For now, if n_components is larger than number of columns in X, set n_components = X.shape[1] - 1)
            if n_components >= X.shape[1]: n_components = X.shape[1] - 1

            # Refit PCA model using n_components
            transformer = PCA(n_components=n_components).fit(X)
            return transformer.transform(X), transformer

        # Use mean feature importance from a random forest (i.e., features with importance < mean importance are dropped)
        else:
            if y is None: raise ValueError("Labels array not provided")
            clf = RandomForestRegressor(n_estimators=200).fit(X, y) if learner_type == "Regressor" else RandomForestClassifier(n_estimators=200).fit(X, y)
            transformer = SelectFromModel(estimator=clf, threshold="mean", prefit=True)
            return transformer.transform(X), transformer


def cross_validation(X, y, learner_type, model, standardize=True, feature_reduction_method=None, k=3):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    # Make sure y is flattened to 1d array-like
    if y.ndim == 2:
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        else:
            y = y.ravel()  # assume a numpy array then

    # Create k-fold cross-validation object based on learning task
    scores, fold = np.zeros(k), 0
    cv = KFold(n_splits=k) if learner_type == "Regressor" else StratifiedKFold(n_splits=k)
    for train_index, test_index in cv.split(X, y):

       # Split into train/test and features/labels
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Standardize features if specified
        if standardize:
            X_train, scaler = standardize_features(X=X_train)
            X_test = standardize_features(X=X_test, scaler=scaler)
        else:
            scaler = None

        # Reduce features if specified
        if feature_reduction_method:
            X_train, transformer = feature_reduction(X=X_train, learner_type=learner_type, method=feature_reduction_method,
                                                     y=y_train, transformer=None)
            X_test = feature_reduction(X=X_test, learner_type=learner_type, method=feature_reduction_method,
                                       y=None, transformer=transformer)
        else:
            transformer = None

        # Train model
        model.fit(X_train, y_train)

        # Get predictions and metric on test fold
        scores[fold] = score = helper.calculate_metric(y_true=y_test, y_hat=model.predict(X_test), learner_type=learner_type)
        fold += 1

    # Refit on all data now and return parameters
    if standardize: scaler = standardize_features(X=X)
    if feature_reduction_method: transformer = feature_reduction(X=X, learner_type=learner_type, method=feature_reduction_method,
                                                                 y=y, transformer=None)

    model.fit(X, y)
    return scores.mean(), model, scaler, transformer


def holdout(X, y, learner_type, model, standardize=True, feature_reduction_method=None, test_size=.33):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    # Make sure y is flattened to 1d array-like
    if y.ndim == 2:
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        else:
            y = y.ravel()  # assume a numpy array then

    # Split into train/test and features/labels (account for stratification if classification task)
    if learner_type == "Regressor":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

    # Standardize features if specified
    if standardize:
        X_train, scaler = standardize_features(X=X_train)
        X_test = standardize_features(X=X_test, scaler=scaler)
    else:
        scaler = None

    # Reduce features if specified
    if feature_reduction_method:
        X_train, transformer = feature_reduction(X=X_train, learner_type=learner_type, method=feature_reduction_method,
                                                 y=y_train, transformer=None)
        X_test = feature_reduction(X=X_test, learner_type=learner_type, method=feature_reduction_method,
                                   y=None, transformer=transformer)
    else:
        transformer = None

    # Train model
    model.fit(X_train, y_train)

    # Get predictions and metric on test fold
    score = helper.calculate_metric(y_true=y_test, y_hat=model.predict(X_test), learner_type=learner_type)

    # Refit on all data now and return parameters
    if standardize: scaler = standardize_features(X=X)
    if feature_reduction_method: transformer = feature_reduction(X=X, learner_type=learner_type, method=feature_reduction_method,
                                                                 y=y, transformer=None)
    model.fit(X, y)
    return score, model, scaler, transformer


def automatically_tune(X, y, learner_type, model, model_name, standardize=True, feature_reduction_method=None,
                       training_method="holdout", widget_analysis_log=None):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    print(widget_analysis_log)
    # Set training method for all models
    trainer = holdout if training_method == "holdout" else cross_validation

    # Generate hyperparameter grid
    hp_grid = helper.generate_hyperparameter_grid(model=model_name, learner_type=learner_type)
    hp_names, hp_combos = helper.hyperparameter_combinations(hp_grid)

    # Parameters for current model
    n_combos, best_model, best_params, best_scaler, best_transformer = len(hp_combos), None, None, None, None

    # Set initial metric based on learning task
    # (MSE for regression -> lower is better, AUC for classifier -> higher is better)
    best_metric = 0. if learner_type == "Classifier" else 1e10

    # Iterate over all hyperparameter combos
    for n in range(n_combos):

        # Grab current hyperparameter combination
        current_params = {}
        for i, hp_name in enumerate(hp_names):
            current_params[hp_name] = hp_combos[n][i]

        # Grab metric based on training method
        current_metric, current_model, current_scaler, current_transformer \
            = trainer(X=X, y=y, learner_type=learner_type, model=model(**current_params),
                      standardize=standardize, feature_reduction_method=feature_reduction_method)

        # Compare current_metric to best_metric based on learning task
        if learner_type == "Regressor":
            if current_metric < best_metric:
                best_metric, best_model, best_params, best_scaler, best_transformer = \
                    current_metric, current_model, current_params, current_scaler, current_transformer
                print("Next best model (%s) at combination %d/%d:" % (model_name, n+1, n_combos))
                print("\tValidation Metric: %.4f" % best_metric)
                print("\tHyperparameters: %s\n" % best_params)

                # Update display
                widget_analysis_log.append("Next best model (%s) at combination %d/%d:" % (model_name, n+1, n_combos))
                widget_analysis_log.append("\tValidation metric: %.4f" % best_metric)
                widget_analysis_log.append("\tHyperparameters: %s\n" % best_params)

        else:
            if current_metric > best_metric:
                best_metric, best_model, best_params, best_scaler, best_transformer = \
                    current_metric, current_model, current_params, current_scaler, current_transformer
                print("Next best model (%s) at combination %d/%d:" % (model_name, n+1, n_combos))
                print("\tValidation metric: %.4f" % best_metric)
                print("\tHyperparameters: %s\n" % best_params)

                # Update display
                widget_analysis_log.append("Next best model (%s) at combination %d/%d:" % (model_name, n+1, n_combos))
                widget_analysis_log.append("\tValidation metric: %.4f" % best_metric)
                widget_analysis_log.append("\tHyperparameters: %s\n" % best_params)

    # Update display
    widget_analysis_log.append("Best model (%s)" % model_name)
    widget_analysis_log.append("\tValidation Metric: %.4f" % best_metric)
    widget_analysis_log.append("\tHyperparameters: %s\n" % best_params)

    return best_metric, best_model, best_scaler, best_transformer


if __name__ == "__main__":

    # Simple tests
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor

    ### Classifier ###

    # Generate data
    X, y = np.random.normal(0, 1, (100, 30)), np.random.binomial(1, .5, 100)

    # Create two simple models
    models = {"ExtraTrees": ExtraTreesClassifier, 'RandomForest': RandomForestClassifier}

    # Iterate over standardization, feature reduction, and training method
    for std in [True, False]:
        for fr in ["PCA", "FR", None]:
            for tr in ["holdout", "cv"]:
                automatically_tune(X=X,
                                   y=y,
                                   learner_type="Classifier",
                                   models=models,
                                   standardize=std,
                                   feature_reduction_method=fr,
                                   training_method=tr)

    ### Regressor ###

    # Generate data
    X, y = np.random.normal(0, 1, (100, 30)), np.random.normal(0, 1, 100)

    # Create two simple models
    models = {"ExtraTrees": ExtraTreesRegressor, 'RandomForest': RandomForestRegressor}

    # Iterate over standardization, feature reduction, and training method
    for std in [True, False]:
        for fr in ["PCA", "FR", None]:
            for tr in ["holdout", "cv"]:
                automatically_tune(X=X,
                                   y=y,
                                   learner_type="Regressor",
                                   models=models,
                                   standardize=std,
                                   feature_reduction_method=fr,
                                   training_method=tr)

    print("Pseudo-Tests passed")