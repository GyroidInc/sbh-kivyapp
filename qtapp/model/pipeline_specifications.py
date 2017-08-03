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
import pandas as pd
from sklearn.decomposition import FastICA, PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


try:
    from qtapp.model.builder import ModelBuilder
    from qtapp.utils import helper
except:
    from model.builder import ModelBuilder
    from utils import helper


__description__ = """Functions to handle pipeline specifications on Tab 2: Train Model"""


def get_model(learner_type, model_name, hyperparameters):
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

    return CLASSIFIERS[model_name](**hyperparameters) if learner_type == "Classifier" else \
        REGRESSORS[model_name](**hyperparameters)


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


def cross_validation(X, y, learner_type, model_name, model=None, standardize=True, feature_reduction_method=None,
                     widget_analysis_log=None, save_path=None, configuration_file=None,
                     verbose=False):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    if model is None:
        # Get model based on learning task and model name and instantiate
        model = get_model(learner_type=learner_type, model_name=model_name,
                          hyperparameters= configuration_file["Models"][model_name]["hyperparameters"])

    # Make sure y is flattened to 1d array-like
    if y.ndim == 2:
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        else:
            y = y.ravel()  # assume a numpy array then

    # Update display
    if verbose:
        widget_analysis_log.append("------------------------------")
        widget_analysis_log.append("Training %s using cross-validation method with hyperparameters\n%s" % \
                                   (model_name, (model.get_params(),)))

    # Create 3-fold cross-validation object based on learning task
    scores, fold = np.zeros(3), 0
    cv = KFold(n_splits=3) if learner_type == "Regressor" else StratifiedKFold(n_splits=3)
    for train_index, test_index in cv.split(X, y):

        if verbose:
            widget_analysis_log.append("\n\tFold %d" % (fold+1))

       # Split into train/test and features/labels
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        # Standardize features if specified
        if standardize:
            if verbose:
                widget_analysis_log.append("\tStandardizing features...")
            X_train, scaler = standardize_features(X=X_train)
            X_test = standardize_features(X=X_test, scaler=scaler)
        else:
            scaler = None

        # Reduce features if specified
        if feature_reduction_method:
            if verbose:
                widget_analysis_log.append("\tPerforming feature reduction...")
            X_train, transformer = feature_reduction(X=X_train, learner_type=learner_type, method=feature_reduction_method,
                                                     y=y_train, transformer=None)
            X_test = feature_reduction(X=X_test, learner_type=learner_type, method=feature_reduction_method,
                                       y=None, transformer=transformer)
        else:
            transformer = None

        # Train model
        if verbose:
            widget_analysis_log.append("\tTraining model...")
        model.fit(X_train, y_train)

        # Get predictions and metric on test fold
        scores[fold] = score = helper.calculate_metric(y_true=y_test, y_hat=model.predict(X_test), learner_type=learner_type)
        if verbose:
            widget_analysis_log.append("\tValidation metric: %f" % scores[fold])
        fold += 1

    if verbose:
        widget_analysis_log.append("\n\tOverall validation metric across folds: %f" % scores.mean())

    # Refit on all data now and return parameters
    if verbose:
        widget_analysis_log.append("\tRetraining model on all data...")

    if standardize: scaler = standardize_features(X=X)
    if feature_reduction_method: transformer = feature_reduction(X=X, learner_type=learner_type, method=feature_reduction_method,
                                                                 y=y, transformer=None)

    model.fit(X, y)

    # Package model into an object that holds the trained model, scaler, and transformer
    trained_learner = ModelBuilder(model_name=model_name,
                                   trained_model=model,
                                   trained_scaler=scaler,
                                   trained_transformer=transformer)

    # Save model if specified
    if save_path:
        helper.serialize_trained_model(model_name=model_name,
                                       trained_learner=trained_learner,
                                       path_to_model=save_path,
                                       configuration_file=configuration_file)
        configuration_file["Models"][model_name]["path_trained_learner"] = save_path
        if verbose:
            widget_analysis_log.append("\tTrained learner saved at %s" % save_path)

    # Update configuration file
    configuration_file["Models"][model_name]["clf_trained_learner"] = trained_learner
    configuration_file["Models"][model_name]["validation_score"] = scores.mean()
    configuration_file["Models"][model_name]["hyperparameters"] = model.get_params()

    # If not verbose, then automatically_tune is calling the method and needs return arguments
    if verbose:
        widget_analysis_log.append("\tConfiguration file updated")
        widget_analysis_log.append("\nModel training complete\n")
        widget_analysis_log.append("------------------------------\n")
    else:
        return scores.mean(), model, scaler, transformer


def holdout(X, y, learner_type, model_name, model=None, standardize=True, feature_reduction_method=None,
            widget_analysis_log=None, save_path=None, configuration_file=None, verbose=False):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    if model is None:
        # Get model based on learning task and model name and instantiate
        model = get_model(learner_type=learner_type, model_name=model_name,
                          hyperparameters= configuration_file["Models"][model_name]["hyperparameters"])

    # Make sure y is flattened to 1d array-like
    if y.ndim == 2:
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        else:
            y = y.ravel()  # assume a numpy array then

    # Split into train/test and features/labels (account for stratification if classification task)
    if learner_type == "Regressor":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, stratify=y)

    # Update display
    if verbose:
        widget_analysis_log.append("------------------------------")
        widget_analysis_log.append("Training %s using holdout method with hyperparameters\n%s" % \
                                   (model_name, (model.get_params(),)))

    # Standardize features if specified
    if standardize:
        if verbose:
            widget_analysis_log.append("\tStandardizing features...")
        X_train, scaler = standardize_features(X=X_train)
        X_test = standardize_features(X=X_test, scaler=scaler)
    else:
        scaler = None

    # Reduce features if specified
    if feature_reduction_method:
        if verbose:
            widget_analysis_log.append("\tPerforming feature reduction...")

        X_train, transformer = feature_reduction(X=X_train, learner_type=learner_type, method=feature_reduction_method,
                                                 y=y_train, transformer=None)
        X_test = feature_reduction(X=X_test, learner_type=learner_type, method=feature_reduction_method,
                                   y=None, transformer=transformer)
    else:
        transformer = None

    # Train model
    if verbose:
        widget_analysis_log.append("\tTraining model and calculating validation metric on holdout set...")
    model.fit(X_train, y_train)

    # Get predictions and metric on test fold
    metric = helper.calculate_metric(y_true=y_test, y_hat=model.predict(X_test), learner_type=learner_type)
    if verbose:
        widget_analysis_log.append("\tValidation metric: %f" % metric)

    # Refit on all data now and return parameters
    if verbose:
        widget_analysis_log.append("\tRetraining model on all data...")

    if standardize: scaler = standardize_features(X=X)
    if feature_reduction_method: transformer = feature_reduction(X=X, learner_type=learner_type, method=feature_reduction_method,
                                                                 y=y, transformer=None)
    model.fit(X, y)

    # Package model into an object that holds the trained model, scaler, and transformer
    trained_learner = ModelBuilder(model_name=model_name,
                                   trained_model=model,
                                   trained_scaler=scaler,
                                   trained_transformer=transformer)

    # Save model if specified
    if save_path:
        helper.serialize_trained_model(model_name=model_name,
                                       trained_learner=trained_learner,
                                       path_to_model=save_path,
                                       configuration_file=configuration_file)
        configuration_file["Models"][model_name]["path_trained_learner"] = save_path
        if verbose:
            widget_analysis_log.append("\tTrained learner saved at %s" % save_path)

    # Update configuration file
    if verbose:
        configuration_file["Models"][model_name]["clf_trained_learner"] = trained_learner
        configuration_file["Models"][model_name]["validation_score"] = metric
        configuration_file["Models"][model_name]["hyperparameters"] = model.get_params()

    # If not verbose, then automatically_tune is calling the method and needs return arguments
    if verbose:
        widget_analysis_log.append("\tConfiguration file updated")
        widget_analysis_log.append("\nModel training complete\n")
        widget_analysis_log.append("------------------------------\n")
    else:
        return metric, model, scaler, transformer


def automatically_tune(X, y, learner_type, model_name, standardize=True, feature_reduction_method=None,
                       training_method="holdout", widget_analysis_log=None, status_bar=None, save_path=None,
                       configuration_file=None):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    # Update display
    widget_analysis_log.append("------------------------------")
    widget_analysis_log.append("Automatically tuning hyperparameters for %s using %s method" % \
                               (model_name, training_method))

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
        model = get_model(learner_type=learner_type, model_name=model_name, hyperparameters=current_params)
        current_metric, current_model, current_scaler, current_transformer \
            = trainer(X=X, y=y, learner_type=learner_type, model_name=model_name, model=model,
                      standardize=standardize, feature_reduction_method=feature_reduction_method,
                      widget_analysis_log=widget_analysis_log, save_path=save_path,
                      configuration_file=configuration_file, verbose=False)

        # Compare current_metric to best_metric based on learning task
        if learner_type == "Regressor":
            if current_metric < best_metric:
                best_metric, best_model, best_params, best_scaler, best_transformer = \
                    current_metric, current_model, current_params, current_scaler, current_transformer

                # Update display
                widget_analysis_log.append("Next best model (%s) at combination %d/%d:" % (model_name, n+1, n_combos))
                widget_analysis_log.append("\tValidation metric: %.4f" % best_metric)
                widget_analysis_log.append("\tHyperparameters: %s\n" % best_params)

        else:
            if current_metric > best_metric:
                best_metric, best_model, best_params, best_scaler, best_transformer = \
                    current_metric, current_model, current_params, current_scaler, current_transformer

                # Update display
                widget_analysis_log.append("Next best model (%s) at combination %d/%d:" % (model_name, n+1, n_combos))
                widget_analysis_log.append("\tValidation metric: %.4f" % best_metric)
                widget_analysis_log.append("\tHyperparameters: %s\n" % best_params)

    # Update display
    widget_analysis_log.append("Best model (%s)" % model_name)
    widget_analysis_log.append("\tValidation Metric: %.4f" % best_metric)
    widget_analysis_log.append("\tHyperparameters: %s\n" % best_params)

    # Package model into an object that holds the trained model, scaler, and transformer
    trained_learner = ModelBuilder(model_name=model_name,
                                   trained_model=best_model,
                                   trained_scaler=best_scaler,
                                   trained_transformer=best_transformer)

    # Save model if specified
    if save_path:
        helper.serialize_trained_model(model_name=model_name,
                                       trained_learner=trained_learner,
                                       path_to_model=save_path,
                                       configuration_file=configuration_file)
        configuration_file["Models"][model_name]["path_trained_learner"] = save_path

    # Update configuration file
    configuration_file["Models"][model_name]["clf_trained_learner"] = trained_learner
    configuration_file["Models"][model_name]["validation_score"] = best_metric
    configuration_file["Models"][model_name]["hyperparameters"] = best_params

    widget_analysis_log.append("\tConfiguration file updated\n")
    widget_analysis_log.append("\nModel training complete\n")
    widget_analysis_log.append("------------------------------\n")
