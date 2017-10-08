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
import json
import numpy as np
import os
import pandas as pd
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


try:
    from qtapp.model.builder import ModelBuilder
    from qtapp.utils import helper
    from qtapp.utils.nonguiwrapper import nongui
except:
    from model.builder import ModelBuilder
    from utils import helper
    from utils.nonguiwrapper import nongui


__description__ = """Functions to handle pipeline specifications on Tab 2: Train Model"""


def get_model(learner_type, model_name, hyperparameters):
    """Returns instantiated machine learning model based on hyperparameters

    Parameters
    ----------
    learner_type : str
        Type of learner, either 'Regressor' or 'Classifier'

    model_name : str
        Name of machine learning model

    hyperparameters : dict
        Dictionary of key/value pairs for hyperparameters

    Returns
    -------
    model : sklearn object
        Instantiated sklearn machine learning model
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
        Input dataset

    scaler : sklearn preprocessor object
        Trained StandardScaler object

    Returns
    -------
    X_transformed : 2d array-like
        Standardized input

    scaler : sklearn preprocessor object
        Trained StandardScaler preprocessor
    """
    # Apply scaler transformation
    if scaler:
        return scaler.transform(X)

    # Create scaler transformation
    else:
        scaler = StandardScaler().fit(X)
        return scaler.transform(X), scaler


def feature_reduction(X, learner_type, method, y=None, transformer=None):
    """Subsets features prior to model training using specified method

    Parameters
    ----------
    X : 2d array-like
        Input dataset

    learner_type : str
        Type of learner, either 'Regressor' or 'Classifier'

    method : str
        Feature reduction method

    y : 1d array-like
        Labels for input dataset

    transformer : sklearn transformer object
        Trained sklearn transformer object

    Returns
    --------
    X_transformed : 2d array-like
        Transformed input dataset

    transformer : sklearn transformer object
        Trained sklearn transformer object
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


def feature_importance_analysis(X, y, configuration_file):
    """Calculates feature importance on training data using Random Forests

    Parameters
    ----------
    X : 2d array-like
        Input dataset

    y : 1d array-like
        Labels for input dataset

    configuration_file : dict
        Configuration file for analysis

    Returns
    --------
    importances : dict
        Top k importances in dictionary with key = feature name and value = feature importance score
    """
    # Open log file for writing
    summary = open(os.path.join(os.path.join(configuration_file["SaveDirectory"], "Summary"), "feature_importance_analysis.txt"), "w")
    summary.write("Feature importance analysis conducted using random forest model\n\n")

    # Define random forest for feature importance analysis and train
    params = {'n_estimators': 200}
    clf = RandomForestClassifier(**params) if configuration_file["LearningTask"] == "Classifier" else RandomForestRegressor(**params)
    clf.fit(X, y)

    # Grab importances and sort
    var_names = X.columns
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Write all results
    summary.write("{:<25}{:<20}\n".format("Rank. Feature Name", "Importance Score"))
    for i in range(X.shape[1]):
        summary.write("{:<25}{:<20.4f}\n".format(str(i+1) + '. ' + var_names[indices[i]],
                                               importances[indices[i]]))
    summary.close()

    # Return top 15 features to print in analysis log
    return var_names[indices[:15]], importances[indices[:15]]


def create_predictions_table(y_test, y_hat, learning_task, files):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    table = PrettyTable()
    if learning_task == "Classifier":
        # Calculate errors
        errors, y_test, y_hat = [], np.array(y_test).astype('int'), np.array(y_hat).astype('int')
        for i in range(y_test.shape[0]):
            if y_test[i] != y_hat[i]:
                errors.append("X")
            else:
                errors.append("-")

    else:
        # Calculate errors
        errors, y_test, y_hat = np.zeros(y_test.shape[0]), np.array(y_test).astype('float'), np.array(y_hat).astype('float')
        for i in range(y_test.shape[0]):
            errors[i] = y_test[i] - y_hat[i]

    # Create table
    table.add_column("Test File", files)
    table.add_column("Observed Label", y_test)
    table.add_column("Predicted Label", y_hat)
    table.add_column("Error", errors)

    return table.get_string()


def summary_model_predictions(y_test, configuration_file):
    """ADD

    Parameters
    ----------
    y : 1d array-like
        Labels for input dataset

    configuration_file : dict
        Configuration file for analysis

    Returns
    --------
    ADD
    """
    # Open log file for writing
    summary = open(os.path.join(os.path.join(configuration_file["SaveDirectory"], "Summary"), "summary_model_predictions.txt"), "w")
    summary.write("Summary of %s model predictions for %d testing samples\n\n" % \
                  (configuration_file["LearningTask"].lower(), len(configuration_file["TestFiles"])))

    # Loop over models and print predictions
    counter, n = 1, y_test.shape[0]
    for model, model_info in configuration_file["Models"].items():
        if model_info["test_score"] is not None:

            # Make PrettyTable
            summary.write("%d. Model Predictions for %s:\n" % (counter, model))
            table = create_predictions_table(y_test=y_test,
                                             y_hat=model_info["y_hat"],
                                             learning_task=configuration_file["LearningTask"],
                                             files=configuration_file["TestFiles"])

            counter += 1
            summary.write(table)

            # Add confusion matrix for classifier
            if configuration_file["LearningTask"] == "Classifier":
                summary.write("\nConfusion Matrix (Rows = True, Columns = Predicted)\n")
                summary.write(np.array_str(confusion_matrix(y_true=y_test, y_pred=model_info["y_hat"])))
                summary.write("\n\nAccuracy: %.4f\n\n" % model_info["test_score"])
            else:
                summary.write("\nMean Squared Error: %.4f\n\n" % model_info["test_score"])
        else:
            continue

    summary.close()


@nongui
def cv(X, y, learner_type, standardize=True, feature_reduction_method=None,
       widget_analysis_log=None, configuration_file=None):
    """3-fold cross-validation for hyperparameter testing

    Parameters
    ----------
    X : 2d array-like
        Input dataset

    y : 1d array-like
        Labels for input dataset

    standardize : bool
        Standardize data or not

    feature_reduction_method : str
        Method of feature reduction

    widget_analysis_log : QTextBrowser
        Widget to output messages in analysis log

    configuration_file : dict
        Configuration file for analysis

    Returns
    -------
    None
    """
    # Loop over models and train
    try:
        for model_name, model_information in configuration_file['Models'].items():
            if model_information['selected']:
                try:
                    # Create save path
                    save_path = os.path.join(os.path.join(configuration_file["SaveDirectory"], "Models"), model_name + '.pkl')

                    # Make sure y is flattened to 1d array-like
                    if y.ndim == 2:
                        if isinstance(y, pd.DataFrame):
                            y = y.values.ravel()
                        else:
                            y = y.ravel()  # assume a numpy array then

                    # Grab model
                    model = get_model(learner_type=learner_type, model_name=model_name,
                                      hyperparameters=model_information["hyperparameters"])

                    # Update display
                    widget_analysis_log.append("Training %s using cross-validation method with hyperparameters\n%s\n" % \
                                               (model_name, (model.get_params(),)))

                    # Create 3-fold cross-validation object based on learning task
                    scores, fold = np.zeros(3), 0
                    cv = KFold(n_splits=3) if learner_type == "Regressor" else StratifiedKFold(n_splits=3)
                    for train_index, test_index in cv.split(X, y):

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
                            widget_analysis_log.append("\tStandardizing features...")
                            X_train, scaler = standardize_features(X=X_train)
                            X_test = standardize_features(X=X_test, scaler=scaler)
                        else:
                            scaler = None

                        # Reduce features if specified
                        if feature_reduction_method:
                            widget_analysis_log.append("\tPerforming feature reduction...")
                            X_train, transformer = feature_reduction(X=X_train, learner_type=learner_type, method=feature_reduction_method,
                                                                     y=y_train, transformer=None)
                            X_test = feature_reduction(X=X_test, learner_type=learner_type, method=feature_reduction_method,
                                                       y=None, transformer=transformer)
                        else:
                            transformer = None

                        # Train model
                        widget_analysis_log.append("\tTraining model...")
                        model.fit(X_train, y_train)

                        # Get predictions and metric on test fold
                        scores[fold] = score = helper.calculate_metric(y_true=y_test, y_hat=model.predict(X_test), learner_type=learner_type)
                        widget_analysis_log.append("\tValidation metric: %f" % scores[fold])
                        fold += 1

                    widget_analysis_log.append("\n\tOverall validation metric across folds: %f" % scores.mean())

                    # Refit on all data now and return parameters
                    widget_analysis_log.append("\tRetraining model on all data...")
                    if standardize:
                        X, scaler = standardize_features(X=X)
                    if feature_reduction_method:
                        X, transformer = feature_reduction(X=X, learner_type=learner_type,
                                                           method=feature_reduction_method,
                                                           y=y, transformer=None)

                    model.fit(X, y)

                    # Package model into an object that holds the trained model, scaler, and transformer
                    trained_learner = ModelBuilder(model_name=model_name,
                                                   trained_model=model,
                                                   trained_scaler=scaler,
                                                   trained_transformer=transformer)

                    # Save model
                    helper.serialize_trained_model(model_name=model_name,
                                                   trained_learner=trained_learner,
                                                   path_to_model=save_path,
                                                   configuration_file=configuration_file)
                    widget_analysis_log.append("\tTrained learner saved at %s" % save_path)

                    # Update configuration file
                    configuration_file["Models"][model_name]["path_trained_learner"] = save_path
                    configuration_file["Models"][model_name]["validation_score"] = scores.mean()

                    # Make sure Gaussian process kernel is converted to a string
                    if model_name == "GaussianProcess":
                        if model.get_params():
                            configuration_file["Models"][model_name]["hyperparameters"] = \
                                model.get_params()['kernel'].__str__()
                        else:
                            # If None provided as kernel argument, default is RBF kernel
                            configuration_file["Models"][model_name]["hyperparameters"] = \
                                'RBF(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0)'
                    else:
                        configuration_file["Models"][model_name]["hyperparameters"] = model.get_params()

                    # If not verbose, then automatically_tune is calling the method and needs return arguments
                    widget_analysis_log.append("\tConfiguration file updated")
                    widget_analysis_log.append("\nModel training complete for %s\n" % model_name)
                    widget_analysis_log.append("------------------------------\n")

                except Exception as e:
                    # Model failed for current hyperparameters
                    if model_name == "GaussianProcess":
                        widget_analysis_log.append("***ERROR: Training model (%s) with hyperparameters %s because %s" % \
                                                   (model_name, model_information["hyperparameters"], str(e)))
                        widget_analysis_log.append("\nTip: Click Set Parameters button and select a kernel\n")
                    else:
                        widget_analysis_log.append("***ERROR: Training model (%s) with hyperparameters %s because %s" % \
                                                   (model_name, model_information["hyperparameters"], str(e)))
                        widget_analysis_log.append("\nTip: Check input data set and try again\n")
                    continue
    except Exception as e:
        return e
    return None

@nongui
def holdout(X, y, learner_type, standardize=True, feature_reduction_method=None,
            widget_analysis_log=None, configuration_file=None):
    """66/33 train/validation holdout split for hyperparameter testing

    Parameters
    ----------
    X : 2d array-like
        Input dataset

    y : 1d array-like
        Labels for input dataset

    standardize : bool
        Standardize data or not

    feature_reduction_method : str
        Method of feature reduction

    widget_analysis_log : QTextBrowser
        Widget to output messages in analysis log

    configuration_file : dict
        Configuration file for analysis

    Returns
    -------
    None
    """
    try:
        # Loop over models and train
        for model_name, model_information in configuration_file['Models'].items():
            if model_information['selected']:
                try:
                    # Create save path
                    save_path = os.path.join(os.path.join(configuration_file["SaveDirectory"], "Models"), model_name  + '.pkl')

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

                    # Grab model
                    model = get_model(learner_type=learner_type, model_name=model_name,
                                      hyperparameters=model_information["hyperparameters"])

                    # Update display
                    widget_analysis_log.append("Training %s using holdout method with hyperparameters\n%s\n" % \
                                               (model_name, (model.get_params(),)))

                    # Standardize features if specified
                    if standardize:
                        widget_analysis_log.append("\tStandardizing features...")
                        X_train, scaler = standardize_features(X=X_train)
                        X_test = standardize_features(X=X_test, scaler=scaler)
                    else:
                        scaler = None

                    # Reduce features if specified
                    if feature_reduction_method:
                        widget_analysis_log.append("\tPerforming feature reduction...")
                        X_train, transformer = feature_reduction(X=X_train, learner_type=learner_type, method=feature_reduction_method,
                                                                 y=y_train, transformer=None)
                        X_test = feature_reduction(X=X_test, learner_type=learner_type, method=feature_reduction_method,
                                                   y=None, transformer=transformer)
                    else:
                        transformer = None

                    # Train model
                    widget_analysis_log.append("\tTraining model and calculating validation metric on holdout set...")
                    model.fit(X_train, y_train)

                    # Get predictions and metric on test fold
                    metric = helper.calculate_metric(y_true=y_test, y_hat=model.predict(X_test), learner_type=learner_type)
                    widget_analysis_log.append("\tValidation metric: %f" % metric)

                    # Refit on all data now and return parameters
                    widget_analysis_log.append("\tRetraining model on all data...")
                    if standardize:
                        X, scaler = standardize_features(X=X)
                    if feature_reduction_method:
                        X, transformer = feature_reduction(X=X, learner_type=learner_type,
                                                           method=feature_reduction_method,
                                                           y=y, transformer=None)
                    model.fit(X, y)

                    # Package model into an object that holds the trained model, scaler, and transformer
                    trained_learner = ModelBuilder(model_name=model_name,
                                                   trained_model=model,
                                                   trained_scaler=scaler,
                                                   trained_transformer=transformer)

                    # Save model
                    helper.serialize_trained_model(model_name=model_name,
                                                   trained_learner=trained_learner,
                                                   path_to_model=save_path,
                                                   configuration_file=configuration_file)
                    widget_analysis_log.append("\tTrained learner saved at %s" % save_path)

                    # Update configuration file
                    configuration_file["Models"][model_name]["path_trained_learner"] = save_path
                    configuration_file["Models"][model_name]["validation_score"] = metric

                    # Make sure Gaussian process kernel is converted to a string
                    if model_name == "GaussianProcess":
                        if model.get_params():
                            configuration_file["Models"][model_name]["hyperparameters"] = \
                                model.get_params()['kernel'].__str__()
                        else:
                            # If None provided as kernel argument, default is RBF kernel
                            configuration_file["Models"][model_name]["hyperparameters"] = \
                                'RBF(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0)'
                    else:
                        configuration_file["Models"][model_name]["hyperparameters"] = model.get_params()

                    widget_analysis_log.append("\tConfiguration file updated")
                    widget_analysis_log.append("\nModel training complete for %s\n" % model_name)
                    widget_analysis_log.append("------------------------------\n")

                except Exception as e:
                    # Model failed for current hyperparameters
                    if model_name == "GaussianProcess":
                        widget_analysis_log.append("***ERROR: Training model (%s) with hyperparameters %s because %s" % \
                                                   (model_name, model_information["hyperparameters"], str(e)))
                        widget_analysis_log.append("\nTip: Click Set Parameters button and select a kernel\n")
                    else:
                        widget_analysis_log.append("***ERROR: Training model (%s) with hyperparameters %s because %s" % \
                                                   (model_name, model_information["hyperparameters"], str(e)))
                        widget_analysis_log.append("\nTip: Check input data set and try again\n")
                    continue
    except Exception as e:
        return e
    return None


def autotune_cv(X, y, learner_type, model=None, standardize=True, feature_reduction_method=None,
                configuration_file=None):
    """Automatically tune a pre-defined grid of hyperparameters using 3-fold cross-validation

    Parameters
    ----------
    X : 2d array-like
        Input dataset

    y : 1d array-like
        Labels for input dataset

    model : object
        Instantiated sklearn model object

    standardize : bool
        Standardize data or not

    feature_reduction_method : str
        Method of feature reduction

    configuration_file : dict
        Configuration file for analysis

    Returns
    -------
    metric : float
        Average metric over 3 folds

    model : object
        Trained machine learning model

    scaler : sklearn preprocessor object
        Trained preprocessor object

    transformer : sklearn transformer object
        Trained transformer object
    """
    # Make sure y is flattened to 1d array-like
    if y.ndim == 2:
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        else:
            y = y.ravel()  # assume a numpy array then

    # Create 3-fold cross-validation object based on learning task
    scores, fold = np.zeros(3), 0
    cv = KFold(n_splits=3) if learner_type == "Regressor" else StratifiedKFold(n_splits=3)
    for train_index, test_index in cv.split(X, y):

       # Split into train/test and features/labels
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:
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
    if standardize:
        X, scaler = standardize_features(X=X)
    if feature_reduction_method:
        X, transformer = feature_reduction(X=X, learner_type=learner_type, method=feature_reduction_method,
                                           y=y, transformer=None)

    model.fit(X, y)
    return scores.mean(), model, scaler, transformer


def autotune_holdout(X, y, learner_type, model=None, standardize=True, feature_reduction_method=None,
                     configuration_file=None):
    """Automatically tune a pre-defined grid of hyperparameters using 66/33 train/validation split

    Parameters
    ----------
    X : 2d array-like
        Input dataset

    y : 1d array-like
        Labels for input dataset

    model : object
        Instantiated sklearn model object

    standardize : bool
        Standardize data or not

    feature_reduction_method : str
        Method of feature reduction

    configuration_file : dict
        Configuration file for analysis

    Returns
    -------
    metric : float
        Validation metric

    model : object
        Trained machine learning model

    scaler : sklearn preprocessor object
        Trained preprocessor object

    transformer : sklearn transformer object
        Trained transformer object
    """
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
    metric = helper.calculate_metric(y_true=y_test, y_hat=model.predict(X_test), learner_type=learner_type)

    # Refit on all data now and return parameters
    if standardize:
        X, scaler = standardize_features(X=X)
    if feature_reduction_method:
        X, transformer = feature_reduction(X=X, learner_type=learner_type, method=feature_reduction_method,
                                           y=y, transformer=None)
    model.fit(X, y)
    return metric, model, scaler, transformer


@nongui
def automatically_tune(X, y, learner_type, standardize=True, feature_reduction_method=None,
                       training_method="holdout", widget_analysis_log=None,
                       configuration_file=None):
    """Automatically tune a pre-defined grid of hyperparameters using 3-fold cross-validation

    Parameters
    ----------
    X : 2d array-like
        Input dataset

    y : 1d array-like
        Labels for input dataset

    learner_type : str
        Type of learner, either 'Regressor' or 'Classifier'

    standardize : bool
        Standardize data or not

    feature_reduction_method : str
        Method of feature reduction

    training_method : str
        Method to train models

    widget_analysis_log : QTextBrowser
        Widget to output messages in analysis log

    configuration_file : dict
        Configuration file for analysis

    Returns
    -------
    None
    """
    try:
        # Loop over models and train
        for model_name, model_information in configuration_file['Models'].items():
            if model_information['selected']:

                # Specify save path
                save_path = os.path.join(os.path.join(configuration_file["SaveDirectory"], "Models"), model_name + '.pkl')

                # Update display
                widget_analysis_log.append("Automatically tuning hyperparameters for %s using %s method\n" % \
                                           (model_name, training_method))

                # Set training method for all models
                trainer = autotune_holdout if training_method == "holdout" else autotune_cv

                # Generate hyperparameter grid
                hp_grid = helper.generate_hyperparameter_grid(model=model_name, learner_type=learner_type)
                hp_names, hp_combos = helper.hyperparameter_combinations(hp_grid)

                # Parameters for current model
                n_combos, best_model, best_params, best_scaler, best_transformer = len(hp_combos), None, None, None, None

                # Set initial metric based on learning task
                # (MSE for regression -> lower is better, AUC for classifier -> higher is better)
                best_metric = 0. if learner_type == "Classifier" else 1e10

                # Iterate over all hyperparameter combos
                n_models_success = 0
                for n in range(n_combos):

                    # Grab current hyperparameter combination
                    current_params = {}
                    for i, hp_name in enumerate(hp_names):
                        current_params[hp_name] = hp_combos[n][i]

                    # Try the current hyperparameter combination
                    try:
                        # Grab metric based on training method
                        model = get_model(learner_type=learner_type, model_name=model_name, hyperparameters=current_params)
                        current_metric, current_model, current_scaler, current_transformer \
                            = trainer(X=X, y=y, learner_type=learner_type, model=model,
                                      standardize=standardize, feature_reduction_method=feature_reduction_method,
                                      configuration_file=configuration_file)

                        # Compare current_metric to best_metric based on learning task
                        if learner_type == "Regressor":
                            if current_metric < best_metric:
                                best_metric, best_model, best_params, best_scaler, best_transformer = \
                                    current_metric, current_model, current_params, current_scaler, current_transformer

                                # Update display
                                n_models_success += 1
                                widget_analysis_log.append("Next best model (%s) at combination %d/%d:" % (model_name, n+1, n_combos))
                                widget_analysis_log.append("\tValidation metric: %.4f" % best_metric)
                                widget_analysis_log.append("\tHyperparameters: %s\n" % best_params)

                        else:
                            if current_metric > best_metric:
                                best_metric, best_model, best_params, best_scaler, best_transformer = \
                                    current_metric, current_model, current_params, current_scaler, current_transformer

                                # Update display
                                n_models_success += 1
                                widget_analysis_log.append("Next best model (%s) at combination %d/%d:" % (model_name, n+1, n_combos))
                                widget_analysis_log.append("\tValidation metric: %.4f" % best_metric)
                                widget_analysis_log.append("\tHyperparameters: %s\n" % best_params)


                    # Catch first error that most likely occurred during training
                    except Exception as e:
                        # Skip current hyperparameter combination
                        widget_analysis_log.append("Error training model (%s) with hyperparameter combination %s because %s\n" %
                                                (model_name, current_params, str(e)))
                        continue

                # If at least one model successfully trained, continue processing
                if n_models_success > 0:
                    # Update display
                    widget_analysis_log.append("Best model (%s)" % model_name)
                    widget_analysis_log.append("\tValidation Metric: %.4f" % best_metric)
                    widget_analysis_log.append("\tHyperparameters: %s\n" % best_params)

                    # Package model into an object that holds the trained model, scaler, and transformer
                    trained_learner = ModelBuilder(model_name=model_name,
                                                   trained_model=best_model,
                                                   trained_scaler=best_scaler,
                                                   trained_transformer=best_transformer)

                    # Save model
                    helper.serialize_trained_model(model_name=model_name,
                                                   trained_learner=trained_learner,
                                                   path_to_model=save_path,
                                                   configuration_file=configuration_file)
                    widget_analysis_log.append("\tTrained learner saved at %s" % save_path)

                    # Update configuration file
                    configuration_file["Models"][model_name]["path_trained_learner"] = save_path
                    configuration_file["Models"][model_name]["validation_score"] = best_metric

                    # Make sure Gaussian process kernel is converted to a string
                    if model_name == "GaussianProcess":
                        if best_model.get_params():
                            configuration_file["Models"][model_name]["hyperparameters"] = \
                                best_model.get_params()['kernel'].__str__()
                        else:
                            # If None provided as kernel argument, default is RBF kernel
                            configuration_file["Models"][model_name]["hyperparameters"] = \
                                'RBF(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0)'
                    else:
                        configuration_file["Models"][model_name]["hyperparameters"] = best_params

                    widget_analysis_log.append("\tConfiguration file updated\n")
                    widget_analysis_log.append("\nModel training complete for %s\n" % model_name)
                    widget_analysis_log.append("------------------------------\n")

                # All models failed for current model_name, create error message
                else:
                    widget_analysis_log.append("***ERROR: 0/%d models successfully trained for %s" % \
                                               (n_combos, model_name))
                    widget_analysis_log.append("\nTip: Check input data set and try again\n")
    except Exception as e:
        return e
    return None

@nongui
def deploy_models(X, y, models_to_test, widget_analysis_log=None, configuration_file=None):
    """Runs trained models on test data set

    Parameters
    ----------
    X : 2d array-like
        Input dataset

    y : 1d array-like
        Labels for input dataset

    models_to_test : dict
        Models to test, where key = model name and value = trained model

    widget_analysis_log : QTextBrowser
        Widget to output messages in analysis log

    configuration_file : dict
        Configuration file for analysis

    Returns
    -------
    None
    """
    try:
        for model_name, clf in models_to_test.items():

            widget_analysis_log.append("Deploying model %s on test data..." % model_name)

            # Get predictions on test set
            y_hat = models_to_test[model_name].predict(X)

            # Grab metrics
            metric = helper.calculate_metric(y, y_hat, learner_type=configuration_file["LearningTask"])
            widget_analysis_log.append("\tMetric: %.4f\n" % metric)

            # Update configuration file
            configuration_file["Models"][model_name]["test_score"] = metric
            configuration_file["Models"][model_name]["y_hat"] = y_hat.tolist() # To list for serializing .json file

        # Automatically try and save configuration file
        try:
            json.dump(configuration_file, open(os.path.join(configuration_file['SaveDirectory'], 'configuration.json'), 'w'))
        except:
            pass
        widget_analysis_log.append("Testing finished. Click Generate Report to obtain analysis summary\n")
    except Exception as e:
        return e
    return None