# -*- coding: utf-8 -*-
from itertools import product
import numpy as np
import os
import pandas as pd
from PyQt5.QtWidgets import QMessageBox
import shutil
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import label_binarize
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, ExpSineSquared, Matern, RationalQuadratic, RBF


def get_base_filename(file):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    try:
        return os.path.basename(file)
    except:
        return ''


def parse_label(file):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    # Get name and extension of file
    try:
        name, ext = os.path.splitext(file)
        return float(name.split('_')[-1])
    except:
        return ''


def get_labels_from_filenames(files):
    """Get labels from the filenames based on format "filename_label.file_ext" -- the relative path is parsed
    such that the label extracted is the number after the undescore

    Parameters
    ----------
    file : list
        List of relative filenames

    Returns
    -------
    parsed_labels : list
        List of parsed labels
    """
    parsed_labels = []
    for f in files:
        try:
            parsed_labels.append(parse_label(f))
        except Exception as e:
            raise ValueError("Error trying to convert label %s from file %s to float because %s" % (label, f, e))
    return parsed_labels


def find_unique_cols(data_dict):
    """Find intersection of column names across all data files

    Parameters
    ----------

    Returns
    -------
    unique_names : list
        Unique names across data files
    """
    data_columns = []
    for data in data_dict.values():
        if data["features"] is not None:
            data_columns.append(set(data["features"].columns))
    return list(set.intersection(*data_columns))

def find_unique_freqs(data_dict):
    """Find intersection of frequencies names across all data files

    Parameters
    ----------

    Returns
    -------
    unique_freqs : list
        Unique frequencies across data files
    """
    data_freqs = []
    for data in data_dict.values():
        if data["features"] is not None:
            data_freqs.append(set(data['features']['Freq']))
    return list(set.intersection(*data_freqs))


def load(file):
    """ADD

    Parameters
    ----------
    file : add
        add

    Returns
    -------
    """
    return pd.ExcelFile(os.path.join(file)).parse('Data', header=3)


def hyperparameter_combinations(hyperparameter_dict):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    hp_combos = list(product(*[params for params in hyperparameter_dict.values()]))
    hp_names = [name for name in hyperparameter_dict.keys()]
    return hp_names, hp_combos


def generate_hyperparameter_grid(model, learner_type):
    """Generate hyperparameter grid for specific models and learning tasks

    Parameters
    ----------
    model : str
        Name of machine learning model. Valid arguments are: ExtraTrees, RandomForest, GradientBoostedTrees,
        NeuralNetwork, KNearestNeighbor, GaussianProcess, LinearModel, SupportVectorMachine

    learner_type : str
        learner_type of learning task. Valid arguments are: Regressor or Classifier

    Returns
    -------
    grid : dict
        Hyperparameter grid for specified model and learning task
    """
    # Define hyperparameter grids

    ### EXTRA TREES AND RANDOM FORESTS ###
    if model == "ExtraTrees" or model == "RandomForest":
        """ NOTE: Both ExtraTrees and RandomForest have similar hyperparameters
            3 hyperparameters: (1) n_estimators: Number of trees
                               (2) max_features: Number of features to examine per split
                               (3) criterion: Objective function to optimize during training """
        grid = {"n_estimators": [10, 100, 200, 500],
                "max_features": [None, "log2", "auto"]}
        grid['criterion'] = ["mse", "mae"] if learner_type == "Regressor" else ["gini", "entropy"]

    ### GRADIENT BOOSTED TREES ###
    elif model == "GradientBoostedTrees":
        """ ADD """
        grid = {"n_estimators": [100, 500, 1000],
                "learning_rate": [.1, .01, .001],
                "subsample": [1, .8],
                "max_depth": [1, 3, 5]}
        grid['loss'] = ["ls", "huber"] if learner_type == "Regressor" else ["deviance", "exponential"]

    ### GAUSSIAN PROCESSES ###
    elif model == "GaussianProcess":
        """ 1 hyperparameter: (1) kernel: Covariance function that determines shape of prior and posterior 
                                          distribution of the Gaussian process """
        grid = {"kernel": [RBF(), DotProduct(), RationalQuadratic(), ConstantKernel(), Matern(), ExpSineSquared()]}

    ### K-NEAREST NEIGHBORS ###
    elif model == "KNearestNeighbors":
        """ 2 hyperparameters: (1) n_neighbors: Number of nearest neighbors to use for prediction
                               (2) p: Power parameter for Minkowski metric """
        grid = {"n_neighbors": [1, 3, 5],
                "p": [1, 2]}

    ### LINEAR MODELS ###
    elif model == "LinearModel":
        """ 0 hyperparameters """
        grid = {}

    ### NEURAL NETWORKS ###
    elif model == "NeuralNetwork":
        """ 3 hyperparameters : (1) hidden_layer_sizes: Number of hidden neurons
                                (2) learning_rate: Learning rate schedule for weight updates
                                (3) solver: The solver for weight optimization """
        grid = {"hidden_layer_sizes": [(32,), (64,), (100,), (500,), (1000,)],
                "learning_rate": ["constant", "adaptive"],
                "solver": ["adam", "lbfgs", "sgd"]}

    ### SUPPORT VECTOR MACHINES ###
    elif model == "SupportVectorMachine":
        """ 4 hyperparameters : (1) kernel: ADD
                                (2) degree: ADD
                                (3) C: ADD
                                (4) gamma: ADD """
        grid = {"kernel": ["rbf", "poly"],
                "degree": [1, 2, 3],
                "C": [.001, .01, .1, 1, 10, 100, 1000],
                "gamma": ['auto', .0001, .001, .01]}

    else:
        raise ValueError("Model (%s) is not a valid argument" % model)

    return grid


def calculate_metric(y_true, y_hat, learner_type):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    if learner_type == "Regressor":
        return mean_squared_error(y_true=y_true, y_pred=y_hat)
    else:
        if len(set(y_true)) > 2: y_true, y_hat = label_binarize(y_true), label_binarize(y_hat)
        return roc_auc_score(y_true=y_true, y_score=y_hat, average="weighted")


def index_for_freq(freqs, freq_to_find):
    """Returns index of closest frequency

    Parameters
    ----------
    freqz : list or numpy array
        Numerical frequencies

    freq_to_find : float
        Numerical frequency to find index in freqz

    Returns
    -------
    """
    return np.argmin(np.abs(np.asarray(freqs) - freq_to_find))


def generate_feature_names(freqs, columns, idx_freq_ranges):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    names, start, stop = [], idx_freq_ranges[0], idx_freq_ranges[1]
    for c in columns:
        for f in freqs[start:stop+1]:
            names.append(c + '_' + str(f))
    names.append('label')
    return names


def tranpose_and_append_columns(data, freqs, columns, idx_freq_ranges):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    learner_input, labels, start, stop = [], [], idx_freq_ranges[0], idx_freq_ranges[1]

    # Get labels
    for ds in data.values():
        labels.append(float(ds['label']))
    labels = np.array(labels).reshape(-1, 1)

    # Get features
    for ds in data.values():
        tmp = []
        for c in columns:
            tmp.append(ds['features'][c].values[start:stop+1].reshape(1, -1))
        learner_input.append(np.hstack(tmp))
    learner_input, feature_names = np.vstack(learner_input), generate_feature_names(freqs, columns, idx_freq_ranges)
    return pd.DataFrame(np.hstack((learner_input, labels)), columns=feature_names)


def create_directory_structure(save_directory, overwrite, configuration_file):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    SUBDIRECTORIES_TO_CREATE = ['Summary', 'Models']
    # Directory exists and user specifies to overwrite
    if os.path.isdir(save_directory) and overwrite:
        shutil.rmtree(save_directory)

    # Directory exists and user does not want to overwrite (so add random integer at end)
    elif os.path.isdir(save_directory) and not overwrite:
        valid_name = False
        while not valid_name:
            new_save_directory = save_directory + '_' + str(np.random.random_integers(0, 1000000, 1)[0])

            # If new directory created, then break loop and save name
            if not os.path.isdir(save_directory):
                valid_name = True
                break

        # Overwrite save_directory name with new name and update configuration file
        save_directory = new_save_directory
        configuration_file['SaveDirectory'] = save_directory

    # Directory does not exist
    else:
        pass

    # Create simple directory structure
    os.mkdir(save_directory)
    for sub_dirs in SUBDIRECTORIES_TO_CREATE:
        os.mkdir(os.path.join(save_directory, sub_dirs))


def check_categorical_labels(labels):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    unique_labels = set(labels)
    counts = []
    for y in unique_labels:
        counts.append(np.sum(labels == y))
    counts = np.asarray(counts)
    return np.sum(counts < 3)


def messagePopUp(message, informativeText, windowTitle, type, question=False):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    msg = QMessageBox()
    msg.setText(message)
    msg.setInformativeText(informativeText)
    msg.setWindowTitle(windowTitle)

    if type == "warning":
        msg.setIcon(QMessageBox.Warning)
    elif type == "error":
        msg.setIcon(QMessageBox.Critical)
    else:
        msg.setIcon(QMessageBox.Information)

    if question:
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return msg.exec_()
    else:
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


def serialize_trained_model(model_name, trained_learner, path_to_model, configuration_file):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    try:
        joblib.dump(trained_learner, path_to_model)
    except Exception as e:
        messagePopUp(message="Error saving model %s because" % model_name,
                     informativeText=e,
                     windowTitle="Error: Saving Trained Model",
                     type="error")


def create_blank_config():
    return {
        'SaveDirectory': '',
        'ExperimentName': '',
        'LearningTask': '',
        'TrainSamples': '',
        'TrainFeatures': '',
        'Freqs': '',
        'Columns': '',
        'StandardizeFeatures': '',
        'FeatureReduction': '',
        'TrainingMethod': '',
        'AutomaticallyTune': '',
        'SaveModels': '',
        'Models':
            {
                'ExtraTrees':
                    {
                        "hyperparameters": '',
                        "selected": False,
                        "validation_score": '',
                        "clf_trained_learner": '',
                        "path_trained_learner": ''
                    },
                'GaussianProcess':
                    {
                        "hyperparameters": '',
                        "selected": False,
                        "validation_score": '',
                        "clf_trained_learner": '',
                        "path_trained_learner": ''
                    },
                'GradientBoostedTrees':
                    {
                        "hyperparameters": '',
                        "selected": False,
                        "validation_score": '',
                        "clf_trained_learner": '',
                        "path_trained_learner": ''
                    },
                'KNearestNeighbors':
                    {
                        "hyperparameters": '',
                        "selected": False,
                        "validation_score": '',
                        "clf_trained_learner": '',
                        "path_trained_learner": ''
                    },
                'LinearModel':
                    {
                        "hyperparameters": '',
                        "selected": False,
                        "validation_score": '',
                        "clf_trained_learner": '',
                        "path_trained_learner": ''
                    },
                'NeuralNetwork':
                    {
                        "hyperparameters": '',
                        "selected": False,
                        "validation_score": '',
                        "clf_trained_learner": '',
                        "path_trained_learner": ''
                    },
                'RandomForest':
                    {
                        "hyperparameters": '',
                        "selected": False,
                        "validation_score": '',
                        "clf_trained_learner": '',
                        "path_trained_learner": ''
                    },
                'SupportVectorMachine':
                    {
                        "hyperparameters": '',
                        "selected": False,
                        "validation_score": '',
                        "clf_trained_learner": '',
                        "path_trained_learner": ''
                    }
            }

        }