# -*- coding: utf-8 -*-
from itertools import product
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import label_binarize


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
        return '-'


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
        data_columns.append(set(data['features'].columns))
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
    return pd.ExcelFile(os.path.join(file)).parse('Data', header = 3)


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


def generate_hyperparameter_grid(model, type):
    """Generate hyperparameter grid for specific models and learning tasks

    Parameters
    ----------
    model : str
        Name of machine learning model. Valid arguments are: ExtraTrees, RandomForest, GradientBoostedTrees,
        NeuralNetwork, KNearestNeighbor, GaussianProcess, LinearModel, SupportVectorMachine

    type : str
        Type of learning task. Valid arguments are: Regressor or Classifier

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
        grid['criterion'] = ["mse", "mae"] if type == "Regressor" else ["gini", "entropy"]

    ### GRADIENT BOOSTED TREES ###
    elif model == "GradientBoostedTrees":
        """ ADD """
        grid = {"n_estimators": [100, 500, 1000],
                "learning_rate": [.1, .01, .001],
                "subsample": [1, .8],
                "max_depth": [1, 3, 5]}
        grid['loss'] = ["ls", "huber"] if type == "Regressor" else ["deviance", "exponential"]

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


def calculate_metric(y_true, y_hat, type):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    if type == "Regressor":
        return mean_squared_error(y_true=y_true, y_pred=y_hat)
    else:
        if len(set(y_true)) > 2: y_true, y_hat = label_binarize(y_true), label_binarize(y_hat)
        return roc_auc_score(y_true=y_true, y_score=y_hat, average="weighted")
