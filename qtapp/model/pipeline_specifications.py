# -*- coding: utf-8 -*-

from __future__ import division

from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, ExpSineSquared, Matern, RationalQuadratic, RBF
from sklearn.model_selection import KFold, ShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

__description__ = """Functions to handle pipeline specifications on Tab 2: Train Model"""


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


def feature_reduction(X, type, method, y=None, transformer=None):
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
            if not y: raise ValueError("Labels array not provided")
            clf = RandomForestRegressor(n_estimators=200) if type == "Regressor" else RandomForestClassifier(n_estimators=200).fit(X, y)
            transformer = SelectFromModel(estimator=clf, threshold="mean", prefit=True)
            return transformer.transform(X), transformer


def cross_validation(X, y, type, model, standardize=True, feature_reduction_method=None, k=3):
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
    cv = KFold(n_splits=k) if type == "Regressor" else StratifiedKFold(n_splits=k)
    for train_index, test_index in cv.split(X, y):

       # Split into train/test and features/labels
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Standardize features if specified
        if standardize:
            X_train, scaler = standardize_features(X=X_train)
            X_test = standardize_features(X=X_test, scaler=scaler)

        # Reduce features if specified
        if feature_reduction_method:
            X_train, transformer = feature_reduction(X=X_train, type=type, method=feature_reduction_method,
                                                     y=y_train, transformer=None)
            X_test = feature_reduction(X=X_test, type=type, method=feature_reduction_method,
                                       y=None, transformer=transformer)

        # Train model
        model.fit(X_train, y_train)

        # Get predictions and metric on test fold
        y_hat = model.predict(X_test)
        scores[fold] = model.score(y_test, y_hat)
        fold += 1

    # Refit on all data now
    # TODO: ADD THIS FUNCTIONALITY

    # Calculate mean and std of metric
    return scores, model


def holdout(X, y, type, test_size=.33):
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
    if type == "Regressor":
        X_train, X_test, y_train, y_test = train_test_split(test_size=test_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(test_size=test_size, stratify=y)

    # Standardize features if specified
    if standardize:
        X_train, scaler = standardize_features(X=X_train)
        X_test = standardize_features(X=X_test, scaler=scaler)

    # Reduce features if specified
    if feature_reduction_method:
        X_train, transformer = feature_reduction(X=X_train, type=type, method=feature_reduction_method,
                                                 y=y_train, transformer=None)
        X_test = feature_reduction(X=X_test, type=type, method=feature_reduction_method,
                                   y=None, transformer=transformer)

    # Train model
    model.fit(X_train, y_train)

    # Get predictions and metric on test fold
    y_hat = model.predict(X_test)
    score = model.score(y_test, y_hat)

    # Refit on all data now
    # TODO: ADD THIS FUNCTIONALITY

    return score, model


def automatically_tune(X, y, type, models, standardize=True, feature_reduction_method=None, training_method="holdout"):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    def get_hyperparameter_grid(model, type):
        """Closure to generate hyperparameter grid for specific models and learning tasks

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
            """ ADD """
            grid = {"kernel": ["rbf", "poly"],
                    "degree": [1, 2, 3],
                    "C": [.001, .01, .1, 1, 10, 100, 1000]}

        else:
            raise ValueError("Model (%s) is not a valid argument" % model)

        return grid

def calculate_metrics(y_true, y_hat, type):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    pass


