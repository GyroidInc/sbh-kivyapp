from __future__ import division

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

CLASSIFIERS = {"ExtraTrees": ExtraTreesClassifier,
               "GaussianProcess": GaussianProcessClassifier,
               "GradientBoostedTrees": GradientBoostingClassifier,
               "KNearestNeighbors": KNeighborsClassifier,
               "LinearModel": LogisticRegression,
               "NeuralNetwork": MLPClassifier,
               "RandomForest": RandomForestClassifier,
               "SupportVectorMachine": SVC}

