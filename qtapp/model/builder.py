# -*- coding: utf-8 -*-

import numpy as np

class ModelBuilder(object):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self, model_name, trained_model, trained_scaler=None, trained_transformer=None):
        self.model_name = model_name
        self.trained_model = trained_model
        self.trained_scaler = trained_scaler
        self.trained_transformer = trained_transformer


    def predict(self, X):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        if self.trained_scaler:
            X = self.trained_scaler.transform(X)

        if self.trained_transformer:
            X = self.trained_transformer.transform(X)

        return self.trained_model.predict(X)
