from __future__ import division, print_function

import os
import pandas as pd
import shutil


__author__ = "Gyroid, Inc." 
__owner__ = "Smart BioHealth"
__description__ = """I/O functions for reading, saving, and manipulating SansEC data files"""
__all__ = ["DataIngest"]


class DataIngest(object):
    """ADD

    Parameters
    ----------
    file_ext : str (default = .xlsx)
        File extension for SansEC data files

    experiment_dir : str
        Absolute path to experiment directory containing SansEC data files

    experiment_name : str
        Name of experiment

    learning_task : str
        Type of learning task, either regression or classification

    Returns
    -------
    self : object
        Instance of DataIngest
    """
    def __int__(self, file_ext = '.xlsx', experiment_dir = None, experiment_name = None, learning_task = None):
        # Basic error checking
        if not os.path.isdir(experiment_dir): raise IOError("%s not a valid experiment directory" % experiment_dir)        
        files = [f for f in os.listdir(experiment_dir) if f.endswith(file_ext)]
        if len(files) == 0:
            raise IOError("No files with extension %s exist within directory %s" % (file_ext, experiment_dir))

        # Define attributes
        self.file_ext = file_ext
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.learning_task = learning_task
        self.freqz = None
        
        # Load all files into memory and check frequency ranges across files
        overall_min_freq, overall_max_freq = 1e10, 1e-10
        self.data = {}
        for f in files:
            self.data[f] = pd.ExcelFile(os.path.join(experiment_dir, f)).parse('Data', header = 3)
            current_freq = self.data[f]['Freq']
            current_min_freq, current_max_freq = current_freq.min(), current_freq.max()
            if current_min_freq < overall_min_freq:
                overall_min_freq = current_min_freq
            if current_max_freq > overall_max_freq:
                overall_max_freq = current_max_freq
        self.min_freq, self.max_freq = overall_min_freq, overall_max_freq

        # Initialize configuration file
        self.config = {"experiment_name": experiment_name,
                       "learning_task": learning_task,
                       "experiment_dir": experiment_dir}


    def _create_dir_structure(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        pass


    def _create_config(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        pass


    def _select_columns(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        pass


    def _select_freqz(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        pass


    def load(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        pass


    def save(self, file_ext = '.csv'):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        pass

