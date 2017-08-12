# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import pandas as pd
import shutil
import time


__author__ = "Gyroid, Inc." 
__owner__ = "Smart BioHealth"
__description__ = """I/O functions for reading, saving, and manipulating SansEC data files"""
__all__ = ["DataIngest"]


class DataIngest(object):
    """Data ingestion functionality

    Parameters
    ----------
    file_ext : str (default = .xlsx)
        File extension for SansEC data files

    experiment_dir : str
        Absolute path to experiment directory containing SansEC data files

    load_labels_from : str
        How to load ground truth labels. Valid arguments are "filename" or absolute path to .csv file
        containing two columns: (1) name of SansEC file and (2) ground truth label for file

    experiment_name : str
        Name of experiment

    learning_task : str
        Type of learning task, valid arguments are "regression" or "classification"

    Returns
    -------
    self : object
        Instance of DataIngest
    """
    def __init__(self, file_ext = '.xlsx', experiment_dir = None, load_labels_from = None, experiment_name = None, 
                learning_task = None):
        # Basic error checking
        if not os.path.isdir(experiment_dir): raise IOError("%s not a valid experiment directory" % experiment_dir)        
        files = [f for f in os.listdir(experiment_dir) if f.endswith(file_ext)]
        if len(files) == 0:
            raise IOError("No files with extension %s exist within directory %s" % (file_ext, experiment_dir))

        # Define attributes
        self.file_ext = file_ext
        self.experiment_dir = experiment_dir
        self.load_labels_from = load_labels_from
        self.experiment_name = experiment_name
        self.learning_task = learning_task
        self.freqz = None
        
        # Load all files into memory
        overall_min_freq, overall_max_freq = 1e10, 1e-10
        self.data = {}
        for f in files:
            self.data[f] = {'features': self.load(f)}

        # Add ground truth labels for files
        if self.load_labels_from == "filename":
            self._get_labels_from_filenames(files)
        else:
            if not os.path.isfile(self.load_labels_from): raise IOError("%s is not a valid labels file" % self.load_labels_from)
            self._get_labels_from_csvfile()

        # Initialize configuration file
        self.config = {"experiment_name": experiment_name,
                       "learning_task": learning_task,
                       "experiment_dir": experiment_dir,
                       "dataset": os.path.join(self.experiment_dir)}


    def _get_labels_from_filenames(self, files = None):
        """Get labels from the filenames based on format "filename_label.file_ext" -- the relative path is parsed 
        such that the label extracted is the number after the undescore

        Parameters
        ----------
        file : list
            List of relative filenames

        Returns
        -------
        None
        """
        for f in self.data.keys():
            try:
                label = f.split("_")[-1].split(self.file_ext)[0]
            except:
                if self.file_ext == '.xlsx':
                    try:
                        label = f.split("_")[-1].split('.xls')[0]
                    except:
                        raise ValueError("Error trying to parse label from file %s with extension .xlsx, tried again parsing using .xls extension but encountered error" % f)
                raise ValueError("Error trying to parse label from file %s extension %s" % (f, self.file_ext))

            # Try and convert label to float
            try:
                self.data[f]['label'] = float(label)
            except Exception as e:
                raise ValueError("Error trying to convert label %s from file %s to float because %s" % (label, f, e))


    def _get_labels_from_csvfile(self):
        """Get labels from .csv file specified in self.load_labels_from -- the labels file should contain two columns:
        (1) filename = filename for each SansEC file in experiment directory, (2) label = ground truth label for 
        data file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            labels_file = pd.read_csv(self.load_labels_from, delimiter = ',')
        except Exception as e:
            raise IOError("Error reading labels file %s because %s" % (self.load_labels_from, e))

        # Check for header file and handle if missing
        if set(['filename', 'label']) != set(labels_file.columns.tolist()):
            filenames, labels = labels_file.iloc[:, 0], labels_file.iloc[:, 1]
        else:
            filenames, labels = labels_file['filename'], labels_file['label']

        # Iterate over filenames and add label
        for i, f in enumerate(filenames):
            try:
                self.data[f]['label'] = float(labels[i])
            except Exception as e:
                raise ValueError("Error trying to parse label from labels file %s for data set %s because %s" % (self.load_labels_from, f, e))


    def _create_dir_structure(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        pass


    def _create_feature_names(self, freqz = None, cols = None):
        """Creates feature names with naming convention: f2.0_S11 to indicate frequency '2.0' (e.g., MHz) and 
        column 'S11'

        Parameters
        ----------
        freqz : list or numpy array
            Numerical frequencies 

        cols : list or numpy array
            Column names

        Returns
        -------
        features : list
            Feature names
        """
        features = []
        for f in freqz:
            for c in cols:
                features.append("f" + str(f) + "_" + c)
        return features


    def _create_config(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        pass


    def _select_columns(self, cols = None):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        pass


    def _find_unique_freqz(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        pass


    def _find_unique_cols(self):
        """Find intersection of column names across all data files

        Parameters
        ----------
        None 

        Returns
        -------
        unique_names : list
            Unique names across data files
        """
        data_columns = []
        for data in self.data.values():
            data_columns.append(set(data.columns))
        return list(set.intersection(*data_columns))



    def _index_for_freq(self, freqz = None, freq_to_find = None):
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
        return np.argmin(np.abs(freqz - freq_to_find))
        

    def load(self, file = None, data_source = "classic"):
        """ADD

        Parameters
        ----------
        file : add
            add

        data_source : add
            Placeholder for specifying different data inputs

        Returns
        -------
        """
        return pd.ExcelFile(os.path.join(self.experiment_dir, file)).parse('Data', header = 3)


    def save(self, file_ext = '.csv', data = None):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        # Get date and time for save name
        timestr = time.strftime("%Y%m%d-%H%M%S")

        # Create name for save file, save data, and update configuration file
        self.save_path = os.path.join(self.experiment_dir, self.experiment_name + "_" + timestr + ".csv")
        
        # ADD SAVING FUNCTIONALITY -- NEED TO ENSURE FEATURE NAMES ARE CREATED (probably use pandas here)
        try:
            pass
        except:
            pass
