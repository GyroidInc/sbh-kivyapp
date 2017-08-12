# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import collections
import os.path, glob

class importHolder(object):
    """

    Parameters
    ----------

    Returns
    -------
    """
    class _internalFile(object):
        """ADD

         Parameters
         ----------

         Returns
         -------
         """
        def __init__(self, filePath, label, data, isSelected=True):
            self.isSelected=isSelected
            self.filePath=filePath
            self.label=label
            self.data=data

    def __init__(self):

        #TODO convert shared indices and columns to dict with bools as values to represent selection.
        self.sharedInd={}
        self.sharedCol={}
        self.files = {}
        self.overall_min_freq, self.overall_max_freq = 1e10, 1e-10

    def update_ind_list(self):
       inx = (next(iter(self.files.values()))).data.index
       for file in self.files.values():
           inx = file.data.index & inx

       self.sharedInd = [inx]


    def update_col_list(self):
        col = (next(iter(self.files.values()))).data.columns
        for file in self.files.values():
            inx = file.data.columns & col
        self.sharedCol = [col]

    def remove_file(self, fileName=""):
        self.files[fileName].pop()
        self.update_ind_list()
        self.update_col_list()

    def add_file_manually(self, fileName, filePath, label, data, isSelected=True):
        self.files[fileName]= self._internalFile(filePath=filePath, label=label, data=data, isSelected=isSelected)
        self.update_ind_list()
        self.update_col_list()

    def load_add_excel(self, filepath):
        df = pd.read_excel(io=filepath, skiprows=3, index_col=0)
        filename = os.path.basename(filepath)
        label=""
        self.add_file_manually(fileName=filename, filePath=filepath, label=label, data=df)

    def load_add_csv(self, filepath):
        #TODO Make sure this works/fix

        df = pd.read_csv(io=filepath, skiprows=3, index_col=0)
        filename = os.path.basename(filepath)
        label=""
        self.add_file_manually(fileName=filename, filePath=filepath, label=label, data=df)




    def load_dir(self, path):
        os.chdir(path)
        path = os.path.abspath(path)

        types=("*.csv", "*.xlsx", "*.xls")
        files = []
        for type in types:
            files.extend(glob.glob(type))
        if files:
            for file in files:
                file = os.path.abspath(".\\" + file)
                if file.endswith(".xlsx") or file.endswith(".xls"):
                    self.load_add_excel(filepath=file)
                else:
                    self.load_add_csv(filepath=file)

        else:
            raise IOError("No files found in{}".format(path))






