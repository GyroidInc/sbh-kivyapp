import pandas as pd
import numpy as np
import collections

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
            self.fileName=isSelected
            self.filePath=filePath
            self.label=label
            self.data=data

    def __init__(self):
        self.rowList=[]
        self.colList=[]
        self.files = {}
        self.overall_min_freq, self.overall_max_freq = 1e10, 1e-10

    def update_row_list(self):
       for file in self.files:
           self.rowList = file.data.index.intersect(self.rowList)

    def add_file_manually(self, fileName, filePath, label, data, isSelected=True):
        self.files[fileName]= self._internalFile(filePath=filePath, label=label, data=data, isSelected=isSelected)
        self.update_row_list()

    def load_and_add_file(self):
        pass

