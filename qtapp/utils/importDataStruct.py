import pandas as pd
import numpy as np
import collections

class importHolder():
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self):
        self.isEmpty = true
        self.FileFactory = collections.namedtuple("File", "Filepath Label Data")
        self.Files = []
