# -*- coding: utf-8 -*-
import os
import pandas as pd


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
        return '-'


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


def load(file = None):
    """ADD

    Parameters
    ----------
    file : add
        add

    Returns
    -------
    """
    return pd.ExcelFile(os.path.join(file)).parse('Data', header = 3)