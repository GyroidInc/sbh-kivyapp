# -*- coding: utf-8 -*-
import os

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


def _parse_label(file):
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
            parsed_labels.append(_parse_label(f))
        except Exception as e:
            raise ValueError("Error trying to convert label %s from file %s to float because %s" % (label, f, e))
    return parsed_labels