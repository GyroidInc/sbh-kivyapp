from itertools import chain, combinations
from functools import reduce
import os
import pandas as pd
import shutil

try:
    from qtapp.utils.nonguiwrapper import nongui
except:
    from utils.nonguiwrapper import nongui


def num_string(n):
    div=n
    string=""
    while div>0:
        module=(div-1)%26
        string=chr(65+module)+string
        div=int((div-module)/26)
    return string


def makeFileFeatures (filepath):
    v = pd.read_excel(io=filepath, skiprows=3, index_col=0)
    entry = {"file": os.path.basename(filepath)}
    entry["index"], entry["columns"] = set(v.index), set(v.columns)
    return entry


def makeDirectoryList (dirpath):
    filenames = os.listdir(dirpath)
    return [makeFileFeatures(os.path.join(dirpath, filename)) for filename in filenames if filename.endswith("xlsx")]


def lvl1_intersection(sets):
    adjacency_list = []
    for i, s1 in enumerate(sets):
        added = False
        for j, s2 in enumerate(adjacency_list):
            if s2[2] == s1["index"] and s2[3] == s1["columns"]:
                s2[1].append(s1["file"])
                added = True
                continue
        if added == False:
            adjacency_list.append( (num_string(len(adjacency_list) + 1), [s1["file"]], s1["index"], s1["columns"]) )
    return adjacency_list


def complete_intersection(lvl1):
    subsets =  chain(*map(lambda x: combinations(lvl1, x), range(1, len(lvl1)+1)))
    subsets = map(lambda x: [*zip(*list(x))], subsets)
    subsets = map(lambda x: ("|".join(x[0]), reduce(set.union, map(set, x[1])), reduce(set.intersection, x[2]),
                             reduce(set.intersection, x[3])), subsets)
    subsets = [*filter(lambda x: len(x[2]) * len(x[3]) > 0, subsets)]

    ##for future: join all sets in power set that are the same. i.e., if A|B and C are the same, it becomes "A|B & C"
    #for a, b in combinations(enumerate(llv), 2):
    #    if a[1][3] == b[1][3] and a[1][2] == b[1][2]:

    return subsets

@nongui
def split_files(dirpath):
    try:
        # Grab directory list and create groupings
        dirlist = makeDirectoryList(dirpath)
        lvl1 = lvl1_intersection(dirlist)
        groupings = complete_intersection(lvl1)

        if not groupings:
            return 0, 'no .xlsx files found'

        # Iterate over file groupings
        KEY = 0; FILES = 1; FREQS = 2; FEATS = 3;
        n_folders = 0; n_files = 0
        for group in groupings:

            # Only aggregate files unconditional on other groupings (e.g., 'A' as opposed to 'A|B')
            if len(group[KEY].split('|')) == 1:

                # Create subdirectory within main directory
                n_folders += 1; n_files += len(group[FILES])
                freqs_str = 'fRange_' + str(min(group[FREQS])) + '-' + str(max(group[FREQS]))
                cols_str = 'nFeats' + str(len(group[FEATS]) - 1) # Subtract 1 for remove frequency column
                subdir = os.path.join(dirpath, freqs_str + '_' + cols_str)
                if not os.path.isdir(subdir): os.mkdir(subdir)

                # Move all files in current grouping into subdirectory
                for file in group[FILES]:
                    shutil.move(os.path.join(dirpath, file), subdir)

            else:
                continue

        return 1, 'Success', n_folders, n_files
    except Exception as e:
        return 0, str(e), n_folders, n_files