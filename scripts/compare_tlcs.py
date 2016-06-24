import argparse
import matplotlib.pyplot as plt
import numpy as np
from pyTLC.TLCdataset import  TLCdataset
from pyTLC.tools import run_database_search 

def get_library_matches(filename, libfile, x_dim):
    tlc_dataset = TLCdataset(filename, x_dim=x_dim)
    feature_list = run_database_search(tlc_dataset, libfile)
    print feature_list
    return feature_list

def compare_tlc_datasets(tlc_dataset_files, libfile, x_dim):
    """
    
    """
    feature_lists = []
    for tlc_dataset_file in tlc_dataset_files:
        feature_list = get_library_matches(tlc_dataset_file, libfile, x_dim)
        print feature_list
        feature_lists.append(feature_list)

    print "len(feature_lists):", len(feature_lists)
    print "len(feature_lists[0]):",len(feature_lists[0]), "type",type(feature_lists[0])
        
    # each feature has: LibraryMZ:(foundMZ,Intensity)
    # Compare only matching libraryMZ
    # Create 2d histogram data

    # 1: Create full list of found mzs (library value)
    library_mzs = feature_lists[0].keys()
    for feature_list in feature_lists[1:]:
        for mz in feature_list.keys():
            if mz not in library_mzs:
                library_mzs.append(mz)

    # figure out how to dimension the numpy array properly
    dims = [len(library_mzs),len(feature_lists)]
    heatmap_data = np.zeros(dims)

    for i,library_mz in enumerate(library_mzs):
        for j,feature_list in enumerate(feature_lists):
            try:
                heatmap_data[i,j] = feature_list[library_mz][1]
            except(KeyError):
                pass
            
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(heatmap_data, cmap = plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(heatmap_data.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(heatmap_data.shape[1])+0.5, minor=False)

    # want a more natural, table-like display
    #ax.invert_yaxis()
    #ax.xaxis.tick_top()

    column_labels = range(len(feature_lists))
    row_labels = [str(mz) for mz in library_mzs]
    print row_labels

    ax.set_xticklabels(column_labels, minor=False)
    ax.set_yticklabels(row_labels, minor=False)
    plt.show()
        
if __name__ == "__main__":
    # do argparsing etc
    help_msg = ('show features for a dataset')
    parser = argparse.ArgumentParser(description=help_msg)
    parser.add_argument('--filenames', help='data filepaths (imzml)', nargs='+')
    parser.add_argument('--libfile', help='library file (.csv)')
    parser.add_argument('--x_dim', help='chomatographic dimension (0=columns, 1=rows)')
    parser.set_defaults(filename='', libfile="../data/swiss_lipids_quick_Species.csv", x_dim=0)
    args = parser.parse_args()
    # files:
    compare_tlc_datasets(args.filenames, args.libfile, args.x_dim)
