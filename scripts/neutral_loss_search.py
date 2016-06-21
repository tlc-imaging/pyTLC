from pyTLC.TLCdataset import  TLCdataset
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
neutral_losses_pos={'PC':184.0739,
                   'PG': 172.0137,
                   "PA": 97.9769,
                   "PE": 141.0191,
                   "PI": 260.0297,
                   "PS": 185.0089}
lipid_database = "/Users/palmer/Documents/python_codebase/pyTLC/data/swiss_lipids_quick_Species.csv"

def search_headgroup_loss(filename, x_dim):
    from scipy.spatial import KDTree
    mz_tol = 0.005
    rt_tol = 1
    tlc_dataset = TLCdataset(filename, x_dim=x_dim)
    tlc_dataset.get_features_from_mean_spectrum(ppm=6.)
    feature_array = [[f[0], f[1]] for f in tlc_dataset.feature_list]
    kdtree = KDTree(feature_array)
    neutral_losses_rt = {}
    neutral_losses_i = {}
    for head_group in neutral_losses_pos:
        neutral_losses_rt[head_group]=[]
        neutral_losses_i[head_group] = []
    for f in tlc_dataset.feature_list:
        for head_group in neutral_losses_pos.keys():
            neutral_loss_mz = f[0]-neutral_losses_pos[head_group]
            nearest_neighbour = kdtree.query([neutral_loss_mz, f[1]])
            if all([np.abs(nearest_neighbour[0]-neutral_loss_mz < mz_tol), np.abs(f[1]-nearest_neighbour[1]<rt_tol)]):
                neutral_losses_rt[head_group].append(f[1])
                neutral_losses_i[head_group].append(f[2])
    plt.figure()
    for head_group in neutral_losses_rt:
        h = np.histogram(neutral_losses_rt[head_group], bins=tlc_dataset.x_pos, weights=neutral_losses_i[head_group])[0]
        plt.plot(h, label=head_group)
    plt.legend()
    plt.show()


def search_headgroup_loss_database(filename, x_dim):
    from pyTLC.tools import run_database_search
    tlc_dataset = TLCdataset(filename, x_dim=x_dim)
    run_database_search(tlc_dataset)


if __name__ == "__main__":
    help_msg = ('show features for a dataset')
    parser = argparse.ArgumentParser(description=help_msg)
    parser.add_argument('--filename', help='data filepath (imzml)')
    parser.add_argument('--x_dim', help='chomatographic dimension (0=columns, 1=rows)')
    parser.set_defaults(filename='', x_dim=0)
    args = parser.parse_args()
    #search_headgroup_loss(args.filename, int(args.x_dim))
    search_headgroup_loss_database(args.filename, int(args.x_dim))
