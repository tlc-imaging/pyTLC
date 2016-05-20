from pyTLC.TLCdataset import  TLCdataset
import matplotlib.pyplot as plt
import argparse
def plot_features(tlc_dataset):
    plt.figure()
    for m,x,i in tlc_dataset.feature_list:
        plt.plot(x, m, '.', color='black')
    plt.xlabel('distance along tlc track')
    plt.ylabel('m/z')
    plt.show()

def show_features(filename):
    tlc_dataset = TLCdataset(filename)
    tlc_dataset.get_features_from_mean_spectrum(ppm=6.)
    plot_features(tlc_dataset)

if __name__ == "__main__":
    help_msg = ('show features for a dataset')
    parser = argparse.ArgumentParser(description=help_msg)
    parser.add_argument('--filename', help='input filename')
    parser.set_defaults(filename='')
    args = parser.parse_args()
    show_features(args.filename)
