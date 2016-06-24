from pyTLC.TLCdataset import  TLCdataset
import pyImagingMSpec.smoothing as im_smoothing
import matplotlib.pyplot as plt
import argparse
import numpy as np
from pyMSpec import smoothing
from pyMSpec import centroid_detection

def plot_features(tlc_dataset):
    plt.figure()
    for m,x,i in tlc_dataset.feature_list:
        plt.plot(x, m, '.', color='black')
    plt.xlabel('distance along tlc track')
    plt.ylabel('m/z')
    plt.show()

def do_peak_plot(im, m_s, m_s_f,label):
    # Im should always be presented so that x-axis goes along the tlc track
    plt.figure(figsize=(10,10))
    plt.subplot(211)
    plt.imshow(im)
    plt.subplot(212)
    for x in range(im.shape[0]):
        if x==1:
            plt.plot(im[x,:],color="black", label='data peak')
        else:
            plt.plot(im[x,:],color="black")
    plt.plot(np.mean(im,axis=0),color='blue', label='data mean')
    plt.plot(m_s_f, color='red', label=label)
    m_s_f_c = centroid_detection.gradient(np.asarray(range(np.shape(m_s_f)[0])), np.asarray(m_s_f), min_intensity=1.)
    if not len(m_s_f_c[0]) == 0:
        plt.stem(m_s_f_c[0], m_s_f_c[1])
    plt.xlim((0,im.shape[1]))
    plt.legend()
    plt.show()

def show_features(filename, x_dim=0):
    tlc_dataset = TLCdataset(filename, x_dim=x_dim)
    tlc_dataset.get_features_from_mean_spectrum(ppm=6.)
    plot_features(tlc_dataset)


def show_traces(filename, mzs, ppm=6., x_dim=0):
    tlc_dataset = TLCdataset(filename, x_dim=x_dim)
    for mz in mzs:
        ion_datacube = tlc_dataset.ims_dataset.get_ion_image(mz,[ppm,])
        im_raw=ion_datacube.xic_to_image(0)
        if x_dim==1:
            im_raw = im_raw.T
        im=im_smoothing.median(im_raw,size=3)
        sqrt_im = np.nan_to_num(np.sqrt(im.copy()))
        ## Sqrt Apodization
        m_s = np.nan_to_num(np.sqrt(np.mean(im,axis=0)))
        m_s_f = smoothing.apodization(range(m_s.shape[0]), m_s, w_size=5)[1]
        label = "m/z =" + str(mz)
        do_peak_plot(sqrt_im, m_s, m_s_f,label)

if __name__ == "__main__":
    help_msg = ('show features for a dataset')
    parser = argparse.ArgumentParser(description=help_msg)
    parser.add_argument('--filename', help='data filepath (imzml)')
    parser.add_argument('--x_dim', help='chomatographic dimension (0=columns, 1=rows)')
    parser.set_defaults(filename='', x_dim=0)
    args = parser.parse_args()
    #show_features(args.filename, x_dim=int(args.x_dim))
    mzs = [782.57,522.35255,756.55064,734.56667,697.4744,544.33805,273.03692,723.49394,753.58882,760.58577,766.25255,776.59358,784.51268,740.52314,749.73687,577.52235,793.49703,771.51573,762.49961]
    # mzs = [273.04107 , 291.05207 , 308.07976 , 312.02751 , 313.03333 , 314.0379 , 334.01046 , 335.01617 , 336.021 , 348.19125 , 353.02686 , 375.00944 , 374.00276 , 409.05964 , 449.05103 , 489.04421 , 501.03944 , 511.02808 , 551.01912 , 603.07963 , 625.06136 , 699.03026 , 721.01364 , 727.03112 , 744.9822 , 687.03518]
    #show_traces(args.filename, mzs, ppm=15., x_dim=int(args.x_dim))
    tlc_dataset = TLCdataset(args.filename, x_dim=args.x_dim)
    #for mz in mzs:
    #    tol=mz*15.*1e-6
    #    feature_list = tlc_dataset.get_xic(np.asarray(mz), 15).get_xic(source='features')
    #    print feature_list[0],':',feature_list[1]

    for mz in mzs:
        print mz,":", np.max(tlc_dataset.ims_dataset.get_ion_image(np.asarray([mz,]),15).xic_to_image(0))

    for i,mz in enumerate(mzs):
        print mz,":", np.max(tlc_dataset.ims_dataset.get_ion_image(np.asarray(mzs),15).xic_to_image(i))
