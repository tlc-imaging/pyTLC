import numpy as np
import pyTLC.smoothing as tlc_smoothing
from pyImagingMSpec import inMemoryIMS
import pyImagingMSpec.smoothing as im_smoothing
from pyMSpec import centroid_detection


class Xic():
    """
       a data container for a single mass spectrum
       includes methods for signal processing
       """

    def __init__(self, xic=[], xic_features=[]):
        self._x = []
        self._i = []
        self._c_x = []
        self._c_i = []
        self._processing = []
        if xic != []:
            self._x, self._i = xic
        if xic_features != []:
            self._c_x, self._c_i= xic_features

    # Private basic spectrum I/O
    def __add_xic_x(self, mzs):
        self._x = mzs

    def __add_xic_intensities(self, intensities):
        self._i = intensities

    def __get_mzs(self):
        return np.asarray(self._x)

    def __get_intensities(self):
        return np.asarray(self._i)

    def __get_features_x(self):
        return np.asarray(self._c_x)

    def __get_features_intensity(self):
        return np.asarray(self._c_i)

    def __add_features_x(self, mz_list):
        self._c_x = mz_list

    def __add_features_intensities(self, intensity_list):
        self._c_i = intensity_list

    # Public methods
    def add_spectrum(self, mzs, intensities):
        if len(mzs) != len(intensities):
            raise IOError("mz/intensities vector different lengths")
        self.__xic_x(mzs)
        self.__add_xic_intensities(intensities)

    def add_features(self, mz_list, intensity_list):
        if len(mz_list) != len(intensity_list):
            raise IOError("mz/intensities vector different lengths")
        self.__add_features_x(mz_list)
        self.__add_features_intensities(intensity_list)

    def get_xic(self, source='profile'):
        if source == 'profile':
            mzs = self.__get_mzs()
            intensities = self.__get_intensities()
        elif source == 'features':
            mzs = self.__get_features_x()
            intensities = self.__get_features_intensity()
        else:
            raise IOError('spectrum source should be profile or features')
        return mzs, intensities

    def normalise_spectrum(self, method="tic", method_args={}):
        from pyMSpec import normalisation
        self._centroids_intensity = normalisation.apply_normalisation(self._centroids_intensity, method)
        self._intensities = normalisation.apply_normalisation(self._intensities, method)
        self._processing.append(method)
        return self

    def smooth_spectrum(self, method="sg_smooth", method_args={}):
        from pyMSpec import smoothing
        self._mzs, self._intensities = smoothing.apply_smoothing(self._mzs, self._intensities, method, method_args)
        self._processing.append(method)
        return self


class TLCdataset():
    def __init__(self, filename, xpos=[], x_dim=0):
        self.ims_dataset = inMemoryIMS.inMemoryIMS(filename)
        self.set_dim(x_dim)
        tic_im = self.ims_dataset.get_summary_image().xic_to_image(0)
        self.tic = np.sum(tic_im, axis=self.y_dim)
        if xpos == []:
            self.x_pos = range(np.shape(tic_im)[x_dim])

    def set_dim(self,x_dim):
        if x_dim == 0:
            self.x_dim = 0
            self.y_dim = 1
        else:
            self.x_dim = 1
            self.y_dim = 0

    def get_xic(self, mz, tol):
        mz = np.asarray(mz)
        tol = np.asarray(tol)
        im = self.ims_dataset.get_ion_image(mz, tol).xic_to_image(0)
        im = im_smoothing.median(im, size=3)
        xic = tlc_smoothing.sqrt_apodization(im)
        xic = Xic(xic=[self.x_pos, xic])
        return xic


    def get_features_from_mean_spectrum(self, ppm=3., w=5., min_int=15., max_peaks=3):
        # Generate mean spectrum, go through all peaks and score TLC feature presence
        mean_spec = self.ims_dataset.generate_summary_spectrum(ppm=ppm)
        mean_spec_c = centroid_detection.gradient(np.asarray(mean_spec[0]), np.asarray(mean_spec[1]), min_intensity=3.)
        ion_datacube = self.ims_dataset.get_ion_image(mean_spec_c[0], ppm)
        self.feature_list=[]
        for ii, m in enumerate(mean_spec_c[0]):
            im = ion_datacube.xic_to_image(ii)
            im = im_smoothing.median(im, size=3)
            m_s_f_s = tlc_smoothing.sqrt_apodization(im, w=w)
            m_s_f_s_c = centroid_detection.gradient(np.asarray(range(len(m_s_f_s))), np.asarray(m_s_f_s),
                                                    min_intensity=min_int)
            n_peaks = len(m_s_f_s_c[0])
            if all((n_peaks<max_peaks,)):
                for x, i in zip(m_s_f_s_c[0],m_s_f_s_c[1]):
                    self.feature_list.append((m,x,i))

