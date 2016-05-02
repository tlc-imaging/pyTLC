#### chromatogram smoothing ####
import numpy as np
from pyMSpec import smoothing

# all functions take a TLC image for a single m/z
def fft_filtered(im):
    m_s = np.mean(im, axis=0)
    f = np.fft.fft(m_s)
    freq = np.fft.fftfreq(m_s.shape[-1])
    f[np.abs(freq)>0.1]=0
    m_s_f = np.fft.ifft(f).real
    return m_s_f


def fft_apodization(im):
    ## FFT Apodization
    m_s = np.mean(im, axis=0)
    w=5
    f = np.fft.fft(m_s)
    freq = np.fft.fftfreq(m_s.shape[-1])
    f[np.abs(freq)>0.1]=0
    m_s_f = np.fft.ifft(f).real
    m_s_f_s = smoothing.apodization(range(len(m_s)),m_s_f,w_size=w)[1]
    return m_s_f_s


def sqrt_apodization(im, w=5.):
    ## Sqrt Apodization
    sqrt_im = np.nan_to_num(np.sqrt(im.copy()))
    m_s = np.mean(sqrt_im,axis=0)
    m_s_f = smoothing.apodization(range(sqrt_im.shape[0]) ,m_s, w_size=w)[1]
    return m_s_f

