import numpy as np
import math
from scipy import signal, fft, interpolate

def lpfilter_sos(data, dt, cutoff):
    """" Low-pass filter using the second-order representation Butterworth implementation
    Inputs:
        data - 2D numpy array, of shape [channels,samples]
        dt - sampling interval (seconds)
        cutoff - cutoff frequency (Hz)
    Output:
        filtered version of data
    """
    sos = signal.iirfilter(N=4, Wn=[cutoff], btype='lowpass', fs=1/dt, output='sos')
    nch, nt = np.shape(data)
    num_sections = int(np.ceil(nt/100000))
    if num_sections == 1:
        return np.float32(signal.sosfilt(sos, data, axis=-1))
    else:
        win_size = int(nch/num_sections)
        for i in range(num_sections-1):
            data[(i-1)*win_size:i*win_size, :] = np.float32(signal.sosfilt(sos, data[(i-1)*win_size:i*win_size, :],
                                                                           axis=-1))
        data[(num_sections-1)*win_size:nt, :] = np.float32(signal.sosfilt(sos, data[(num_sections-1)*win_size:nt, :],
                                                                          axis=-1))
        return data


def lpfilter(data, dt, cutoff):
    """" Low-pass filter using the Butterworth implementation
    Inputs:
        data - 2D numpy array, of shape [channels,samples]
        dt - sampling interval (seconds)
        cutoff - cutoff frequency (Hz)
    Output:
        filtered version of data
    """
    b, a = signal.butter(N=4, Wn=cutoff, btype='low', fs=1/dt)
    return np.float32(signal.filtfilt(b, a, data, axis=- 1, padtype='odd'))


def bpfilter_sos(data, dt, bp_low, bp_high):
    """" Band-pass filter using the SOS implementation
    Inputs:
        data - 2D numpy array, of shape [channels,samples]
        dt - sampling interval (seconds)
        bp_low - minimal frequency in the passband (Hz)
        bp_high - maximal frequency in the passband (Hz)
    Output:
        filtered version of data
    """
    sos = signal.iirfilter(N=4, Wn=[bp_low, bp_high], btype='bandpass', fs=1/dt, output='sos')
    return np.float32(signal.sosfilt(sos, data, axis=-1))


def bpfilter(data, dt, bp_low, bp_high):
    """" Band-pass filter using the second-order representation Butterworth
    Inputs:
        data - 2D numpy array, of shape [channels,samples]
        dt - sampling interval (seconds)
        bp_low - minimal frequency in the passband (Hz)
        bp_high - maximal frequency in the passband (Hz)
    Output:
        filtered version of data
    """
    b, a = signal.butter(N=4, Wn=[bp_low, bp_high], btype='bandpass', fs=1/dt)
    return np.float32(signal.filtfilt(b, a, data, axis=- 1, padtype='odd'))


def remove_median(data):
    """" Sample-by-sample median removal
    Inputs:
        data - 2D numpy array, of shape [channels,samples]
    Output:
        data after median removal
    """
    data -= np.median(data, axis=0, keepdims=True)
    return data


def clip(data, clip_perc_val):
    """" Data clipping
    Inputs:
        data - 2D numpy array, of shape [channels,samples]
        clip_perc_val - percentile of data the defines the clipping value.
        Data are assumed to have both positive and negative values, and clipping also occurs
        at 100 - clip_perc_val as well to handle negative values
    Output:
        data after median removal
    """
    return np.clip(data, np.percentile(data, 100.0 - clip_perc_val), np.percentile(data, clip_perc_val))


def normalization(data, mode):
    """" Trace-by-trace normalization
    Inputs:
        data - 2D numpy array, of shape [channels,samples]
        mode - normalization type
            'std' : standard deviation of each channel
            'max' : maximum value of each channel
            'L2' : L2 norm of each channel (no mean removal)
            'none' : nothing happens
    Output:
        data after normalization
    """
    live_traces = np.nonzero(np.sum(np.abs(data), axis=-1))
    if mode == 'std':
        data[live_traces, :] = np.divide(data[live_traces, :], np.std(data[live_traces, :], axis=-1, keepdims=True))
    elif mode == 'max':
        data[live_traces, :] = np.divide(data[live_traces, :], np.amax(np.abs(data[live_traces, :]), axis=-1, keepdims=True))
    elif mode == 'L2':
        data[live_traces, :] = np.divide(data[live_traces, :], np.sqrt(np.sum(np.power(data[live_traces, :], 2.0),
                                                                           axis=-1, keepdims=True)))
    elif mode == 'none':
        pass
    else:
        raise NameError
    return data


def linear_fv(data, dx, dt, freqs, vels):
    """" Transform data into a frequency-phase velocity image.
         Note: works correctly for 2D (line) data.
    Inputs:
        data - 2D numpy array, of shape [channels,samples]
        dx - distance between channels
        dt - time sampling interval
        freqs - frequencies (Hz) at which to estimate the transformation
        vels - phase velocities (m/s) at which to estimate the transformation
    Output:
        frequency-phase velocity image at desired [f,v] values
    """
    (nch, nt) = np.shape(data)
    nscanv = np.size(vels)
    nf = 2**(math.ceil(math.log(nt, 2)))
    nk = 2**(math.ceil(math.log(nch, 2)))

    fft_f = np.arange(-nf/2, nf/2)/nf/dt
    fft_k = np.arange(-nk/2, nk/2)/nk/dx

    fk_res = fft.fftshift(fft.fft2(data, s=[nk, nf]))
    fk_res = np.absolute(fk_res)

    ones_arr = np.ones(shape=(nscanv,))
    fv_map = np.zeros(shape=(len(freqs), len(vels)), dtype=np.float32)

    interp_fun = interpolate.interp2d(fft_k, fft_f, fk_res.T)

    for ind, fr in enumerate(freqs):
        fv_map[ind, :] = np.squeeze(interp_fun(np.divide(ones_arr*fr, vels), fr))/(nch*nt)

    return fv_map.T


def template_matching(data, template, threshold=0.0):
    """" Applies template matching with approximated normalization
    Important : this is not a true normalized cross-correlation, which is significantly slower.
    The autocorrelation of the data is computed as an average value.

    Inputs:
        data - 2D numpy array, of shape [channels,samples]
        template - 2D numpy array, of shape [template_channels,template_samples].
                   Each dimension has to be smaller than the matching one in data.
        threshold - minimal cross-correlation value for the function to return a result. Otherwise, returns None
    Output:
        List including [channel,sample,cross-correlation value] obtained at the point of maximal cross-correlation
    """
    corr = np.fft.irfft2(np.fft.rfft2(data) * np.fft.rfft2(np.flip(template), data.shape))
    temp_autocorr = np.sum(template*template)
    data_autocorr = np.sum(data*data) * np.prod(template.shape) / np.prod(data.shape)
    (nxtemp, nttemp) = template.shape
    corr = corr/np.sqrt(temp_autocorr*data_autocorr)
    max_val = np.amax(corr)
    if max_val >= threshold:
        ind = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
        return [ind[0]-nxtemp+1, ind[1]-nttemp+1, max_val]
    else:
        return None
