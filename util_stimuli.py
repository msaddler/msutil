import sys
import os
import h5py
import copy
import collections
import numpy as np


def recursive_dict_merge(dict1, dict2):
    '''
    Returns a new dictionary by merging two dictionaries recursively.
    This function is useful for minimally updating dict1 with dict2.
    '''
    result = copy.deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, collections.Mapping):
            result[key] = recursive_dict_merge(result.get(key, {}), value)
        else:
            result[key] = copy.deepcopy(dict2[key])
    return result


def get_hdf5_dataset_key_list(f_input):
    '''
    Walks hdf5 file and returns list of all dataset keys.
    
    Args
    ----
    f_input (str or h5py.File): hdf5 filename or file object
    
    Returns
    -------
    hdf5_dataset_key_list (list): list of paths to datasets in f_input
    '''
    if isinstance(f_input, str):
        f = h5py.File(f_input, 'r')
    else:
        f = f_input
    hdf5_dataset_key_list = []
    def get_dataset_keys(name, node):
        if isinstance(node, h5py.Dataset):
            hdf5_dataset_key_list.append(name)
    f.visititems(get_dataset_keys)
    if isinstance(f_input, str):
        f.close()
    return hdf5_dataset_key_list


def rms(x):
    '''
    Returns root mean square amplitude of x (raises ValueError if NaN)
    '''
    out = np.sqrt(np.mean(np.square(x)))
    if np.isnan(out): raise ValueError('rms calculation resulted in NaN')
    return out


def set_dBSPL(x, dBSPL):
    '''
    Returns x re-scaled to specified SPL in dB re 20e-6 Pa
    '''
    rms_out = 20e-6 * np.power(10, dBSPL/20)
    return rms_out * x / rms(x)


def combine_signal_and_noise(signal, noise, snr, rms_out=0.02):
    '''
    Combine signal and noise with specified SNR.
    
    Args
    ----
    signal (np array): signal waveform
    noise (np array): noise waveform
    snr (float): signal to noise ratio in dB
    rms_out (float): rms amplitude of returned waveform
    
    Returns
    -------
    signal_and_noise (np array) signal in noise waveform
    '''
    signal = signal / rms(signal)
    noise = noise / rms(noise)
    signal_and_noise = (np.power(10, snr / 20) * signal) + noise
    signal_and_noise = signal_and_noise / rms(signal_and_noise)
    return rms_out * signal_and_noise


def power_spectrum(x, fs, rfft=True, dBSPL=True):
    '''
    Helper function for computing power spectrum of sound wave.
    
    Args
    ----
    x (np array): input waveform (units Pa)
    fs (int): sampling rate (Hz)
    rfft (bool): if True, only positive half of power spectra is returned
    dBSPL (bool): if True, power spectrum is rescaled to dB re 20e-6 Pa
    
    Returns
    -------
    freqs (np array): frequency vector (Hz)
    power_spectrum (np array): power spectrum (Pa^2/Hz or dB/Hz SPL)
    '''
    if rfft:
        power_spectrum = np.square(np.abs(np.fft.rfft(x)))
        freqs = np.fft.rfftfreq(len(x), d=1/fs)
    else:
        power_spectrum = np.square(np.abs(np.fft.fft(x)))
        freqs = np.fft.fftfreq(len(x), d=1/fs)
    power_spectrum = power_spectrum / (fs * len(x)) # Rescale to PSD
    if dBSPL:
        power_spectrum = 10. * np.log10(power_spectrum / np.square(20e-6)) 
    return freqs, power_spectrum


def complex_tone(f0, fs, dur, harmonic_numbers=[1], frequencies=None, amplitudes=None, phase_mode='sine',
                 offset_start=True, strict_nyquist=True):
    '''
    Function generates a complex harmonic tone with specified relative phase
    and component amplitudes.
    
    Args
    ----
    f0 (float): fundamental frequency (Hz)
    fs (int): sampling rate (Hz)
    dur (float): duration of tone (s)
    harmonic_numbers (list or None): harmonic numbers to include in complex tone (sorted lowest to highest)
    frequencies (list or None): frequencies to include in complex tone (sorted lowest to highest)
    amplitudes (list): RMS amplitudes of individual harmonics (None = equal amplitude harmonics)
    phase_mode (str): specify relative phases (`sch` and `alt` assume contiguous harmonics)
    offset_start (bool): if True, starting phase is offset by np.random.rand()/f0
    strict_nyquist (bool): if True, function will raise ValueError if Nyquist is exceeded;
        if False, frequencies above the Nyquist will be silently ignored
    
    Returns
    -------
    signal (np array): complex tone
    '''
    # Time vector has step size 1/fs and is of length int(dur*fs)
    t = np.arange(0, dur, 1/fs)[0:int(dur*fs)]
    if offset_start: t = t + (1/f0) * np.random.rand()
    # Create array of frequencies (function requires either harmonic_numbers or frequencies to be specified)
    if frequencies is None:
        assert harmonic_numbers is not None, "cannot specify both `harmonic_numbers` and `frequencies`"
        harmonic_numbers = np.array(harmonic_numbers).reshape([-1])
        frequencies = harmonic_numbers * f0
    else:
        assert harmonic_numbers is None, "cannot specify both `harmonic_numbers` and `frequencies`"
        frequencies = np.array(frequencies).reshape([-1])
    # Set default amplitudes if not provided
    if amplitudes is None:
        amplitudes = 1/len(frequencies) * np.ones_like(frequencies)
    else:
        assert_msg = "provided `amplitudes` must be same length as `frequencies`"
        assert len(amplitudes) == len(frequencies), assert_msg
    # Create array of harmonic phases using phase_mode
    if phase_mode.lower() == 'sine':
        phase_list = np.zeros(len(frequencies))
    elif (phase_mode.lower() == 'rand') or (phase_mode.lower() == 'random'):
        phase_list = 2*np.pi * np.random.rand(len(frequencies))
    elif (phase_mode.lower() == 'sch') or (phase_mode.lower() == 'schroeder'):
        phase_list = np.pi/2 + (np.pi * np.square(frequencies) / len(frequencies))
    elif (phase_mode.lower() == 'cos') or (phase_mode.lower() == 'cosine'):
        phase_list = np.pi/2 * np.ones(len(frequencies))
    elif (phase_mode.lower() == 'alt') or (phase_mode.lower() == 'alternating'):
        phase_list = np.pi/2 * np.ones(len(frequencies))
        phase_list[::2] = 0
    else:
        raise ValueError('Unsupported phase_mode: {}'.format(phase_mode))
    # Build and return the complex tone
    signal = np.zeros_like(t)
    for f, amp, phase in zip(frequencies, amplitudes, phase_list):
        if f > fs/2:
            if strict_nyquist: raise ValueError('Nyquist frequency exceeded')
            else: break
        component = amp * np.sqrt(2) * np.sin(2*np.pi*f*t + phase)
        signal += component
    return signal


def flat_spectrum_noise(fs, dur, dBHzSPL=15.):
    '''
    Function for generating random noise with a maximally flat spectrum.
    
    Args
    ----
    fs (int): sampling rate of noise (Hz)
    dur (float): duration of noise (s)
    dBHzSPL (float): power spectral density in units dB/Hz re 20e-6 Pa
    
    Returns
    -------
    (np array): noise waveform (Pa)
    '''
    # Create flat-spectrum noise in the frequency domain
    fxx = np.ones(int(dur*fs), dtype=np.complex128)
    freqs = np.fft.fftfreq(len(fxx), d=1/fs)
    pos_idx = np.argwhere(freqs>0).reshape([-1])
    neg_idx = np.argwhere(freqs<0).reshape([-1])
    if neg_idx.shape[0] > pos_idx.shape[0]: neg_idx = neg_idx[1:]
    phases = np.random.uniform(low=0., high=2*np.pi, size=pos_idx.shape)
    phases = np.cos(phases) + 1j * np.sin(phases)
    fxx[pos_idx] = fxx[pos_idx] * phases
    fxx[neg_idx] = fxx[neg_idx] * np.flip(phases, axis=0)
    x = np.real(np.fft.ifft(fxx))
    # Re-scale to specified PSD (in units dB/Hz SPL)
    # dBHzSPL = 10 * np.log10 ( PSD / (20e-6 Pa)^2 ), where PSD has units Pa^2 / Hz
    PSD = np.power(10, (dBHzSPL/10)) * np.square(20e-6)
    A_rms = np.sqrt(PSD * fs/2)
    return A_rms * x / rms(x)


def modified_uniform_masking_noise(fs, dur, dBHzSPL=15., attenuation_start=600., attenuation_slope=2.):
    '''
    Function for generating modified uniform masking noise as described by
    Bernstein & Oxenham, JASA 117-6 3818 (June 2005). Long-term spectrum level
    is flat below `attenuation_start` (Hz) and rolls off at `attenuation_slope`
    (dB/octave) above `attenuation_start` (Hz).
    
    Args
    ----
    fs (int): sampling rate of noise (Hz)
    dur (float): duration of noise (s)
    dBHzSPL (float): power spectral density below attenuation_start (units dB/Hz re 20e-6 Pa)
    attenuation_start (float): cutoff frequency for start of attenuation (Hz)
    attenuation_slope (float): slope in units of dB/octave above attenuation_start
    
    Returns
    -------
    (np array): noise waveform (Pa)
    '''
    x = flat_spectrum_noise(fs, dur, dBHzSPL=dBHzSPL)
    fxx = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d=1/fs)
    dB_attenuation = np.zeros_like(freqs)
    nzidx = np.abs(freqs) > 0
    dB_attenuation[nzidx] = -attenuation_slope * np.log2(np.abs(freqs[nzidx]) / attenuation_start)
    dB_attenuation[dB_attenuation > 0] = 0
    amplitudes = np.power(10, (dB_attenuation/20))
    fxx = fxx * amplitudes
    return np.real(np.fft.ifft(fxx))


def freq2erb(freq):
    '''
    Helper function converts frequency from Hz to number of ERBs.
    Glasberg & Moore (1990, Hearing Research) equation 4.
    '''
    return 21.4 * np.log10(0.00437 * freq + 1.0)


def erb2freq(erb):
    '''
    Helper function converts frequency from number of ERBs to Hz.
    Glasberg & Moore (1990, Hearing Research) equation 4.
    '''
    return (1.0/0.00437) * (np.power(10.0, (erb / 21.4)) - 1.0)


def erbspace(freq_min, freq_max, num):
    '''
    Helper function to get array of frequencies linearly spaced on an ERB scale.
    
    Args
    ----
    freq_min (float): minimum frequency in Hz
    freq_max (float): maximum frequency Hz
    num (int): number of frequencies (length of array)
    
    Returns
    -------
    freqs (np.ndarray): array of ERB-spaced frequencies (lowest to highest) in Hz
    '''
    freqs = np.linspace(freq2erb(freq_min), freq2erb(freq_max), num=num)
    freqs = erb2freq(freqs)
    return freqs
