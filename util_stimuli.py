import sys
import os
import numpy as np
import scipy.interpolate
import scipy.signal
import scipy.fftpack


def rms(x):
    '''
    Returns root mean square amplitude of x (raises ValueError if NaN).
    '''
    out = np.sqrt(np.mean(np.square(x)))
    if np.isnan(out):
        raise ValueError('rms calculation resulted in NaN')
    return out


def get_dBSPL(x, mean_subtract=True):
    '''
    Returns sound pressure level of x in dB re 20e-6 Pa (dB SPL).
    '''
    if mean_subtract:
        x = x - np.mean(x)
    out = 20 * np.log10(rms(x) / 20e-6)
    return out


def set_dBSPL(x, dBSPL, mean_subtract=True):
    '''
    Returns x re-scaled to specified SPL in dB re 20e-6 Pa.
    '''
    if mean_subtract:
        x = x - np.mean(x)
    rms_out = 20e-6 * np.power(10, dBSPL/20)
    return rms_out * x / rms(x)


def combine_signal_and_noise(signal, noise, snr, mean_subtract=True):
    '''
    Adds noise to signal with the specified signal-to-noise ratio (snr).
    If snr is finite, the noise waveform is rescaled and added to the
    signal waveform. If snr is positive infinity, returned waveform is
    equal to the signal waveform. If snr is negative inifinity, returned
    waveform is equal to the noise waveform.
    
    Args
    ----
    signal (np.ndarray): signal waveform
    noise (np.ndarray): noise waveform
    snr (float): signal-to-noise ratio in dB
    mean_subtract (bool): if True, signal and noise are first de-meaned
        (mean_subtract=True is important for accurate snr computation)
    
    Returns
    -------
    signal_and_noise (np.ndarray) signal in noise waveform
    '''
    if mean_subtract:
        signal = signal - np.mean(signal)
        noise = noise - np.mean(noise)        
    if np.isinf(snr) and snr > 0:
        signal_and_noise = signal
    elif np.isinf(snr) and snr < 0:
        signal_and_noise = noise
    else:
        rms_noise_scaling = rms(signal) / (rms(noise) * np.power(10, snr / 20))
        signal_and_noise = signal + rms_noise_scaling * noise
    return signal_and_noise


def power_spectrum(x, fs, rfft=True, dBSPL=True):
    '''
    Helper function for computing power spectrum of sound wave.
    
    Args
    ----
    x (np.ndarray): input waveform (Pa)
    fs (int): sampling rate (Hz)
    rfft (bool): if True, only positive half of power spectrum is returned
    dBSPL (bool): if True, power spectrum has units dB re 20e-6 Pa
    
    Returns
    -------
    freqs (np.ndarray): frequency vector (Hz)
    power_spectrum (np.ndarray): power spectrum (Pa^2 or dB SPL)
    '''
    if rfft:
        # Power is doubled since rfft computes only positive half of spectrum
        power_spectrum = 2 * np.square(np.abs(np.fft.rfft(x) / len(x)))
        freqs = np.fft.rfftfreq(len(x), d=1/fs)
    else:
        power_spectrum = np.square(np.abs(np.fft.fft(x) / len(x)))
        freqs = np.fft.fftfreq(len(x), d=1/fs)
    if dBSPL:
        power_spectrum = 10. * np.log10(power_spectrum / np.square(20e-6)) 
    return freqs, power_spectrum


def get_bandlimited_power(x, fs, band=None, rfft=True, dBSPL=True):
    '''
    Helper function for computing power of a signal in a specified frequency band.
    
    Args
    ----
    x (np.ndarray): input waveform (Pa)
    fs (int): sampling rate (Hz)
    band (list): frequency band [low_cutoff, high_cutoff] (Hz)
    rfft (bool): if True, only positive half of spectral domain is used
    dBSPL (bool): if True, returned power is rescaled to dB re 20e-6 Pa
    
    Returns
    -------
    bandlimited_power (float): signal power in frequency band (Pa^2 or dB SPL)
    '''
    if band is None:
        band = [0.0, fs/2]
    assert len(band) == 2, "`band` must have length 2: [low_cutoff, high_cutoff]"
    if rfft:
        X = np.fft.rfft(x) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=1/fs)
        freqs_band_idx = np.logical_and(freqs >= band[0], freqs < band[1])
    else:
        X = np.fft.fft(x) / len(x)
        freqs = np.fft.fftfreq(len(x), d=1/fs)
        freqs_band_idx = np.logical_and(np.abs(freqs) >= band[0], np.abs(freqs) < band[1])
    bandlimited_power = np.sum(np.square(np.abs(X[freqs_band_idx])))
    if rfft:
        # Power is doubled since rfft computes only positive half of spectrum
        bandlimited_power = 2 * bandlimited_power
    if dBSPL:
        bandlimited_power = 10. * np.log10(bandlimited_power / np.square(20e-6))
    return bandlimited_power


def complex_tone(f0,
                 fs,
                 dur,
                 harmonic_numbers=[1],
                 frequencies=None,
                 amplitudes=None,
                 phase_mode='sine',
                 offset_start=True,
                 strict_nyquist=True):
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
    signal (np.ndarray): complex tone
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


def flat_spectrum_noise(fs, dur, dBHzSPL=15.0):
    '''
    Function for generating random noise with a maximally flat spectrum.
    
    Args
    ----
    fs (int): sampling rate of noise (Hz)
    dur (float): duration of noise (s)
    dBHzSPL (float): power spectral density in units dB/Hz re 20e-6 Pa
    
    Returns
    -------
    (np.ndarray): noise waveform (Pa)
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


def modified_uniform_masking_noise(fs, dur, dBHzSPL=15.0, attenuation_start=600.0, attenuation_slope=2.0):
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
    (np.ndarray): noise waveform (Pa)
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
    Helper function converts frequency from Hz to ERB-number scale.
    Glasberg & Moore (1990, Hearing Research) equation 4. The ERB-
    number scale can be defined as the number of equivalent
    rectangular bandwidths below the given frequency (units of the
    ERB-number scale are Cams).
    '''
    return 21.4 * np.log10(0.00437 * freq + 1.0)


def erb2freq(erb):
    '''
    Helper function converts frequency from ERB-number scale to Hz.
    Glasberg & Moore (1990, Hearing Research) equation 4. The ERB-
    number scale can be defined as the number of equivalent
    rectangular bandwidths below the given frequency (units of the
    ERB-number scale are Cams).
    '''
    return (1.0/0.00437) * (np.power(10.0, (erb / 21.4)) - 1.0)


def erbspace(freq_min, freq_max, num):
    '''
    Helper function to get array of frequencies linearly spaced on an
    ERB-number scale.
    
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


def TENoise(fs,
            dur,
            lco=None,
            hco=None,
            dBSPL_per_ERB=70.0):
    '''
    Generates threshold equalizing noise (Moore et al. 1997) in the spectral
    domain with specified sampling rate, duration, cutoff frequencies, and
    level. TENoise produces equal masked thresholds for normal hearing
    listeners for all frequencies between 125 Hz and 15 kHz. Assumption:
    power of the signal at threshold (Ps) is given by the equation,
    Ps = No*K*ERB, where No is the noise power spectral density and K is the
    signal to noise ratio at the output of the auditory filter required for
    threshold. TENoise is spectrally shaped so that No*K*ERB is constant.
    Values for K and ERB are taken from Moore et al. (1997).
    
    Based on MATLAB code last modified by A. Oxenham (2007-JAN-30).
    Modified Python implementation by M. Saddler (2020-APR-21).
    
    Args
    ----
    fs (int): sampling rate in Hz
    dur (float): duration of noise (s)
    lco (float): low cutoff frequency in Hz (defaults to 0.0)
    hco (float): high cutoff frequency in Hz (defaults to fs/2)
    dBSPL_per_ERB (float): level of TENoise is specified in terms of the
        level of a one-ERB-wide band around 1 kHz in units dB re 20e-6 Pa
    
    Returns
    -------
    noise (np.ndarray): noise waveform in units of Pa
    '''
    # Set parameters for synthesizing TENoise
    nfft = int(dur * fs) # nfft = duration in number of samples
    if lco is None:
        lco = 0.0 
    if hco is None:
        hco = fs / 2.0
    
    # K values are from a personal correspondance between B.C.J. Moore
    # and A. Oxenham. A also figure appears in Moore et al. (1997).
    K = np.array([
        [0.0500,   13.5000],
        [0.0630,   10.0000],
        [0.0800,   7.2000],
        [0.1000,   4.9000],
        [0.1250,   3.1000],
        [0.1600,   1.6000],
        [0.2000,   0.4000],
        [0.2500,  -0.4000],
        [0.3150,  -1.2000],
        [0.4000,  -1.8500],
        [0.5000,  -2.4000],
        [0.6300,  -2.7000],
        [0.7500,  -2.8500],
        [0.8000,  -2.9000],
        [1.0000,  -3.0000],
        [1.1000,  -3.0000],
        [2.0000,  -3.0000],
        [4.0000,  -3.0000],
        [8.0000,  -3.0000],
        [10.0000, -3.0000],
        [15.0000, -3.0000],
    ])
    
    # K values are interpolated over rfft frequency vector
    f_interp_K = scipy.interpolate.interp1d(K[:, 0], K[:, 1],
                                            kind='cubic',
                                            bounds_error=False,
                                            fill_value='extrapolate')
    freq = np.fft.rfftfreq(nfft, d=1/fs)
    KdB = f_interp_K(freq / 1000)
    
    # Calculate ERB at each frequency and compute TENoise PSD
    ERB = 24.7 * ((4.37 * freq / 1000) + 1) # Glasberg & Moore (1990) equation 3
    TEN_No = -1 * (KdB + (10 * np.log10(ERB))) # Units: dB/Hz re 1
    
    # Generate random noise_rfft vector and scale to TENoise PSD between lco and hco
    freq_idx = np.logical_and(freq > lco, freq < hco)
    a = np.zeros_like(freq)
    b = np.zeros_like(freq)
    a[freq_idx] = np.random.randn(np.sum(freq_idx))
    b[freq_idx] = np.random.randn(np.sum(freq_idx))
    noise_rfft = a + 1j*b
    noise_rfft[freq_idx] = noise_rfft[freq_idx] * np.power(10, (TEN_No[freq_idx] / 20))
    
    # Estimate power in ERB centered at 1 kHz and compute scale factor for desired dBSPL_per_ERB
    freq_idx_1kHz_ERB = np.logical_and(freq>935.0, freq<1068.1)
    power_1kHz_ERB = 2 * np.sum(np.square(np.abs(noise_rfft[freq_idx_1kHz_ERB]))) / np.square(nfft)
    dBSPL_power_1kHz_ERB = 10 * np.log10(power_1kHz_ERB / np.square(20e-6))
    amplitude_scale_factor = np.power(10, (dBSPL_per_ERB - dBSPL_power_1kHz_ERB) / 20)
    
    # Generate noise signal with inverse rfft, scale to desired dBSPL_per_ERB
    noise = np.fft.irfft(noise_rfft)
    noise = noise * amplitude_scale_factor
    return noise


def get_spectral_envelope_lp_coefficients(x, M=12):
    '''
    Computes "Linear Prediction Coefficients" for spectral envelope extraction.
    Implementation is ported from:
    https://ccrma.stanford.edu/~jos/sasp/Spectral_Envelope_Linear_Prediction.html
    
    Args
    ----
    x (np.ndarray): input waveform (Pa)
    M (int): order of the linear predictor
    
    Returns
    -------
    b_lp (np.ndarray): numerator polynomial coefficients of linear predictor
    a_lp (np.ndarray): denominator polynomial coefficients of linear predictor
    '''
    N = len(x)
    # Compute M-th order autocorrelation function
    rx = np.zeros(M+1)
    for i in range(M+1):
        rx[i] = np.dot(x[0:N-i], x[i:N])
    # Prepare M-by-M Toeplitz covariance matrix
    covmatrix = np.zeros([M, M])
    for i in range(M):
        covmatrix[i, i:M] = rx[0:M-i]
        covmatrix[i:M, i] = rx[0:M-i]
    # Solve "normal equations" for prediction coefficients
    a_coeffs = np.linalg.lstsq(-covmatrix, rx[1:M+1], rcond=None)[0]
    a_lp = np.array([1] + list(a_coeffs)) # LP polynomial A(z)
    b_lp = np.array([1])
    return b_lp, a_lp


def get_spectral_envelope_lp(x, fs, M=12):
    '''
    Computes spectral envelope of a given signal via "Linear Prediction".
    Implementation is ported from:
    https://ccrma.stanford.edu/~jos/sasp/Spectral_Envelope_Linear_Prediction.html
    
    Args
    ----
    x (np.ndarray): input waveform (Pa)
    fs (int): sampling rate (Hz)
    M (int): order of the linear predictor
    
    Returns
    -------
    freqs (np.ndarray): frequency vector (Hz)
    lp_spectral_envelope (np.ndarray): spectral envelope (dB)
    '''
    b_lp, a_lp = get_spectral_envelope_lp_coefficients(x, M=M)
    freqs, h = scipy.signal.freqz(b_lp, a_lp, len(x), fs=fs)
    lp_spectral_envelope = 20 * np.log10(np.abs(h))
    return freqs, lp_spectral_envelope


def get_mfcc(x, M):
    '''
    Compute vector of Mel-frequency cepstral coefficients (mfcc) for a
    given frame (x) and Mel-scale filterbank (M).
    M must have shape [n_fft, n_mels].
    '''
    power_spectrum = np.square(np.abs(np.fft.rfft(x)))
    mel_power_spectrum = np.matmul(M, power_spectrum)
    mfcc = scipy.fftpack.dct(np.log(mel_power_spectrum), norm='ortho')
    return mfcc


def get_power_spectrum_from_mfcc(mfcc, Minv):
    '''
    Compute power spectrum from a given vector of Mel-frequency cepstral
    coefficients (mfcc) and a pseudo-inverse Mel-scale filterbank (Minv).
    M must have shape [n_mels, n_fft].
    '''
    mel_power_spectrum = np.exp(scipy.fftpack.idct(mfcc, norm='ortho'))
    power_spectrum = np.matmul(Minv, mel_power_spectrum)
    power_spectrum[power_spectrum < 0] = 0
    power_spectrum = np.sqrt(power_spectrum)
    return power_spectrum


def impose_power_spectrum(x, power_spectrum):
    '''
    Impose power spectrum in frequency domain by multiplying FFT of a
    frame (x) with square root of the given power_spectrum and applying
    inverse FFT. power_spectrum must have same shape as the rfft of x.
    '''
    x_fft = np.fft.rfft(x, norm='ortho')
    x_fft *= np.sqrt(power_spectrum)
    return np.fft.irfft(x_fft, norm='ortho')
