import tensorflow as tf
import os
import argparse
import time as time_lib
import numpy as np
from subprocess import Popen
import math
from scipy.io import wavfile 
from scipy import signal
from math import log10


def readings(path, ext = ".wav"):
    #reads all the files in a given directory with a given extension
    files = []
    for filename in os.listdir(path):
        if filename.endswith(ext):
             files.append(f"{path}/{filename}")
        else:
            None
    return files

files=readings('yes_no')


def MFCCS_slow(files, plot_flag=False):
    times_slow = []
    mfccs_slow = []
    num_mel_bins = 40
    lower_frequency = 20
    upper_frequency = 4000
    sampling_rate = 16000

    for index, f in enumerate(files):
        start_time = time_lib.time()
        input_rate, audio = wavfile.read(f)
        # build the MFCC
        # we need STFT -> short time fourier transform
        tf_audio = tf.convert_to_tensor(audio, np.float32)
        stft = tf.signal.stft(tf_audio, \
                              frame_length=int(0.016 * input_rate), \
                              frame_step=int(0.008 * input_rate), \
                              fft_length=int(0.016 * input_rate))
        spectrogram = tf.abs(stft)
        
        num_spectrogram_bins = spectrogram.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins,
                                                                                num_spectrogram_bins,
                                                                                sampling_rate,  # 16000 \
                                                                                lower_frequency,
                                                                                upper_frequency)
       
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :10]
        end_time = time_lib.time()
        times_slow.append(end_time - start_time)
        mfccs_slow.append(mfccs)

    return times_slow, mfccs_slow


def MFCCS_fast(files, mel, low, high, sampling_ratio):
    num_mel_bins = mel
    lower_frequency = low  
    upper_frequency = high  
    sampling_rate = 16000  
    times_fast = []
    mfccs_fast = []

    for index, f in enumerate(files):
        start_time = time_lib.time()
        input_rate, audio = wavfile.read(f)
        audio = signal.resample_poly(audio, 1, sampling_ratio)
        tf_audio = tf.convert_to_tensor(audio, dtype=np.float32)
        

        sampling_rate = int(input_rate / sampling_ratio)

        stft = tf.signal.stft(tf_audio, \
                              frame_length=int(0.016 * sampling_rate), \
                              frame_step=int(0.008 * sampling_rate), \
                              fft_length=int(0.016 * sampling_rate))
        # we compute the spectrogram of stft
        spectrogram = tf.abs(stft)
        
        # compute the MFCC
        
        num_spectrogram_bins = spectrogram.shape[-1]
        if index == 0:
            num_spectrogram_bins = spectrogram.shape[-1]
            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins,
                                                                                num_spectrogram_bins,
                                                                                sampling_rate,  # 16000 \
                                                                                lower_frequency,
                                                                                upper_frequency)

        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :10]
        end_time = time_lib.time()
        times_fast.append(end_time - start_time)
        mfccs_fast.append(mfccs)
       
    return times_fast, mfccs_fast

def SNR(fast, slow):
    # compute the Signal to Noise ratio between MFCCS_fast and MFCCS_slow
    snr = 20 * math.log10((np.linalg.norm(slow)) / np.linalg.norm(slow - fast + 10 ** (-6)))
    return snr

ratio = 1
mel_bins = 16
low = 20
high = 4000


times_slow, mfccs_slow = MFCCS_slow(files)
times_fast, mfccs_fast= MFCCS_fast(files, mel_bins, low, high, ratio)

print(f"MFCC slow = {np.mean(times_slow)*1000} milliseconds" )
print(f"MFCC fast = {np.mean(times_fast)*1000} milliseconds" )
SNR_ = []
for i in range(2000):
    fast = mfccs_fast[i]
    slow = mfccs_slow[i]
    SNR_.append(SNR(fast, slow))
    if fast.shape != slow.shape:
        print("Found two different shapes")
        break
print(f"SNR = {np.mean(SNR_)} dB")
