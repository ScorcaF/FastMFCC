#!/usr/bin/env python

# coding: utf-8



import os

import pathlib

from scipy.io import wavfile

import numpy as np

import tensorflow as tf

import time

import math

from subprocess import Popen

from scipy import signal

Popen('sudo sh -c "echo performance >'

 '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',

 shell=True).wait()







def mfcc(filename, num_mel_bins, mel_lower_frequency, mel_upper_frequency, resample, i, linear_to_mel_weight_matrix = None):

    #Resampling

    if resample:

        rate=8000

        input_rate, audio = wavfile.read(filename)

        audio = signal.resample_poly(audio, 1, 2)

        tf_audio = tf.convert_to_tensor(audio, dtype=tf.float32)

        frame_length = 128

        frame_step = 64

        

    else:

        rate=16000

        input_rate, audio = wavfile.read(filename)

        tf_audio = tf.convert_to_tensor(audio, dtype=tf.float32)

        frame_length = 256

        frame_step = 128

    #STFT

    stft = tf.signal.stft(tf_audio, frame_length, frame_step, 

                              fft_length=frame_length)

    spectrogram = tf.abs(stft)

    

    mel_coefficients = 10

    if i==0:

        num_spectrogram_bins = spectrogram.shape[-1]

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(

                num_mel_bins, num_spectrogram_bins, rate, 

                mel_lower_frequency, mel_upper_frequency)

            

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :mel_coefficients]

    if i==0:

        return mfccs, linear_to_mel_weight_matrix

    else: 

        return mfccs







mel_lower_freq = 0

mel_upper_freq = 4000

num_mel_bins = 32

resample = False





data_dir = 'yes_no/'

files_list = os.listdir(data_dir)

MFCCfast_execTime = 0

MFCCslow_execTime = 0

    

num_files = len(files_list)

SNR = 0

    



for i, filename in enumerate(files_list):

    

    if i==0:

        start = time.time()

        mfccSlow, mat1 = mfcc(data_dir+filename, 40, 20, 4000, False, i)

        end = time.time()

        MFCCslow_execTime += (end-start)



        start = time.time()

        mfccFast, mat2 = mfcc(data_dir+filename, num_mel_bins, mel_lower_freq, mel_upper_freq, resample, i)

        end = time.time()

        MFCCfast_execTime += (end-start)

    else:

        start = time.time()

        mfccSlow = mfcc(data_dir+filename, 40, 20, 4000, False, i, mat1)

        end = time.time()

        MFCCslow_execTime += (end-start)

        



        start = time.time()

        mfccFast = mfcc(data_dir+filename, num_mel_bins, mel_lower_freq, mel_upper_freq, resample, i,mat2)

        end = time.time()

        MFCCfast_execTime += (end-start)

    

    SNR += 20 * math.log10(np.linalg.norm(mfccSlow)/np.linalg.norm(mfccSlow-mfccFast + 1e-6))

    

MFCCslow_execTime /= num_files   

MFCCfast_execTime /= num_files

SNR/= num_files



print("Average time for MFFCs slow: {:.1f} ms".format(MFCCslow_execTime*1000))

print("Average time for MFFCs fast: {:.1f} ms".format(MFCCfast_execTime*1000))

print("SNR: {:.2f} dB".format(SNR))

print("\n\n\n")













