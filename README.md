# FastMFCC
A comparison of two feature extraction pipelines for Keyword Spotting. They are referred to as MFCC_slow and MFCC_fast and are
tested on Raspberry4.

The whole preoprocessing pipeline for this Keyword Spotting task consist of a Short Time Fourier Transform (STFT) followed by 
a Mel-frequency cepstral coefficients (MFCCs) representation.
![mfcc_pipeline](https://user-images.githubusercontent.com/70110839/209448406-f5807ab0-b22c-4f2e-b8ef-5dbd3ca17748.png)

The objective is to define a preprocessing routine referred to as MFCC_fast returning a tensor of a given shape
minimizing execution time and maximizing signal-to-noise ratio:

![snr](https://user-images.githubusercontent.com/70110839/209448334-4a056d78-86cc-430e-869e-d33270ab3aad.png)



The developed routine reduces of 32% the execution time,
from 25ms to 17ms, returning an SNR = 22.37dB.

## Background

Always-on speech recognition is not energy efficient as it requires to transmit a continuous
audio stream to the cloud, where data get processed. To mitigate this concern, devices first detect
short keywords such as “Hey Siri” or “Ok Google” that wake up the device and trigger the full-scale
speech recognition. This task, called Keyword Spotting, is much simpler, and therefore can be
performed on board of the sensing nodes with lightweight Convolutional Neural Networks. 
![kspotting](https://user-images.githubusercontent.com/70110839/209448404-210480e5-8197-4929-b671-73e8a5cf86ff.png)

Before feeding the data to a Convolutional Neural Networks, it is required to perform a set of preprocessing steps. 
The most common strategy is to move from the time domain to the frequency
domain using Short Time Fourier Transform (STFT).  This transformation converts a one-dimensional timeseries signal into a two-dimensional image, enabling to solve keyword spotting 
as an image classification problem.
![fft](https://user-images.githubusercontent.com/70110839/209448407-83f84c26-4ad6-456e-8248-7dabeb55be16.png)
Another common feature extraction step relies on the hypothesis that representing sounds as they are perceived by the
human ear improves the classification accuracy and can be achieved extracting the Mel-frequency cepstral coefficients (MFCCs) from the input signal.
The Mel-frequency cepstrum is a representation of the STFT of a sound that tries to mimic how the
membrane in human ears senses the vibrations of sounds. The MFCCs are coefficients that composes the
Mel-frequency cepstrum. 
![mfcc](https://user-images.githubusercontent.com/70110839/209448405-69877357-cf04-4dec-9c86-57d9087511d4.png)



