import matplotlib
import matplotlib.pylab as plt

import os
import sys
import numpy as np
import torch

from hparams import create_hparams

import layers
from utils import load_wav_to_torch


#### Plotting routine

def plot_data(data, plt_path, figsize=(4, 12)):
    fig, axes = plt.subplots(len(data), 1, figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
    plt.savefig(plt_path)


#### Setup hparams

hparams = create_hparams()

stft = layers.TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length,
    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
    hparams.mel_fmax)


#### Load reference wav

ref_wav = 'Blizzard-Challenge-2013/CB-LCL-19-282.wav'
audio, sampling_rate = load_wav_to_torch(ref_wav)
audio_norm = audio / hparams.max_wav_value
audio_norm = audio_norm.unsqueeze(0)
audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
ref_mels = stft.mel_spectrogram(audio_norm)
ref_mels = ref_mels.cuda().float()
ref_mels = ref_mels[:,:,-270:] # trim first 100 or so frames 

ref_linear = stft.linear_spectrogram(audio_norm)
ref_linear = ref_linear.cuda().float()
ref_linear = ref_linear[:,:,-270:]


#### Load prosody wav

infer_wav = 'inference/do-love-[82000]-(1).wav'
audio, sampling_rate = load_wav_to_torch(infer_wav)
audio_norm = audio / hparams.max_wav_value
audio_norm = audio_norm.unsqueeze(0)
audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
infer_mels = stft.mel_spectrogram(audio_norm)
infer_mels = infer_mels.cuda().float()

infer_linear = stft.linear_spectrogram(audio_norm)
infer_linear = infer_linear.cuda().float()


#### Load vanilla wav

vanilla_wav = 'do-love-vanilla.wav'
audio, sampling_rate = load_wav_to_torch(vanilla_wav)
audio_norm = audio / hparams.max_wav_value
audio_norm = audio_norm.unsqueeze(0)
audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
vanilla_mels = stft.mel_spectrogram(audio_norm)
vanilla_mels = vanilla_mels.cuda().float()

vanilla_linear = stft.linear_spectrogram(audio_norm)
vanilla_linear = vanilla_linear.cuda().float()


#### Plot mel spectrograms & alignments

plt_filename = 'do-love-spectrograms.png'
plt_path = plt_filename
plot_data((ref_mels.float().data.cpu().numpy()[0],
           infer_mels.float().data.cpu().numpy()[0],
           vanilla_mels.float().data.cpu().numpy()[0]),
          plt_path)


"""
#### Plot linear spectrograms 

plt_filename = 'do-love-linear.png'
plt_path = plt_filename
plot_data((ref_linear.float().data.cpu().numpy()[0],
           infer_linear.float().data.cpu().numpy()[0],
           vanilla_linear.float().data.cpu().numpy()[0]),
          plt_path)
"""
