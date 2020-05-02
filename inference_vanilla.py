#!/usr/bin/env python
# coding: utf-8

## Tacotron 2 inference code 
"""
Edit the variables **checkpoint_path** and **text** to match yours and run the entire code to generate plots of mel outputs, alignments and audio synthesis from the generated mel-spectrogram using Griffin-Lim.
"""

#### Import libraries and setup matplotlib

import matplotlib
import matplotlib.pylab as plt

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams_vanilla import create_hparams
from model_vanilla import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train_vanilla import load_model
from text import text_to_sequence
from denoiser import Denoiser
from convert_model import update_model

import layers
import scipy.io.wavfile as wav
from utils import load_wav_to_torch


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
    plt.savefig('out.pdf')


#### Setup hparams

hparams = create_hparams()
hparams.sampling_rate = 22050


#### Load model from checkpoint

checkpoint_path = "output/blizzard-vanilla/checkpoint_37000" # "tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()


#### Load WaveGlow for mel2audio synthesis and denoiser

waveglow_path = 'waveglow_256channels.pt'
waveglow_ = torch.load(waveglow_path)['model']
waveglow = update_model(waveglow_)
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


#### Prepare text input

# text = "I never expected to see you here."
# text = "Are there rats there?"
# text = "It could be beautiful."
# text = "Don't let us even ask said Sara."
# text = "Because it isn't Duncan that I do love she said looking up at him."
text = "I will remember if I can."
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()


#### Decode text input and plot results

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T))


#### Synthesize audio from spectrogram using WaveGlow

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    d = audio[0].data.cpu().numpy()
    d_ = np.int16(d/np.max(np.abs(d)) * 32767)
    print(d_)
    wav.write('out.wav', hparams.sampling_rate, d_)


# ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)


#### (Optional) Remove WaveGlow bias

# audio_denoised = denoiser(audio, strength=0.01)[:, 0]
# ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate) 

