#!/usr/bin/env python
# coding: utf-8

##
## Tacotron 2 inference code 
##

"""
Edit the variables **checkpoint_path** and **text** to match yours and run the entire code to generate plots of mel outputs, alignments and audio synthesis from the generated mel-spectrogram using Griffin-Lim.
"""

#### Import libraries and setup matplotlib

import matplotlib
import matplotlib.pylab as plt

import os
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

from scipy.fftpack import fft, dct


#### Setup hparams

hparams = create_hparams()


#### Load model from checkpoint

training_steps = 37000
checkpoint_path = "output/blizzard-vanilla/checkpoint_{}".format(training_steps)
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().float()


#### Load WaveGlow for mel2audio synthesis and denoiser

waveglow_path = 'waveglow_256channels.pt'
waveglow_ = torch.load(waveglow_path)['model']
waveglow = update_model(waveglow_)
waveglow.cuda().eval().float()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


#### Loop over 8 test utterances and corresponding ref wavs

text_100 = [ "I am a little acquainted with Captain Benwick.",
             "And I beheld another beast coming up out of the earth;",
             "too little time alas with her husband whose work grew and grew;",
             "The sight of his uplifted face brought about a sudden change in her own.",
             "He was a buccaneer.",
             "He might as well have divulged it frankly for the neighbours all knew well enough that it was the face of dead Ellen Coleman that he had seen.",
             "For if there be first a willing mind it is accepted according to that a man hath and not according to that he hath not.",
             "attended too with the assurance of her expecting material advantage to Marianne from the communication of what had passed.",
             "He said I hunt for haddocks eyes Among the heather bright And work them into waistcoat buttons In the silent night.",
             "His understanding was good and his education had given it solid improvement.",
             "When I am very happy she kissed him I remember on how little it all hangs.",
             "He was determined at least not to mar it by an imprudent marriage;",
             "I've tried but he bores me.",
             "There he stood by Miss Parry's chair as though he had been cut out of wood he talking about wild flowers.",
             "It had slender green leaves the colour of emeralds and in the centre of the leaves a blossom like a golden cup.",
             "The shells were a little open;",
             "What modern girl could live like those inane females?",
             "asked Robert of Mrs. Pontellier.",
             "They care for that more than for being thanked.",
             "But what are those Master Words?",
             "She hated breaking the flowers but she wanted just one or two to go with her.",
             "She was no more used to considering other people than Colin was and she saw no reason why an ill tempered boy should interfere with the thing she liked best.",
             "But she would think of something else;",
             "The way she said Here is my Elizabeth!",
             "Sometimes it is called a crazy quilt because the patches and colors are so mixed up.",
             "roared Pew. The money's there.",
             "All the dwellers in Starkfield as in more notable communities had had troubles enough of their own to make them comparatively indifferent to those of their neighbours;",
             "Not that they care for it really.",
             "And she added timidly lowering her eyes You could always lie down for a little.",
             "but he would be meeting somebody at luncheon.",
             "All were addressed in large letters To the Little Girl in the right hand attic.",
             "I tell you every generation breeds a more rabbity generation with india rubber tubing for guts and tin legs and tin faces.",
             "then at last You must come to the house she said;",
             "He did answer.",
             "Ask and it shall be given you;",
             "He is not;",
             "It was very different when the masters of the science sought immortality and power;",
             "And if they weren't I'm here.",
             "The child could not be made amenable to rules.",
             "But this state of things was provided for.",
             "Bitter and burning Miss Kilman had turned into a church two years three months ago.",
             "They often admitted into the room a good deal of smoke and soot;",
             "I don't know what I should do without you.",
             "No till Cole alluded to my supposed attachment it had never entered my head.",
             "I shall be away all summer.",
             "She would become a doctor a farmer possibly go into Parliament if she found it necessary all because of the Strand.",
             "I never lived next door to no eathens miss she said;",
             "While his friend punched and patted the Scarecrow's body to smooth out the humps Scraps turned to Ojo and whispered: Roll me out please;",
             "There was to be no crudity in Mrs. Penniman's treatment of the situation;",
             "I didn't so you must tell me.",
             "Well done for a yearling!",
             "I was powerless almost unconscious.",
             "She withdrew her own hand but whether by accident or design he touched it.",
             "The idea of becoming what her mother had been;",
             "I dare say you can manage her.",
             "She wondered what sort of herbs they were which the old man was so sedulous to gather.",
             "Those six years might have given me two or three quite pleasant little happinesses instead of one profound regret.",
             "but if he had been with us in Europe he would have seen that father was never impressed in that way.",
             "I don't know if I will!",
             "I'm going to talk to him about our de parture our luggage.",
             "The firm yet placid mouth the clear veracious glance of the brown eyes speak now of a nature that has been tested and has kept its highest qualities;",
             "concluded the small woman bigly.",
             "His attitude was one of hopeless resignation as he looked toward a distant bird winging its flight away from him.",
             "Their wives never came to the island until late in May or early in June for they did not care to be torn to pieces;",
             "I must speak.",
             "The monkey caught his hand and pulled hard.",
             "She went with you all alone?",
             "Degrading passion!",
             "But all that had happened to her within the last few weeks had stirred her to the sleeping depths.",
             "so I said to Sir John I do think I hear a carriage;",
             "She went to the door and spoke to the beggar child.",
             "for Catherine had addressed him two short notes which met with no acknowledgment.",
             "I have observed it all my life.",
             "True he looked doubtfully fearfully even at times with horror and the bitterness of hatred at the deformed figure of the old physician.",
             "Another charge and they are fairly started.",
             "Well said Martha evidently not in the least aware that she was impudent it's time tha should learn.",
             "The railroad's blocked by a freight train that got stuck in a drift below the Flats he explained as we jogged off into the stinging whiteness.",
             "Shall I tell you my guess?",
             "Was it but the mockery of penitence?",
             "Now said he of us five which is leader?",
             "I thought you was gettin divorced.",
             "and repeated being simple by nature and undebauched because he had tramped and shot;",
             "Then touching the shoulder of a townsman who stood near to him he addressed him in a formal and courteous manner: I pray you good Sir said he who is this woman?",
             "What is that?",
             "Above because the lower surface of the iceberg stretched over us like an immense ceiling.",
             "I have very often wished to undeceive yourself and my mother added Elinor;",
             "Tha's no need to try to hide anything from _him_.",
             "A leaf violently agitated danced past her while other leaves lay motionless.",
             "Was it cheese you said he had a fancy for?",
             "but thou shalt follow me afterwards.",
             "but he was a lover;",
             "She found on reaching home that she had as she intended escaped seeing Mr Elliot;",
             "He whispered: Is it this?",
             "I am very angry with you.",
             "Elinor paid her every quiet and unobtrusive attention in her power;",
             "Perhaps she might;",
             "After supper our car will take you home.",
             "Now I'll give YOU something to believe.",
             "The dog came up licked his hand and made signs implying that he expected some great reward for signal services rendered.",
             "a gap had been broken through it and in the gap were the footprints of the sheep." ]

ref_wav_100 = [ 'Blizzard-Challenge-2013/CB-PER-18-263.wav',
                'Blizzard-Challenge-2013/CB-REV-01-302.wav',
                'Blizzard-Challenge-2013/CB-MD-03-19.wav',
                'Blizzard-Challenge-2013/CB-SG-26-123.wav',
                'Blizzard-Challenge-2013/CB-MD-02-140.wav',
                'Blizzard-Challenge-2013/CB-WCD-01-149.wav',
                'Blizzard-Challenge-2013/CB-COR-02-162.wav',
                'Blizzard-Challenge-2013/CB-SAS-31-192.wav',
                'Blizzard-Challenge-2013/CB-LG-08-234.wav',
                'Blizzard-Challenge-2013/CB-SAS-03-24.wav',
                'Blizzard-Challenge-2013/CB-RV-20-95.wav',
                'Blizzard-Challenge-2013/CB-PER-21-240.wav',
                'Blizzard-Challenge-2013/CB-LCL-13-649.wav',
                'Blizzard-Challenge-2013/CB-MD-02-353.wav',
                'Blizzard-Challenge-2013/CB-VR-01-211.wav',
                'Blizzard-Challenge-2013/CB-20K2-03-114.wav',
                'Blizzard-Challenge-2013/CB-BBH-01-239.wav',
                'Blizzard-Challenge-2013/CB-AW-05-89.wav',
                'Blizzard-Challenge-2013/CB-ALP-16-291.wav',
                'Blizzard-Challenge-2013/CB-JB-02-43.wav',
                'Blizzard-Challenge-2013/CB-LCL-08-62.wav',
                'Blizzard-Challenge-2013/CB-SG-16-45.wav',
                'Blizzard-Challenge-2013/CB-MD-03-950.wav',
                'Blizzard-Challenge-2013/CB-MD-02-19.wav',
                'Blizzard-Challenge-2013/CB-OZ2-02-113.wav',
                'Blizzard-Challenge-2013/CB-TRE-05-26.wav',
                'Blizzard-Challenge-2013/CB-EF-01-85.wav',
                'Blizzard-Challenge-2013/CB-LCL-09-245.wav',
                'Blizzard-Challenge-2013/CB-CHE-12-322.wav',
                'Blizzard-Challenge-2013/CB-MD-03-388.wav',
                'Blizzard-Challenge-2013/CB-ALP-16-227.wav',
                'Blizzard-Challenge-2013/CB-LCL-15-144.wav',
                'Blizzard-Challenge-2013/CB-WSQ-09-159.wav',
                'Blizzard-Challenge-2013/CB-SG-05-38.wav',
                'Blizzard-Challenge-2013/CB-MTW-01-203.wav',
                'Blizzard-Challenge-2013/CB-RV-05-213.wav',
                'Blizzard-Challenge-2013/CB-FRA-03-90.wav',
                'Blizzard-Challenge-2013/CB-JB-05-368.wav',
                'Blizzard-Challenge-2013/CB-SL-06-29.wav',
                'Blizzard-Challenge-2013/CB-20K1-15-40.wav',
                'Blizzard-Challenge-2013/CB-MD-03-821.wav',
                'Blizzard-Challenge-2013/CB-AW-21-04.wav',
                'Blizzard-Challenge-2013/CB-RV-11-21.wav',
                'Blizzard-Challenge-2013/CB-EM-33-184.wav',
                'Blizzard-Challenge-2013/CB-SG-12-210.wav',
                'Blizzard-Challenge-2013/CB-MD-03-1153.wav',
                'Blizzard-Challenge-2013/CB-ALP-10-223.wav',
                'Blizzard-Challenge-2013/CB-OZ2-13-113.wav',
                'Blizzard-Challenge-2013/CB-WSQ-08-35.wav',
                'Blizzard-Challenge-2013/CB-JB-05-150.wav',
                'Blizzard-Challenge-2013/CB-JB-04-187.wav',
                'Blizzard-Challenge-2013/CB-20K2-16-170.wav',
                'Blizzard-Challenge-2013/CB-FFM-24-65.wav',
                'Blizzard-Challenge-2013/CB-PER-17-144.wav',
                'Blizzard-Challenge-2013/CB-ALP-04-101.wav',
                'Blizzard-Challenge-2013/CB-SL-15-06.wav',
                'Blizzard-Challenge-2013/CB-CHE-11-424.wav',
                'Blizzard-Challenge-2013/CB-WSQ-25-112.wav',
                'Blizzard-Challenge-2013/CB-FFM-26-238.wav',
                'Blizzard-Challenge-2013/CB-CHE-12-141.wav',
                'Blizzard-Challenge-2013/CB-SM-16-11.wav',
                'Blizzard-Challenge-2013/CB-FFM-30-111.wav',
                'Blizzard-Challenge-2013/CB-AW-09-84.wav',
                'Blizzard-Challenge-2013/CB-JB-04-24.wav',
                'Blizzard-Challenge-2013/CB-RV-17-55.wav',
                'Blizzard-Challenge-2013/CB-JB2-02-169.wav',
                'Blizzard-Challenge-2013/CB-DM-01-513.wav',
                'Blizzard-Challenge-2013/CB-MD-03-908.wav',
                'Blizzard-Challenge-2013/CB-SUM-05-85.wav',
                'Blizzard-Challenge-2013/CB-SAS-19-93.wav',
                'Blizzard-Challenge-2013/CB-ALP-13-176.wav',
                'Blizzard-Challenge-2013/CB-WSQ-30-83.wav',
                'Blizzard-Challenge-2013/CB-PER-03-56.wav',
                'Blizzard-Challenge-2013/CB-SL-11-22.wav',
                'Blizzard-Challenge-2013/CB-JB-03-227.wav',
                'Blizzard-Challenge-2013/CB-SG-04-71.wav',
                'Blizzard-Challenge-2013/CB-EF-01-153.wav',
                'Blizzard-Challenge-2013/CB-SAS-18-86.wav',
                'Blizzard-Challenge-2013/CB-SL-12-13.wav',
                'Blizzard-Challenge-2013/CB-JB2-03-99.wav',
                'Blizzard-Challenge-2013/CB-LCL-16-636.wav',
                'Blizzard-Challenge-2013/CB-MD-03-517.wav',
                'Blizzard-Challenge-2013/CB-SL-03-19.wav',
                'Blizzard-Challenge-2013/CB-RV-06-287.wav',
                'Blizzard-Challenge-2013/CB-20K2-15-111.wav',
                'Blizzard-Challenge-2013/CB-SAS-37-125.wav',
                'Blizzard-Challenge-2013/CB-SG-07-113.wav',
                'Blizzard-Challenge-2013/CB-RV-16-261.wav',
                'Blizzard-Challenge-2013/CB-TRE-19-99.wav',
                'Blizzard-Challenge-2013/CB-JO-02-50.wav',
                'Blizzard-Challenge-2013/CB-SAS-11-45.wav',
                'Blizzard-Challenge-2013/CB-PER-22-19.wav',
                'Blizzard-Challenge-2013/CB-RV-20-124.wav',
                'Blizzard-Challenge-2013/CB-PER-09-127.wav',
                'Blizzard-Challenge-2013/CB-SAS-29-08.wav',
                'Blizzard-Challenge-2013/CB-EM-32-39.wav',
                'Blizzard-Challenge-2013/CB-SCA-01-454.wav',
                'Blizzard-Challenge-2013/CB-LG-05-114.wav',
                'Blizzard-Challenge-2013/CB-FFM-05-48.wav',
                'Blizzard-Challenge-2013/CB-FFM-05-40.wav' ]


mcd = []

for i, t in enumerate(text_100):

    print(i)

    for j in range(3):

        #### Prepare text input

        sequence = np.array(text_to_sequence(t, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()

        #### Infer mel spectrograms on text input

        stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        audio, sampling_rate = load_wav_to_torch(ref_wav_100[i])
        audio_norm = audio / hparams.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

        ref_mels = stft.mel_spectrogram(audio_norm)
        ref_mels = ref_mels.cuda().float()

        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

        #### Synthesize audio

        with torch.no_grad():
            inferred_audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
            inferred_audio = denoiser(inferred_audio, strength=0.01)[:, 0]
            inferred_audio = inferred_audio[0].data.cpu().numpy()

        #### Compute Mel Cepstral Distortion

        stft_mcd = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            20, hparams.sampling_rate, hparams.mel_fmin, 4000.0) 

        # Log Mel Spectograms

        ref_mels_mcd = stft_mcd.mel_spectrogram(audio_norm)
        ref_mels_mcd = ref_mels_mcd.float().squeeze(0)
        ref_mels_mcd = ref_mels_mcd.data.cpu().numpy()

        inferred_audio = inferred_audio / np.max(np.abs(inferred_audio))
        inferred_audio = torch.tensor(inferred_audio)
        inferred_audio = inferred_audio.unsqueeze(0)

        inferred_mels_mcd = stft_mcd.mel_spectrogram(inferred_audio)
        inferred_mels_mcd = inferred_mels_mcd.float().squeeze(0)
        inferred_mels_mcd = inferred_mels_mcd.data.cpu().numpy()

        # Padding

        if (ref_mels_mcd.shape[1] < inferred_mels_mcd.shape[1]):
            pad_len = inferred_mels_mcd.shape[1] - ref_mels_mcd.shape[1]
            ref_mels_mcd = np.pad(ref_mels_mcd, ((0, 0), (0, pad_len)), constant_values=-13.8)
        else:
            pad_len = ref_mels_mcd.shape[1] - inferred_mels_mcd.shape[1]
            inferred_mels_mcd = np.pad(inferred_mels_mcd, ((0, 0), (0, pad_len)), constant_values=-13.8)

        # Cosine transform

        mfcc_ref = dct(ref_mels_mcd)
        mfcc_inferred = dct(inferred_mels_mcd)

        # Mean squared error

        mcd_ = np.mean(np.sqrt(np.sum(np.square(mfcc_inferred[1:13] - mfcc_ref[1:13]), axis=0)))

        mcd.append(mcd_)


print(np.mean(mcd))

