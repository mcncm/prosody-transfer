import glob
import os
import sys

import numpy as np
import tensorflow as tf
# from tensorflow.summary import event_accumulator
import matplotlib.pyplot as plt

eng_logdir = os.path.join('tacotron2', 'eng-outdir', 'logdir')
ipa_logdir = os.path.join('tacotron2', 'nostress-ipa-outdir', 'logdir')
fig_dir = os.path.join('writeup', 'figures')

figsize=(3, 3)
fontsize=12

def get_event(dir_path):
    r"""Borrowed from Saki Shinoda
    """
    latest = max(
    glob.glob('{}/*'.format(dir_path)),
                key=os.path.getctime)

    print(latest)
    return latest

def get_tag_data(dir_path, tag='validation.loss'):
    data = []
    try: # handle this more gracefully / understand it better
        for e in tf.train.summary_iterator(get_event(dir_path)):
            for v in e.summary.value:
                #if v.tag == 'loss' or v.tag == 'accuracy':
                if v.tag == tag:
                    data.append(v.simple_value)
    except:
        pass

    return np.array(data)

for tag in ['training.loss', 'validation.loss']:
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(get_tag_data(eng_logdir, tag), label='English orthography')
    ax.plot(get_tag_data(ipa_logdir, tag), label='IPA')

    ax.set_xlabel('Training step', fontsize=fontsize)
    ax.set_ylabel(tag, fontsize=fontsize)
    ax.set_yscale('log')

    fig_filename = 'tacotron2_' + tag.replace('.','_') + '.png'
    fig.savefig(os.path.join(fig_dir, fig_filename))
