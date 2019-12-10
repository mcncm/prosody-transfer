import glob
import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

eng_logdir = os.path.join('tacotron2', 'eng-outdir', 'logdir')
ipa_logdir = os.path.join('tacotron2', 'nostress-ipa-outdir', 'logdir')
fig_dir = os.path.join('writeup', 'figures')

figsize=(6, 4)
fontsize=16

### THIS IS A MAGIC NUMBER ###
steps_per_checkpoint = 1000
##############################

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

def smoothed_statistic(statistic, arr, smoothing, starting_point=None):
    if starting_point == None:
        starting_point = int(np.ceil(smoothing/2))
    smoothing_range = int(np.floor(smoothing / 2))
    return np.array([
        statistic(arr[i-smoothing_range:i+smoothing_range])
        for i in range(starting_point,
                       len(arr) - 1 - int(np.ceil(smoothing / 2)),
                       smoothing)
    ])


def plot_training_validation(datasets, smoothing=10, starting_point=100):
    r"""Plot the training and validation loss
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, (dataset_name, logdir) in enumerate(datasets.items()):
        training_loss = get_tag_data(logdir, 'training.loss')
        validation_loss = get_tag_data(logdir, 'validation.loss')

        training_loss_std = smoothed_statistic(np.std, training_loss, smoothing, starting_point)
        training_loss_mean = smoothed_statistic(np.mean, training_loss, smoothing, starting_point)

        training_steps = range(starting_point, len(training_loss) - 1, smoothing)

        ax.plot(training_steps, training_loss_mean, color='C{}'.format(i), label=dataset_name)
        ax.fill_between(training_steps,
                        training_loss_mean - training_loss_std,
                        training_loss_mean + training_loss_std,
                        color='C{}'.format(i), alpha=0.3)

        validation_steps = range(0, steps_per_checkpoint * len(validation_loss), steps_per_checkpoint)
        # ax.plot(validation_steps, validation_loss, '--', color='C{}'.format(i))

    ax.set_xlim(starting_point, training_steps[-1])
    ax.set_ylabel('Loss', fontsize=fontsize)
    ax.set_xlabel('Steps')
    ax.set_yscale('log')
    ax.grid()
    ax.legend()
    print('Saving!')
    fig.savefig(os.path.join(fig_dir, 'tacotron2_training_ipa.png'))

if __name__ == '__main__':
    datasets = {'English orthography': eng_logdir, 'IPA': ipa_logdir}
    plot_training_validation(datasets, smoothing=50, starting_point=100)


# for tag in ['training.loss', 'validation.loss']:
# 
#     ax.plot(get_tag_data(eng_logdir, tag), label='English orthography')
#     ax.plot(get_tag_data(ipa_logdir, tag), label='IPA')
# 
#     ax.set_xlabel('Training step', fontsize=fontsize)
#     ylabel = ' '.join(tag.split('.')).capitalize()
#     ax.set_ylabel(ylabel, fontsize=fontsize)
#     ax.set_yscale('log')
#     ax.legend()
# 
#     fig_filename = 'tacotron2_' + tag.replace('.','_') + '.png'
#     # plt.show(fig)
#     fig.savefig(os.path.join(fig_dir, fig_filename))
