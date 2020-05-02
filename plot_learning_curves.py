import glob
import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import pdb


log_path = os.path.join( 'output', 'log' , 'Prosody-Tacotron2-Blizzard-Challenge-exp2' )
fig_name = 'Prosody-Tacotron2-Blizzard-Challenge-exp2.png'
iters_per_checkpoint = 1000

def get_event( path ):

    """Borrowed from Saki Shinoda
    """
    latest = max( glob.glob( '{}/*'.format(path) ),
                  key=os.path.getctime )

    print( latest )

    return latest


def get_tag_data( path, tag='validation.loss' ):

    tag_data = []

    try:
        for event in tf.train.summary_iterator( get_event(path) ):
            for val in event.summary.value:
                # if val.tag == 'loss' or val.tag == 'accuracy':
                if val.tag == tag:
                    tag_data.append( val.simple_value )
    except:
        pass

    return np.array( tag_data )


def smoothed_statistic( statistic, arr, smoothing, start=None ):

    s = smoothing

    if start == None:
        start = int( np.ceil(s/2) )

    s_half = int( np.floor(s/2) )

    end = len(arr) - 1 - int( np.ceil(s/2) )

    return np.array([
            statistic( arr[i-s_half:i+s_half] )
            for i in range( start, end, s )
            ])


def plot_training_validation(experiments, smoothing=10, start=100):

    """Plot the training and validation loss
    """

    fig, ax = plt.subplots( 1, 1, figsize=(9, 6) )

    for i, (experiment, path) in enumerate( experiments.items() ):
        training_loss = get_tag_data( path, 'training.loss' )
        validation_loss = get_tag_data( path, 'validation.loss' )

        training_loss_std = smoothed_statistic( np.std, training_loss, smoothing, start )
        training_loss_mean = smoothed_statistic( np.mean, training_loss, smoothing, start )

        training_iters = range( start, len(training_loss) - 1, smoothing )

        if len(training_iters) == len(training_loss_mean):
            ax.plot( training_iters, training_loss_mean, color='C{}'.format(i), label='Training Loss' )
            ax.fill_between( training_iters,
                             training_loss_mean - training_loss_std,
                             training_loss_mean + training_loss_std,
                             color='C{}'.format(i), alpha=0.3 )

        # hack
        else:
            ax.plot( training_iters[:-1], training_loss_mean, color='C{}'.format(i), label='Training Loss' )
            ax.fill_between( training_iters[:-1],
                             training_loss_mean - training_loss_std,
                             training_loss_mean + training_loss_std,
                             color='C{}'.format(i), alpha=0.3 )

        validation_iters = range( 0, iters_per_checkpoint * len(validation_loss), iters_per_checkpoint )
        ax.plot( validation_iters, validation_loss, '--', color='C{}'.format(i), label='Validation Loss' )

    ax.set_xlim( start, training_iters[-1] )
    ax.set_ylabel( 'Loss' )
    ax.set_xlabel( 'Steps' )
    ax.set_yscale( 'log' )
    ax.grid()
    ax.legend()

    print( 'Saving!' )

    fig.savefig( fig_name )


if __name__ == '__main__':

    experiments = {'Prosody Tacotron2 on Blizzard Challenge': log_path}
    plot_training_validation( experiments, smoothing=50, start=100 )


"""
for tag in ['training.loss', 'validation.loss']:

    ax.plot(get_tag_data(eng_logdir, tag), label='English orthography')
    ax.plot(get_tag_data(ipa_logdir, tag), label='IPA')

    ax.set_xlabel('Training step', fontsize=fontsize)
    ylabel = ' '.join(tag.split('.')).capitalize()
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_yscale('log')
    ax.legend()

    fig_filename = 'tacotron2_' + tag.replace('.','_') + '.png'
    # plt.show(fig)
    fig.savefig(os.path.join(fig_dir, fig_filename))
"""
