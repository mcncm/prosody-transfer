"""An extremely ad-hoc test harness for basic sanity checking.
Make sure things don't revert /too/ badly when I commit things.

To add a test, just add a new function annotated with the decorator `@test`.
"""

import traceback
from termcolor import colored

import torch
import torch.nn as nn

from ref_model import ConvBlock, RnnBlock, ReferenceEncoder


### DEFINITIONS ### 
test_registry = {}

def test(fn):
    test_registry[fn.__name__] = fn
    return fn


### TESTS ###
@test
def _test_conv_block():
    """Just feed some numbers through a ConvBlock and make sure it doesn't
    crash.
    """
    input_shape = (100, 200)
    # leading 1s for batch size and number of channels (I think? Confirm.)

    input_tensor = torch.randn(1, 1, *input_shape)
    cb = ConvBlock(input_shape, 3, 2, [32, 32, 64])
    output = cb(input_tensor)

@test
def _test_rnn_block():
    """Just feed some numbers through an RnnBlock and make sure it doesn't
    crash.
    """
    LR_reduced = 36  # an arbitrary number for LR / 64
    dR_reduced = 24  # an arbitrary number for dR / 64
    input_dim = 128 * dR_reduced
    cnn_channels = 128
    embedding_dim = 128

    input_tensor = torch.randn(1, cnn_channels, LR_reduced, dR_reduced)
    rb = RnnBlock(input_dim, embedding_dim)
    output = rb(input_tensor)
    assert output.shape == torch.Size([embedding_dim])

@test
def _test_reference_encoder():
    LR = 860
    dR = 240

    input_spec = torch.randn(1, 1, LR, dR)
    re = ReferenceEncoder(input_dims=(LR, dR))
    embedding = re(input_spec)
    assert embedding.shape == torch.Size([128])

# @test
def _test_reference_encoder_real_spectrogram():
    pass


### RUN THE TESTS ###
for name, fn in test_registry.items():
    try:
        fn()
        print(colored("Passed test {}".format(name), 'green'))
    except Exception :  # Here, pokemon exception handling is a-ok
        print(colored("Tests failed in the execution of {}\n\n".format(name), 'red'),
              traceback.format_exc())
        exit(1)
