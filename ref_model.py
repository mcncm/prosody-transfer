"""Reference encoder module for prosody transfer experiment
"""

from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths



class ReferenceEncoder(nn.Module):
    """
    Recall the architecture of the reference encoder:

               Final linear
               layer & act.
                    ^
                    |
                   GRU       (an RNN block)
                    ^
                    |
                CNN block
                    ^
                    |
                reference
               spec frames
    """
    def __init__(self, input_dims, filter_size=3, filter_stride=2
                 layer_filters=[32, 32, 64, 64, 128, 128],
                 output_dim=128, activaton=nn.tanh):
        self.conv_block = ConvBlock(input_dims, filter_Size, filter_stride, filters)
        self.rnn_block = RnnBlock()
        self.output_dim = output_dim

        # TODO: should there be bias in the linear layer?
        self.linear_layer = nn.Linear(rnn_block.output_dim, self.output_dim, bias=True)
        self.activation = activation

    def forward(self, x):
        assert spectrogram.shape = input_dims  # this makes me a little uneasy
        x = self.conv_block(x)
        x = self.rnn_block(x)
        x = self.activation(self.linear_layer(x))
        return x


class ConvBlock(nn.Module):
    """The convolutional block of the reference encoder.
    I'm made a little uncertain by the requirement of constant L_R
    """
    def __init__(self, input_dims, filter_size, filter_stride, layer_filters):
        """
        input_dims: the dimensions (L_R, d_R) of the reference signal to be encoded
        filter_size: the (identical) receptive field of each filter
        filter_stride: the (identical) stride of each filter
        layer_filters: an array containing the /number of filters/ in each layer
        """
        self.input_dims = input_dims
        # TODO: double-check this is correct
        # I don't really think it is.
        self.output_dim = self.filters[-1]

        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.filters = filters

        ### Internal layers ###
        # In each layer, filters with SAME padding, ReLu activation,
        # and Batchnorm.
        self.conv_layers = []
        self.batchnorm_layers = []
        for i, filters in enumerate(filters):
            self.batchnorm_layers.append(
                nn.batchNorm2d(num_features=filter
                               affine=True) # learnable?
            )

            # TODO: how many in_channels and out_channels?
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride=self.filter_stride,
                          padding=0, padding_mode='zero')  # TODO: should use SAME padding.
            )

    def forward(self, x):
        # TODO: resolve the order in which these layers should be applied.
        # Seems that before ReLU is more conventional, but some theory papers
        # have suggested that after is better.
        # TODO: will this be slow if I don't explicitly unroll this loop?
        for i, _ in enumerate(filters):
            x = self.conv_layers[i](x)
            x = self.batch_layers[i](x)
            x = F.relu(x)
        return x


class RnnBlock(nn.Module):
    def __init__(self, rnn_dim):
        pass

    def forward(self):
        pass
