"""Reference encoder module for prosody transfer experiment

"""

from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
# from layers import ConvNorm, LinearNorm
# from utils import to_gpu, get_mask_from_lengths


def conv_output_dims(conv_net, input_dims):
    """A trick to learn the output dimensions of a convnet with known input
    dimensions. Replace this if you find a better way to do it.

    This assumes the convnet should take (N, 1, W1, H1) -> (N, C2, W2, H2). We
    want to know W2, H2.

    """
    input_tensor = torch.zeros(1, 1, *input_dims)
    return conv_net(input_tensor).shape[-2:]


class ReferenceEncoder(nn.Module):
    r"""Recall the architecture of the reference encoder:

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
    def __init__(self, input_dims, filter_size=3, filter_stride=2,
                 layer_filters=[32, 32, 64, 64, 128, 128],
                 embedding_dim=128, activation=nn.Tanh()):
        super().__init__()
        self.input_dims = input_dims
        self.embedding_dim = embedding_dim

        self.conv_block = ConvBlock(input_dims, filter_size, filter_stride, layer_filters)

        # rnn input dimension should be (cnn output channels) * dR_reduced
        rnn_input_dim = layer_filters[-1] * conv_output_dims(self.conv_block, input_dims)[1]
        self.rnn_block = RnnBlock(rnn_input_dim, embedding_dim)

        # TODO: should there be bias in the linear layer?
        self.linear_layer = nn.Linear(self.rnn_block.embedding_dim, self.embedding_dim, bias=True)
        self.activation = activation

    def forward(self, x):
        assert x.shape[2:] == self.input_dims  # this makes me a little uneasy
        x = self.conv_block(x)
        x = self.rnn_block(x)
        x = self.activation(self.linear_layer(x))
        return x


class ConvBlock(nn.Module):
    r"""The convolutional block of the reference encoder. I'm made a little
    uncertain by the requirement of constant L_R

    """
    def __init__(self, input_dims, filter_size, filter_stride, layer_filters):
        r"""input_dims: the dimensions (L_R, d_R) of the reference signal to be encoded
        filter_size: the (identical) receptive field of each filter
        filter_stride: the (identical) stride of each filter layer_filters: an
        array containing the /number of filters/ in each layer

        """
        super().__init__()
        self.input_dims = input_dims
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.layer_filters = layer_filters

        # TODO: double-check this is correct
        # I don't really think it is.
        self.output_dim = self.layer_filters[-1]

        ### Internal layers ###
        # In each layer, filters with SAME padding, ReLU activation, and
        # Batchnorm.
        self.conv_layers = []
        self.batchnorm_layers = []
        for i, filters in enumerate(layer_filters):
            self.batchnorm_layers.append(
                nn.BatchNorm2d(num_features=filters,
                               affine=True) # learnable?
            )

            # TODO: how many in_channels and out_channels?
            in_channels = 1 if i == 0 else layer_filters[i - 1]
            # TODO: should use SAME padding. This doesn't appear to exist in
            # PyTorch. Newer versions of PyTorch do have a `padding_mode`
            # keyword argument, but it has no `SAME` option.
            self.conv_layers.append(
                nn.Conv2d(in_channels, filters, self.filter_size,
                          stride=self.filter_stride,
                          padding=0)
            )

    def forward(self, x):
        r"""Forward pass through the convolution stage of the reference encoder. The if
        x has shape (1, 1, n, m), the output will have shape (1,
        self.layer_filters[-1], ~n/(stride ** depth), ~m/(stride ** depth)).

        """

        # TODO: resolve the order in which these layers should be applied.
        # Seems that before ReLU is more conventional, but some theory papers
        # have suggested that after is better. TODO: will this be slow if I
        # don't explicitly unroll this loop?
        for i, _ in enumerate(self.layer_filters):
            x = self.conv_layers[i](x)
            x = self.batchnorm_layers[i](x)
            x = F.relu(x)
        return x


class RnnBlock(nn.Module):
    r"""The GRU-RNN block that compresses the output down to a fixed dimension.
    A thin wrapper around the built-in torch.nn.GRU module.

    TODO: how should we initialize h_0? Currently just zeros.
    TODO: need to calculate input_dim

    N.b. the PyTorch GRU is a little odd: its batch index is the /second/
    index, rather than the first. see https://pytorch.org/docs/stable/nn.html.

    I think where the prosody paper says "the dR/64 feature dimensions and 128
    channels of the final convolution layer are unrolled as the inner dimension
    or the resulting LR/64 x (128 dR/64) matrix", they are talking about the
    reshaping operation below. (Please excuse the crummy ascii art).

              (N,   128,  LR/64,    dR/64)
                \     \___/           /
                 \_______/\          /
                        /\ \____x___/
                       /  \     |
                     (L,   N,  H_in)
                                 ^ input_dim

    """
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        # TODO: should we use bias here? dropout?
        self.gru = nn.GRU(input_size=input_dim, hidden_size=embedding_dim, num_layers=1)

    def forward(self, x):
        r"""Native input dimension is (N, 128, LR/64, dR/64). Need to first permute the
        batch index into the order supported by the GRU, then flatten the
        ConvNet channel and reference dimensions.

        """
        x = x.permute(2,0,1,3).flatten(start_dim=-2)

        # native input dimension for h0 is (S, N, hidden_size).
        h0 = torch.zeros(1, x.shape[1], self.embedding_dim)

        _, hn = self.gru(x, h0)  # discard the GRU output: don't care about it.

        # Native output dimension is (S, N, H_out). Want (N, H_out).
        return hn.squeeze()
