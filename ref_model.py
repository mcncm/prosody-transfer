from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths


class ReferenceEncoder(nn.Module):
    """ReferenceEncoder module:
        - Six-layer strided Conv2D network
        - 128-unit GRU
        - Linear projection
    """
    def __init__(self, hparams):
        super(ReferenceEncoder, self).__init__()

        #
        # Strided Conv2d w/ BatchNorm:
        #   Downsamples reference signal by a factor of 64 along
        #   both dimensions (seq length & mel channels) due to stride
        #     Note: SAME padding is just (kernel_size - 1)/2, assuming kernel_size is odd
        #
        convolutions = []
        strided_conv2d_in = [1, 32, 32, 64, 64, 128]
        strided_conv2d_out = [32, 32, 64, 64, 128, 128]

        for i in range(0,6):
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels = strided_conv2d_in[i],
                          out_channels = strided_conv2d_out[i],
                          kernel_size = 3,
                          stride = 2,
                          padding = 1,
                          dilation = 1),
                nn.BatchNorm2d(strided_conv2d_out[i]))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        #
        # Gated Recurrent Unit:
        #   Summarizes sequence as a 128-dimensional vector
        #     Note: input_size = ceil(n_mel_channels/64)
        # 
        self.gru = nn.GRU(input_size = 256,
                          hidden_size = 128,
                          num_layers = 1)

        #
        # Fully connected layer
        #   Linear projection of final GRU output to desired dimensionality,
        #   (in this case, also 128-dim) followed by tanh activation
        #     Note: Using Glorot initialization as in NVIDIA implementation of tacotron2
        #
        self.linear_projection = LinearNorm(in_dim = 128,
                                            out_dim = 128,
                                            w_init_gain='tanh')

    #
    # Note: Regularization method not specified, so using dropout as in tacotron2
    #
    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        #
        # TODO: Reshape output of final convolution to a matrix
        #       (L/64 by 2 by 128) tensor ---> (L/64 by 256) matrix
        # 

        #
        # Note: The torch function calls below deal with variable length sequences
        #       so that they can be processed in a batch. Still need to verify that
        #       the syntax is correct
        #
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        #
        # Note: I believe this is to increase performance. Not sure if it's necessary
        #
        self.gru.flatten_parameters()

        #
        # TODO: Take 128-dim output of GRU, at final time step, as summary of sequence
        #       Something like:
        #
        x[-1], _ = self.gru(x)

        x[-1], _ = nn.utils.rnn.pad_packed_sequence(
                     x, batch_first=True) 

        #
        # tanh activation to constrain the information contained in the embedding
        #
        x = F.dropout(torch.tanh(self.linear_projection(x)), 0.5, self.training)

        return x

    #
    # TODO: Implement corresponding routine for inference. It will look very similar,
    #       but will be processing a single (speech sample, text seq) pair rather than
    #       a batch. Likely, it will be identical to the forward function, except it
    #       won't need the pack_padded_sequence() or pad_packed_sequence() functions
    #


