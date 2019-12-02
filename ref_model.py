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
        #           Method for initialization not specified in Skerry-Ryan et al, so using
        #           Glorot initialization as in NVIDIA tacotron2 implementation
        #
        convolutions = []
        strided_conv2d_in = [1, 32, 32, 64, 64, 128]
        strided_conv2d_out = [32, 32, 64, 64, 128, 128]

        for i in range(0,6):
            conv_layer = nn.Sequential(
                ConvNorm(in_channels = strided_conv2d_in[i],
                         out_channels = strided_conv2d_out[i],
                         kernel_size = 3,
                         stride = 2,
                         padding = 1,
                         dilation = 1,
                         w_init_gain = 'relu'),
                nn.BatchNorm2d(strided_conv2d_out[i]))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        #
        # Gated Recurrent Unit:
        #   Summarizes sequence as a 128-dimensional vector
        # 
        self.gru = nn.GRU(input_size = 256,
                          hidden_size = 128,
                          num_layers = 1)

        #
        # Fully connected layer
        #   Linear projection of final GRU state to desired dimensionality,
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

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        self.gru.flatten_parameters()

        #
        # Take 128-dim hidden state as summary of sequence
        #
        _, x = self.gru(x)

        _, x = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True) 

        x = F.dropout(torch.tanh(self.linear_projection(x)), 0.5, self.training)

        return x


