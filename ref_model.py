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

        convolutions = []
        bottleneck_in = [1, 32, 32, 64, 64, 128]
        bottleneck_out = [32, 32, 64, 64, 128, 128]

        for i in range(0,6):
            conv_layer = nn.Sequential(
                ConvNorm(in_channels = bottleneck_in[i],
                         out_channels = bottleneck_out[i],
                         kernel_size = 3,
                         stride = 2,
                         padding = 1,
                         dilation = 1,
                         w_init_gain = 'relu'),
                nn.BatchNorm2d(bottleneck_out[i]))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.gru = nn.GRU(input_size = 256,
                          hidden_size = 128,
                          num_layers = 1)

        self.linear_projection = LinearNorm(in_dim = 128,
                                            out_dim = 128,
                                            w_init_gain='tanh')

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        #
        # TODO: Reshape output of final convolution to a matrix
        #       (L/64 by d/64 by 128) tensor --> (L/64 by 128 d/64) matrix
        # 

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True) 

        x = F.dropout(torch.tanh(self.linear_projection(x)), 0.5, self.training)

        return x


