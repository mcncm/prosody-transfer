from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import LinearNorm


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
        #     TODO: bias? initialization? dropout?
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
                          padding = 1),
                nn.BatchNorm2d(strided_conv2d_out[i]))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        #
        # Gated Recurrent Unit:
        #   Summarizes sequence as a 128-dimensional vector
        #     Note: input_size = 128 * ceil(n_mel_channels/64)
        #     TODO: bias? dropout? bidirectional? deeper? initialization?
        # 
        self.gru = nn.GRU(input_size = 256,
                          hidden_size = 128,
                          num_layers = 1,
                          batch_first = True)

        #
        # Fully connected layer
        #   Linear projection of final GRU output to desired dimensionality,
        #   (in this case, also 128-dim) followed by tanh activation
        #     Note: Using Glorot initialization as in NVIDIA implementation of tacotron2
        #
        self.linear_projection = LinearNorm(in_dim = 128,
                                            out_dim = 128,
                                            w_init_gain='tanh')

    def forward(self, x, lengths):

        # Strided Conv2d w/ BatchNorm
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        # Unroll downsampled feature dim & signal length to a single inner dim
        conv_size = x.size
        x = x.view(conv_size(0), conv_size(1)*conv_size(2), conv_size(3))

        x = x.transpose(1, 2)

        # Pack padded sequence
        lengths = torch.ceil(lengths.double()/64).long()
        lengths, i = torch.sort(lengths, descending=True)
        lengths = lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x[i], lengths, batch_first=True)

        # Aggregate all weight tensors into contiguous GPU memory
        self.gru.flatten_parameters()

        # Propagate through GRU
        outputs, _ = self.gru(x)

        # Pad packed sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # Take final output of GRU as pooled summarization of sequence
        finals = outputs.gather(1, (lengths-1).view(-1,1,1).expand(48, 1, 128))

        # Restore original sorting
        _, i_inv = torch.sort(i)
        finals[i_inv]

        # tanh activation to constrain the information contained in the embedding
        embeddings = F.dropout(torch.tanh(self.linear_projection(finals)), 0.5, self.training)

        return embeddings 


    def inference(self, x, lengths):

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        
        conv_size = x.size
        x = x.view(conv_size(0), conv_size(1)*conv_size(2), conv_size(3))

        x = x.transpose(1, 2)

        self.gru.flatten_parameters()
        output, _ = self.gru(x)

        final = output[:,-1]

        embedding = F.dropout(torch.tanh(self.linear_projection(final)), 0.5, self.training)

        return embedding

