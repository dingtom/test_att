""" Follows the definition from
Convolutional LSTM Network: A Machine Learning Approach for Precipitation
Nowcasting """

"Code based on https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py"

import torch
import torch.nn as nn
from torch.autograd import Variable
from model.TAAConv2d import TAAConv2d

class TAAConvLSTMCell(nn.Module):

    """
    LSTM:
        x - input
        h - hidden representation
        c - memory cell
        f - forget gate
        o - output gate
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, num_past_frames, dk, dv,
                Nh, width, height, attention_input_mode='representation', positional_encoding = True, forget_bias = 1.0):

        super(TAAConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_past_frames = num_past_frames
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.width = width
        self.height = height
        self.attention_input_mode = attention_input_mode
        self.positional_encoding = positional_encoding
        self.forget_bias = forget_bias

        self.padding = int((kernel_size - 1) / 2) # Padding

        self.W_xi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.W_hi = TAAConv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.padding,
                    self.num_past_frames, self.dk, self.dv, self.Nh, self.width, self.height, self.attention_input_mode, self.positional_encoding)
        self.W_xf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.W_hf = TAAConv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.padding,
                    self.num_past_frames, self.dk, self.dv, self.Nh, self.width, self.height, self.attention_input_mode, self.positional_encoding)
        self.W_xc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.W_hc = TAAConv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.padding,
                    self.num_past_frames, self.dk, self.dv, self.Nh, self.width, self.height, self.attention_input_mode, self.positional_encoding)
        self.W_xo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.W_ho = TAAConv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.padding,
                    self.num_past_frames, self.dk, self.dv, self.Nh, self.width, self.height, self.attention_input_mode, self.positional_encoding)
        self.W_ci = nn.Parameter(torch.zeros(1, self.hidden_channels, self.height, self.width, device=torch.device('cuda:0')))
        self.W_cf = nn.Parameter(torch.zeros(1, self.hidden_channels, self.height, self.width, device=torch.device('cuda:0')))
        self.W_co = nn.Parameter(torch.zeros(1, self.hidden_channels, self.height, self.width, device=torch.device('cuda:0')))

    def forward(self, inputs, rep, c, history):

        i_t = torch.sigmoid(self.W_xi(inputs) + self.W_hi(rep,history) + c * self.W_ci)
        f_t = torch.sigmoid(self.W_xf(inputs) + self.W_hf(rep,history) + c * self.W_cf + self.forget_bias)
        c_t = f_t * c + i_t * torch.tanh(self.W_xc(inputs) + self.W_hc(rep,history))
        o_t = torch.sigmoid(self.W_xo(inputs) + self.W_ho(rep,history) +  c_t * self.W_co)
        h_t = o_t * torch.tanh(c_t)


        return h_t, c_t

    # Peephole connection issue


    # https://github.com/pytorch/pytorch/issues/1706
    # def init_hidden(self, batch_size, hidden, shape):
    #     if self.W_ci is None:
    #         self.W_ci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], device=torch.device('cuda:0')))
    #         self.W_cf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], device=torch.device('cuda:0')))
    #         self.W_co = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], device=torch.device('cuda:0')))
    #     else:
    #         assert shape[0] == self.W_ci.size()[2], 'Input Height Mismatched!'
    #         assert shape[1] == self.W_ci.size()[3], 'Input Width Mismatched!'

    def reset_parameters(self):
        self.W_xi.reset_parameters()
        self.W_hi.reset_parameters()
        self.W_xf.reset_parameters()
        self.W_hf.reset_parameters()

        self.W_xc.reset_parameters()
        self.W_hc.reset_parameters()
        self.W_xo.reset_parameters()
        self.W_ho.reset_parameters()
