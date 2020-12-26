"https://github.com/coxlab/prednet/blob/master/prednet.py"

import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.autograd import Variable
from model.ConvLSTMCell import ConvLSTMCell
from model.TAAConvLSTMCell import TAAConvLSTMCell
from model.SAAConvLSTMCell import SAAConvLSTMCell

class Model(nn.Module):
    def __init__(self, 
    layer_list, 
    stack_sizes, 
    R_stack_sizes, 
    A_filt_sizes, 
    Ahat_filt_sizes,
    R_filt_sizes, 
    num_past_frames,  #  ？？？？？？？？？？？？？？？？？？传入attention_horizon
    dk, dv, Nh, 
    width,  height, 
    error_activation=nn.ReLU(), A_activation=nn.ReLU(), LSTM_activation=nn.Tanh(), 
    LSTM_inner_activation='hard_sigmoid', output_mode='error', 
    extrap_start_time=None, attention_input_mode='representation', 
    positional_encoding=True, pixel_max=1., forget_bias=1.0):

        '''
        PredNet with TAA/SAAConvLSTM mechanism.
        Extended PredNet baseline implementation - Lotter 2016.
         !!!!!!!!!!!!!!!useless!!!!!!!
        error_activation: activation function for the error (E) units.
        A_activation: activation function for the target (A) and prediction (A_hat) units.
        LSTM_activation: activation function for the cell and hidden states of the LSTM.
        '''
        super(Model, self).__init__()
        self.stack_sizes = stack_sizes   # (2, 48, 96, 192)
        # (n_channels, *tuple(config["model"]["stack_sizes"])),  # [48, 96, 192] # same as R_stack_sizes
        # stack_sizes: number of channels in targets (A) and predictions (Ahat) in each layer of the architecture.
        # Length is the number of layers in the architecture.
        # 体系结构每一层中targets（A）和predictions（Ahat）中的通道数。 长度是体系结构中的层数。
        self.layer_list = layer_list  #   [ConvLSTM, ConvLSTM, SAAConvLSTM, SAAConvLSTM] 
        self.nb_layers = len(stack_sizes)  # 4
        self.R_stack_sizes = R_stack_sizes   # [48, 96, 192]
        # R_stack_sizes: number of channels in the representation (R) modules.
        #  representation（R）模块中的通道数。
        self.A_filt_sizes = A_filt_sizes  # [3, 3, 3]
        # A_filt_sizes: filter sizes for the target (A) modules.
        #  targets（A）模块的过滤器尺寸。
        self.Ahat_filt_sizes = Ahat_filt_sizes  # [3, 3, 3]
        # Ahat_filt_sizes: filter sizes for the prediction (Ahat) modules.
        #  predictions（Ahat）模块的过滤器大小。
        self.R_filt_sizes = R_filt_sizes # [3, 3, 3]
        # R_filt_sizes: filter sizes for the representation (R) modules. 
        #  representation（R）模块的过滤器大小。
        self.pixel_max = pixel_max  
        #  最大像素值。
        self.LSTM_inner_activation = LSTM_inner_activation  
        # LSTM_inner_activation: activation function for the gates in the LSTM.
        #  LSTM中门的激活功能。
        self.output_mode = output_mode  
        # output_mode: either 'error', 'prediction', 'all' or layer specification (ex. R2, see below).
        # Controls what is outputted by the PredNet.
        # If 'error', the mean response of the error (E) units of each layer will be outputted.
        #            That is, the output shape will be (batch_size, nb_layers).
        # If 'prediction', the frame prediction will be outputted.
        
        # “error”，“predictions”，“all”或指定的层（例如R2，请参见下文）。 控制PredNet输出的内容。 
        # 如果为“error”，则将输出每层 error（E）单元的平均响应。 也就是说，输出形状将为（batch_size，nb_layers）。 
        # 如果为“predictions”，则将输出帧predictions。
        self.extrap_start_time = extrap_start_time  # None
        # extrap_start_time: time step for which model will start extrapolating.
        #         Starting at this time step, the prediction from the previous time step will be treated as the "actual".
        # 模型将开始外推的时间点。从这个时间步开始，上一个时间步的predictions将被视为“实际”。
        self.num_past_frames = num_past_frames   #  4
        #  num_past_frames: number of past frames in the attention calculation (not used, for compatibility with TAAConvLSTM).
        #  attention 计算中过去的帧数（！！！！！！！！！不使用，为了与TAAConvLSTM兼容）。
        self.dk = dk  
        # dk: ratio of number of channels in the key/query to number of channels in the output at each layer 
        # 每一层的key/query的通道数与每层输出的通道数之比
        self.dv = dv  
        #  dv: ratio of number of channels in the value to number of channels in the output at each layer 
        #  dv：value中的通道数与每层输出中的通道数之比
        self.Nh = Nh  
        # 
        self.width = width
        self.height = height
        self.attention_input_mode = attention_input_mode  #  representation  传入cell
        self.positional_encoding = positional_encoding  
        self.forget_bias = forget_bias  # 1.0

        default_output_modes = ['prediction', 'error']
        layer_output_modes = [layer + str(n) for n in range(self.nb_layers) for layer in ['R', 'E', 'A', 'Ahat']]

        if self.output_mode in layer_output_modes:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None
            self.output_layer_num = None

        self.channel_axis = -3
        self.row_axis = -2
        self.column_axis = -1

        for l in range(self.nb_layers):
            if l < self.nb_layers - 1:  # layer没添加完
                if self.layer_list[l] == 'ConvLSTM':
                    cell = ConvLSTMCell(3*self.stack_sizes[l]+self.R_stack_sizes[l+1], 
                    self.R_stack_sizes[l], 
                    self.R_filt_sizes[l])
                    setattr(self, 'cell{}'.format(l), cell)
                elif self.layer_list[l] == 'TAAConvLSTM':
                    cell = TAAConvLSTMCell(2*self.stack_sizes[l]+self.R_stack_sizes[l+1], 
                    self.R_stack_sizes[l], 
                    self.R_filt_sizes[l],
                    self.num_past_frames, 
                    self.dk, self.dv, self.Nh, 
                    width, height, 
                    self.attention_input_mode,
                    self.positional_encoding, 
                    self.forget_bias)
                    setattr(self, 'cell{}'.format(l), cell)
                elif self.layer_list[l] == 'SAAConvLSTM':
                    cell = SAAConvLSTMCell(2*self.stack_sizes[l]+self.R_stack_sizes[l+1],
                     self.R_stack_sizes[l], 
                     self.R_filt_sizes[l],
                     self.num_past_frames, 
                     self.dk, self.dv, self.Nh, 
                     width, height, 
                     self.attention_input_mode,
                     self.positional_encoding, 
                     self.forget_bias)
                    setattr(self, 'cell{}'.format(l), cell)
                else:
                    print("Error. Layer type not recognized.")
            else: #l==self.nb_layers
                if self.layer_list[l] == 'ConvLSTM':
                    cell = ConvLSTMCell(3*self.stack_sizes[l], 
                    self.R_stack_sizes[l], 
                    self.R_filt_sizes[l])
                    setattr(self, 'cell{}'.format(l), cell)
                elif self.layer_list[l] == 'TAAConvLSTM':
                    cell = TAAConvLSTMCell(2*self.stack_sizes[l], 
                    self.R_stack_sizes[l], 
                    self.R_filt_sizes[l],
                    self.num_past_frames, 
                    self.dk, self.dv, self.Nh, 
                    width, height,
                    self.attention_input_mode, 
                    self.positional_encoding, 
                    self.forget_bias)
                    setattr(self, 'cell{}'.format(l), cell)
                elif self.layer_list[l] == 'SAAConvLSTM':
                    cell = SAAConvLSTMCell(2*self.stack_sizes[l], 
                    self.R_stack_sizes[l], 
                    self.R_filt_sizes[l],
                    self.num_past_frames, 
                    self.dk, self.dv, self.Nh, 
                    width, height,
                    self.attention_input_mode, 
                    self.positional_encoding, 
                    self.forget_bias)
                    setattr(self, 'cell{}'.format(l), cell)
                else:
                    print("Error. Layer type not recognized.")
            #Creating prediction A_hat from the representation
            conv = nn.Sequential(nn.Conv2d(self.R_stack_sizes[l], self.stack_sizes[l], self.Ahat_filt_sizes[l], padding=1), nn.ReLU())
            setattr(self, 'conv_ahat{}'.format(l), conv)

            if l < self.nb_layers - 1:
                conv = nn.Sequential(nn.Conv2d(2*self.stack_sizes[l], self.stack_sizes[l+1], self.A_filt_sizes[l], padding=1), nn.ReLU())
                setattr(self, 'conv_a{}'.format(l), conv)
            width, height = width//2, height//2;

        self.upsample = nn.Upsample(scale_factor=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def reset_parameters(self):
        for module_name in ['conv_ahat', 'conv_a']:
            for l in range(self.nb_layers):
                module_name = getattr(self, module_name+'{}'.format(l))
                module_name[0].reset_parameters()

    def forward(self, input):
        RepVec = [None] * self.nb_layers
        ErrorVec = [None] * self.nb_layers
        HidVec = [None] * self.nb_layers
        CellVec = [None] * self.nb_layers
        History = []

        batch_size, time_steps, _, h, w = input.size()

        for l in range(self.nb_layers):

            # if self.layer_list[l] == "TAAConvLSTM":
            #     cell = getattr(self, 'cell{}'.format(l))
            #     cell.init_hidden(batch_size=batch_size, hidden=self.R_stack_sizes[l],
            #                                                      shape=(h, w))
            ErrorVec[l] = torch.zeros(batch_size, 2*self.stack_sizes[l], h, w, device=torch.device('cuda:0'))
            RepVec[l] = torch.zeros(batch_size, self.R_stack_sizes[l], h, w, device=torch.device('cuda:0'))
            History.append([torch.zeros(batch_size, self.R_stack_sizes[l], h, w, device=torch.device('cuda:0'))])
            w = w//2
            h = h//2

        total_error = []
        total_prediction = []
        for t in range(time_steps):
            A = input[:,t]

            if(self.extrap_start_time!=None):
                if(t>=self.extrap_start_time):
                    A = frame_prediction

            for l in reversed(range(self.nb_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                Error = ErrorVec[l]
                Rep = RepVec[l]

                if t == 0:
                    Rep = Rep
                    Cell = Rep
                else:
                    Rep = RepVec[l]
                    Cell = CellVec[l]

                if l == self.nb_layers-1:
                    tmpPredNet = torch.cat((Rep, Error), 1)
                    tmpAtt = Error
                else:
                    tmpPredNet = torch.cat((Rep, Error, self.upsample(RepVec[l+1])), 1)
                    tmpAtt = torch.cat((Error, self.upsample(RepVec[l+1])), 1)

                if self.layer_list[l] == "ConvLSTM":
                    Rep, Cell = cell(tmpPredNet, Cell)
                elif self.layer_list[l] == "TAAConvLSTM":
                    His = self.return_history(History, l)
                    Rep, Cell = cell(tmpAtt, Rep, Cell, His)
                    History[l].append(Rep)
                elif self.layer_list[l] == "SAAConvLSTM":
                    Rep, Cell = cell(tmpAtt, Rep, Cell)
                else:
                    print("Error. Layer type not recognized.")


                RepVec[l] = Rep
                CellVec[l] = Cell

            for l in range(self.nb_layers):
                conv_prediction = getattr(self, 'conv_ahat{}'.format(l))
                A_hat = conv_prediction(RepVec[l])

                if l == 0:
                    A_hat = torch.min(A_hat, torch.ones_like(A_hat)*self.pixel_max).cuda() #We clipped the pixels values.
                    frame_prediction = A_hat

                e_up = nn.ReLU()(A_hat-A)
                e_down = nn.ReLU()(A - A_hat)
                Error = torch.cat([e_up, e_down],self.channel_axis)
                ErrorVec[l] = Error

                if l < self.nb_layers - 1:
                    conv_target = getattr(self, 'conv_a{}'.format(l))
                    A = conv_target(Error)
                    A = self.pool(A)

            if self.output_mode == 'error':
                mean_error = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in ErrorVec], 1)
                total_error.append(mean_error)
            else:
                total_prediction.append(frame_prediction)

        if self.output_mode == 'error':
            return torch.stack(total_error, 2) # batch x n_layers x nt
        elif self.output_mode == 'prediction':
            return torch.stack(total_prediction, 1)

    def return_history(self, history, l):
        length = len(history[l])
        idx = np.linspace(max(length-10,1),length,self.num_past_frames,endpoint=False,dtype=np.int)
        if length == 1:
            #We want to avoid 0th representation except at the very beginning when we have no choice.
            idx = idx - 1

        batch_size,c,w,h = history[l][0].shape
        history_tensor = torch.zeros(batch_size,self.num_past_frames,c,h,w, device=torch.device('cuda:0'))
        for t in range(self.num_past_frames):
            history_tensor[:,t,:,:,:] = history[l][idx[t]]
        return history_tensor
