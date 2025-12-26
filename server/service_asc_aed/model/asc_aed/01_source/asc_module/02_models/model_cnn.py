#from hypara import *
from util import *

import torch.nn as nn
import torch

class model_cnn(nn.Module):

    def __init__(self):
        super(model_cnn, self).__init__()

        #CLASS_NUM = hypara().class_num
        CLASS_NUM = 15
        cnn_config = [3, 64, 64, 128, 128]  #input, C1, C2, C3
        den_config = [128, 256, CLASS_NUM]


        #------------- General layers
        # activation
        self.relu         = nn.ReLU()
        self.leak_relu_01 = nn.LeakyReLU(negative_slope=0.1)
        self.glu          = nn.GLU(dim=1)
        # dropout
        self.drop_01      = nn.Dropout(p=0.1)
        self.drop_02      = nn.Dropout(p=0.15)
        self.drop_03      = nn.Dropout(p=0.2)
        self.drop_04      = nn.Dropout(p=0.25)

        #------------- convolution layers
        self.Doub_Inc_01 = Doub_Inc(cnn_config[0], cnn_config[1], 'relu', 4, 1)
        self.Doub_Inc_02 = Doub_Inc(cnn_config[1], cnn_config[2], 'relu', 3, 1)
        self.Doub_Inc_03 = Doub_Inc(cnn_config[2], cnn_config[3], 'relu', 2, 1)
        self.Doub_Inc_04 = Doub_Inc(cnn_config[3], cnn_config[4], 'relu', 1, 0)

        self.Res_Inc_01  = Res_Inc(cnn_config[1],  cnn_config[2], 'relu', 1, 1)
        self.Res_Inc_02  = Res_Inc(cnn_config[2],  cnn_config[3], 'relu', 1, 1)
        self.Res_Inc_03  = Res_Inc(cnn_config[3],  cnn_config[4], 'relu', 1, 0)

        #self.final_avepool  = nn.AvgPool2d(kernel_size=(8,8), stride=(2,2), count_include_pad=True)
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        #------------- dense layers
        self.Dense_01    = nn.Linear(den_config[0], den_config[1])  #1024-1024
        self.Dense_02    = nn.Linear(den_config[1], den_config[2])  #1024-10
        #------------- softmax
        self.softmax     = nn.Softmax(dim=1)

        #--------------- Initial weight and bias
        self.apply(self._init_weights)


    #---- func to init trainable parameters
    def _init_weights(self, module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                module.weight.data.normal_(mean=0.0, std=0.1)
                module.bias.data.normal_(mean=0.0, std=0.1)
                

    #---- forward function
    def forward(self, i_tensor):
        # shape of i_tensor: (batch, channel, freq, time)
        #---convolution

        #print('YYYYY 01: INPUT', i_tensor.size())
        x = self.Doub_Inc_01(i_tensor)
        #print('YYYYY 02: DOUB', x.size())

        #x = self.Doub_Inc_02(x)
        #x = self.Doub_Inc_03(x)
        #x = self.Doub_Inc_04(x)

        x = self.Res_Inc_01(x)       
        ##print('YYYYY 03: RES01', x.size())
        x = self.Res_Inc_02(x)       
        ##print('YYYYY 04: RES02', x.size())
        x = self.Res_Inc_03(x)       
        ##print('YYYYY 05: RES03', x.size())

        x = self.adapt_pool(x)
        nB, nC, nF, nT = x.size()
        x = torch.reshape(x, (nB, nC))
        x = self.drop_01(x)
        #print('YYYYY 06: pooling', x.size())

        #---Fully connection
        x = self.Dense_01(x)
        x = self.relu(x)
        x = self.drop_01(x)
        #print('YYYYY 07, dense1:', x.size())

        x = self.Dense_02(x)
        x = self.softmax(x)
        #print('YYYYY 08, output:', x.size())
        #exit()

 
        output = x
        return output  # shape: (seq_len, batch, num_class)



