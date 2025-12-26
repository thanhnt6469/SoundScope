from hypara import *
import torch.nn as nn
import torch

class model_baseline(nn.Module):

    def __init__(self):
        super(model_baseline, self).__init__()

        CLASS_NUM = hypara().class_num
        cnn_conf = [3, 16, 32, 64]  #input, C1, C2, C3
        den_conf = [128, CLASS_NUM]
        

        #--- Conv01
        self.C1_conv     = nn.Conv2d(cnn_conf[0], cnn_conf[1], kernel_size=3, stride=1, padding='same') 
        self.C1_bn       = nn.BatchNorm2d(num_features = cnn_conf[1], eps=1e-05, momentum=0.1)
        self.C1_Relu     = nn.ReLU()
        self.C1_avepool  = nn.AvgPool2d(kernel_size=(2,4), stride=(2,4), count_include_pad=True)
        self.C1_drop     = nn.Dropout(p=0.2)


        #--- Conv02
        self.C2_conv     = nn.Conv2d(cnn_conf[1], cnn_conf[2], kernel_size=3, stride=1, padding='same') 
        self.C2_bn       = nn.BatchNorm2d(num_features = cnn_conf[2], eps=1e-05, momentum=0.1)
        self.C2_Relu     = nn.ReLU()
        self.C2_avepool  = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), count_include_pad=True)
        self.C2_drop     = nn.Dropout(p=0.2)

        #--- Conv02
        self.C3_conv     = nn.Conv2d(cnn_conf[2], cnn_conf[3], kernel_size=3, stride=1, padding='same') 
        self.C3_bn       = nn.BatchNorm2d(num_features = cnn_conf[3], eps=1e-05, momentum=0.1)
        self.C3_Relu     = nn.ReLU()
        self.C3_avepool  = nn.AvgPool2d(kernel_size=(32,38), stride=(2,2), count_include_pad=True)
        self.C3_drop     = nn.Dropout(p=0.2)

        #--- dense 01
        self.D1_dense    = nn.Linear(cnn_conf[3], den_conf[0])  #64, 128
        self.D1_Relu     = nn.ReLU()
        self.D1_drop     = nn.Dropout(p=0.2)

        #--- dense 02
        self.D2_dense    = nn.Linear(den_conf[0], den_conf[1])
        self.D2_softmax  = nn.Softmax(dim=1)

    def forward(self, i_tensor):
        # shape of i_tensor: (batch, channel, freq, time)

        #--- Conv01
        x = self.C1_conv(i_tensor)     
        x = self.C1_bn(x)       
        x = self.C1_Relu(x) 
        x = self.C1_avepool(x)  
        x = self.C1_drop(x)     
        #print(x.size())

        #--- Conv02
        x = self.C2_conv(x)     
        x = self.C2_bn(x)    
        x = self.C2_Relu(x) 
        x = self.C2_avepool(x)  
        x = self.C2_drop(x)     
        #print(x.size())

        #--- Conv02
        x = self.C3_conv(x)     
        x = self.C3_bn(x)    
        x = self.C3_Relu(x) 
        x = self.C3_avepool(x)  
        x = self.C3_drop(x)     
        #print(x.size())

        nB, nC, nF, nT = x.size()
        x = torch.reshape(x, (nB, nC))
        #print(x.size())


        #--- dense 01
        x = self.D1_dense(x)    
        x = self.D1_Relu(x) 
        x = self.D1_drop(x)     
        #print(x.size())

        #--- dense 02
        x = self.D2_dense(x)  
        x = self.D2_softmax(x)
        #print(x.size())
        #exit()
        
        output = x
        return output  # shape: (seq_len, batch, num_class)

