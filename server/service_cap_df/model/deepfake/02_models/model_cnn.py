from hypara import *
import torch.nn as nn
import torch

class model_cnn(nn.Module):

    def __init__(self):
        super(model_cnn, self).__init__()

        CLASS_NUM = hypara().class_num
        cnn_conf = [3, 32, 64, 128]  #input, C1, C2, C3
        den_conf = [256, CLASS_NUM]
        
        #--- Conv01
        self.C1_conv     = nn.Conv2d(cnn_conf[0], cnn_conf[1], kernel_size=3, stride=1, padding='same') 
        self.C1_bn       = nn.BatchNorm2d(num_features = cnn_conf[1], eps=1e-05, momentum=0.1)
        self.C1_leakRelu = nn.LeakyReLU(negative_slope=0.1)
        self.C1_ReLU     = nn.ReLU()
        self.C1_avepool  = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), count_include_pad=True)
        self.C1_drop     = nn.Dropout(p=0.1)


        #--- Conv02
        self.C2_bn_01    = nn.BatchNorm2d(num_features = cnn_conf[1], eps=1e-05, momentum=0.1)
        self.C2_conv     = nn.Conv2d(cnn_conf[1], cnn_conf[2], kernel_size=3, stride=1, padding='same') 
        self.C2_bn_02    = nn.BatchNorm2d(num_features = cnn_conf[2], eps=1e-05, momentum=0.1)
        self.C2_leakRelu = nn.LeakyReLU(negative_slope=0.1)
        self.C2_ReLU     = nn.ReLU()
        self.C2_avepool  = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), count_include_pad=True)
        self.C2_drop     = nn.Dropout(p=0.1)

        #--- Conv03
        self.C3_bn_01    = nn.BatchNorm2d(num_features = cnn_conf[2], eps=1e-05, momentum=0.1)
        self.C3_conv     = nn.Conv2d(cnn_conf[2], cnn_conf[3], kernel_size=3, stride=1, padding='same') 
        self.C3_bn_02    = nn.BatchNorm2d(num_features = cnn_conf[3], eps=1e-05, momentum=0.1)
        self.C3_leakRelu = nn.LeakyReLU(negative_slope=0.1)
        self.C3_ReLU     = nn.ReLU()
        self.C3_avepool  = nn.AvgPool2d(kernel_size=(16,16), stride=(16,16), count_include_pad=True)
        self.C3_drop     = nn.Dropout(p=0.1)

        #--- dense 01
        self.D1_dense    = nn.Linear(cnn_conf[3], den_conf[0])  #64, 128
        self.D1_bn       = nn.BatchNorm1d(num_features = den_conf[0], eps=1e-05, momentum=0.1)
        self.D1_leakRelu = nn.LeakyReLU(negative_slope=0.1)
        self.D1_ReLU     = nn.ReLU()
        self.D1_drop     = nn.Dropout(p=0.1)

        #--- dense 02
        self.D2_dense    = nn.Linear(den_conf[0], den_conf[1])
        self.D2_softmax  = nn.Softmax(dim=1)

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

        #--------------------------------------------------------------- C-RNN
        #--- Conv01
        x = self.C1_conv(i_tensor)     
        #x = self.C1_bn(x)       
        #x = self.C1_leakRelu(x) 
        x = self.C1_ReLU(x) 
        x = self.C1_avepool(x)  
        x = self.C1_drop(x)     
        #print(x.size())

        #--- Conv02
        #x = self.C2_bn_01(x)    
        x = self.C2_conv(x)     
        #x = self.C2_bn_02(x)    
        x = self.C2_ReLU(x) 
        #x = self.C2_leakRelu(x) 
        x = self.C2_avepool(x)  
        x = self.C2_drop(x)     
        #print(x.size())

        #--- Conv03
        #x = self.C3_bn_01(x)    
        x = self.C3_conv(x)     
        #x = self.C3_bn_02(x)    
        x = self.C3_ReLU(x) 
        #x = self.C3_leakRelu(x) 
        x = self.C3_avepool(x)  
        x = self.C3_drop(x)     
        #print(x.size())
        nB, nC, nF, nT = x.size()
        x = torch.reshape(x, (nB, nC))
        #print(x.size())
        #exit()

        #--- dense 01
        x = self.D1_dense(x)    
        #x = self.D1_bn(x)       
        #x = self.D1_leakRelu(x) 
        x = self.D1_ReLU(x) 
        x = self.D1_drop(x)     
        #print(x.size())

        #--- dense 02
        x = self.D2_dense(x)  
        x = self.D2_softmax(x)
        #print(x.size())
        
        output = x
        return output  # shape: (seq_len, batch, num_class)


#class weightConstraint(object):
#    def __init__(self):
#        pass
#
#    def __call__(self,module):
#        if hasattr(module,'weight'):
#            print("Entered")
#            w=module.weight.data
#            w=w.clamp(0.5,0.7)
#            module.weight.data=w
#constraints=weightConstraint()
#model=Model()
#model._modules['l3'].apply(constraints)            
