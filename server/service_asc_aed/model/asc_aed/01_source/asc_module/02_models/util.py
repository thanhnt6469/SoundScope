#from hypara import *
import torch.nn as nn
import torch

#--------------------------------------------------- CLASS: Residual Inception
class Res_Inc(nn.Module):
    def __init__(self, i_ch, o_ch, act_type, drop_rate, is_act):
        super(Res_Inc, self).__init__()             
       
        self.act_type = act_type
        self.drop_rate = drop_rate
        self.is_act = is_act

        o_ch_8 = int(o_ch/8)
        o_ch_4 = int(o_ch/4)
        o_ch_2 = int(o_ch/2)

        # activation
        self.relu         = nn.ReLU()
        self.leak_relu_01 = nn.LeakyReLU(negative_slope=0.1)
        self.glu          = nn.GLU(dim=1)
        # dropout
        self.drop_01      = nn.Dropout(p=0.1)
        self.drop_02      = nn.Dropout(p=0.15)
        self.drop_03      = nn.Dropout(p=0.2)
        self.drop_04      = nn.Dropout(p=0.25)
        # pooling 
        self.avepool_3x3  = nn.AvgPool2d(kernel_size=(3,3), stride=(1,1), count_include_pad=True, padding=(1,1)) #no reduce size 
        self.avepool_3x1  = nn.AvgPool2d(kernel_size=(3,1), stride=(1,1), count_include_pad=True, padding=(1,0))
        self.avepool_1x3  = nn.AvgPool2d(kernel_size=(1,3), stride=(1,1), count_include_pad=True, padding=(0,1))

        self.maxpool_3x3  = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1)) #no reduce size 
        self.maxpool_3x1  = nn.MaxPool2d(kernel_size=(3,1), stride=(1,1), padding=(1,0))
        self.maxpool_1x3  = nn.MaxPool2d(kernel_size=(1,3), stride=(1,1), padding=(0,1))

        self.maxpool_2x2  = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) #reduce size
        # batchnorm
        self.bn_f_01         = nn.BatchNorm2d(num_features = o_ch,   eps=1e-05, momentum=0.1)
        self.bn_f_02         = nn.BatchNorm2d(num_features = o_ch,   eps=1e-05, momentum=0.1)
        self.bn_f_03         = nn.BatchNorm2d(num_features = o_ch,   eps=1e-05, momentum=0.1)
        self.bn_f_04         = nn.BatchNorm2d(num_features = o_ch,   eps=1e-05, momentum=0.1)
        self.bn_f_05         = nn.BatchNorm2d(num_features = o_ch,   eps=1e-05, momentum=0.1)
        self.bn_f_06         = nn.BatchNorm2d(num_features = o_ch,   eps=1e-05, momentum=0.1)
        self.bn_f_07         = nn.BatchNorm2d(num_features = o_ch,   eps=1e-05, momentum=0.1)
        self.bn_f_08           = nn.BatchNorm2d(num_features = o_ch_2, eps=1e-05, momentum=0.1)

        #convolution
        self.conv_1x1_f   = nn.Conv2d(i_ch, o_ch,   kernel_size=(1,1), stride=1, padding='same')  #sc

        self.conv_1x1_f2  = nn.Conv2d(i_ch, o_ch_2, kernel_size=(1,1), stride=1, padding='same') #x_m 

        self.conv_1x1_f8_01_01  = nn.Conv2d(o_ch_2, o_ch_8, kernel_size=(1,1), stride=1, padding='same') #kxk l1
        self.conv_1x1_f8_01_02  = nn.Conv2d(o_ch_2, o_ch_8, kernel_size=(1,1), stride=1, padding='same') 
        self.conv_3x3_f2_01     = nn.Conv2d(o_ch_2, o_ch_2, kernel_size=(3,3), stride=1, padding='same') 
        self.conv_5x5_f4_01     = nn.Conv2d(o_ch_2, o_ch_4, kernel_size=(5,5), stride=1, padding='same') 

        self.conv_1x1_f8_02_01  = nn.Conv2d(o_ch, o_ch_8, kernel_size=(1,1), stride=1, padding='same') #kxk l2
        self.conv_1x1_f8_02_02  = nn.Conv2d(o_ch, o_ch_8, kernel_size=(1,1), stride=1, padding='same') 
        self.conv_3x3_f2_02     = nn.Conv2d(o_ch, o_ch_2, kernel_size=(3,3), stride=1, padding='same') 
        self.conv_5x5_f4_02     = nn.Conv2d(o_ch, o_ch_4, kernel_size=(5,5), stride=1, padding='same') 

        self.conv_1x1_f8_03_01  = nn.Conv2d(o_ch_2, o_ch_8, kernel_size=(1,1), stride=1, padding='same') #1xk l1
        self.conv_1x1_f8_03_02  = nn.Conv2d(o_ch_2, o_ch_8, kernel_size=(1,1), stride=1, padding='same') 
        self.conv_1x3_f2_03     = nn.Conv2d(o_ch_2, o_ch_2, kernel_size=(1,3), stride=1, padding='same') 
        self.conv_1x5_f4_03     = nn.Conv2d(o_ch_2, o_ch_4, kernel_size=(1,5), stride=1, padding='same') 

        self.conv_1x1_f8_04_01  = nn.Conv2d(o_ch, o_ch_8, kernel_size=(1,1), stride=1, padding='same') #1xk l2
        self.conv_1x1_f8_04_02  = nn.Conv2d(o_ch, o_ch_8, kernel_size=(1,1), stride=1, padding='same') 
        self.conv_1x3_f2_04     = nn.Conv2d(o_ch, o_ch_2, kernel_size=(1,3), stride=1, padding='same') 
        self.conv_1x5_f4_04     = nn.Conv2d(o_ch, o_ch_4, kernel_size=(1,5), stride=1, padding='same') 


        self.conv_1x1_f8_05_01  = nn.Conv2d(o_ch_2, o_ch_8, kernel_size=(1,1), stride=1, padding='same') #kx1 l1
        self.conv_1x1_f8_05_02  = nn.Conv2d(o_ch_2, o_ch_8, kernel_size=(1,1), stride=1, padding='same') 
        self.conv_3x1_f2_05     = nn.Conv2d(o_ch_2, o_ch_2, kernel_size=(3,1), stride=1, padding='same') 
        self.conv_5x1_f4_05     = nn.Conv2d(o_ch_2, o_ch_4, kernel_size=(5,1), stride=1, padding='same') 

        self.conv_1x1_f8_06_01  = nn.Conv2d(o_ch, o_ch_8, kernel_size=(1,1), stride=1, padding='same') #kx1 l2
        self.conv_1x1_f8_06_02  = nn.Conv2d(o_ch, o_ch_8, kernel_size=(1,1), stride=1, padding='same') 
        self.conv_3x1_f2_06     = nn.Conv2d(o_ch, o_ch_2, kernel_size=(3,1), stride=1, padding='same') 
        self.conv_5x1_f4_06     = nn.Conv2d(o_ch, o_ch_4, kernel_size=(5,1), stride=1, padding='same') 

        self.apply(self._init_weights)

    def _init_weights(self, module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                module.weight.data.normal_(mean=0.0, std=0.1)
                module.bias.data.normal_(mean=0.0, std=0.1)

    def l_normalize(self, x, axes_sel):
        mean  = x.mean(dim=axes_sel)
        var   = x.std(dim=axes_sel)
        [nS, nC, nF, nT] = x.size()

        mean = torch.reshape(mean, (nS, 1, nF, 1))
        var  = torch.reshape(mean, (nS, 1, nF, 1))

        x_std = (x - mean)/torch.sqrt(var)
        x_nor = 0.6*x + x_std
        return x_nor

    def forward(self, i_x):   #f/8:f/2:f/4:f/8
        #------------------ shortcut branch
        x_sc = self.conv_1x1_f(i_x)
        x_sc = self.bn_f_01(x_sc)
        if self.act_type=='glu':
           x_sc = self.glu(x_sc)
        elif self.act_type=='leak_relu':
           x_sc = self.leak_relu_01(x_sc)
        else:   
           x_sc = self.relu(x_sc)
        x_sc = self.avepool_3x3(x_sc)
        #print('================= RES_INC')
        #print('input:', i_x.size())
        #print('short cut:', x_sc.size())
        x_sc = self.l_normalize(x_sc, (1,3)) #normalize across frequence

        #------------------ main branch
        #--------first conv
        xm = self.conv_1x1_f2(i_x)
        xm = self.bn_f_08(xm)
        if self.act_type=='glu':
           xm = self.glu(xm)
        elif self.act_type=='leak_relu':
           xm = self.leak_relu_01(xm)
        else:   
           xm = self.relu(xm)
        xm = self.l_normalize(xm, (1,3)) #normalize across frequence
        #print('xm l1:', xm.size())
        
        #--------Kernel branch
        #--- kxk
        x_kk_01 = self.conv_1x1_f8_01_01(xm)
        x_kk_02 = self.conv_3x3_f2_01(xm)
        x_kk_03 = self.conv_5x5_f4_01(xm)
        x_kk_04 = self.maxpool_3x3(xm)
        x_kk_04 = self.conv_1x1_f8_01_02(x_kk_04)
        x_kk = torch.cat((x_kk_01, x_kk_02, x_kk_03, x_kk_04), 1) #concat across channel dim
        x_kk = self.bn_f_02(x_kk)
        #print('kxk l1:', x_kk.size())

        x_kk_01 = self.conv_1x1_f8_02_01(x_kk)
        x_kk_02 = self.conv_3x3_f2_02(x_kk)
        x_kk_03 = self.conv_5x5_f4_02(x_kk)
        x_kk_04 = self.maxpool_3x3(x_kk)
        x_kk_04 = self.conv_1x1_f8_02_02(x_kk_04)
        x_kk = torch.cat((x_kk_01, x_kk_02, x_kk_03, x_kk_04), 1) #concat across channel dim
        x_kk = self.bn_f_03(x_kk)

        if self.act_type=='glu':
           x_kk = self.glu(x_kk)
        elif self.act_type=='leak_relu':
           x_kk = self.leak_relu_01(x_kk)
        else:   
           x_kk = self.relu(x_kk)
        x_kk = self.avepool_3x3(x_kk)
        #print('kxk l2:', x_kk.size())
        x_kk = self.l_normalize(x_kk, (1,3)) #normalize across frequence

        #--- 1xk
        x_1k_01 = self.conv_1x1_f8_03_01(xm)
        x_1k_02 = self.conv_1x3_f2_03(xm)
        x_1k_03 = self.conv_1x5_f4_03(xm)
        x_1k_04 = self.maxpool_1x3(xm)
        x_1k_04 = self.conv_1x1_f8_03_02(x_1k_04)
        x_1k = torch.cat((x_1k_01, x_1k_02, x_1k_03, x_1k_04), 1) #concat across channel dim
        x_1k = self.bn_f_04(x_1k)
        #print('1xk l1:', x_1k.size())

        x_1k_01 = self.conv_1x1_f8_04_01(x_1k)
        x_1k_02 = self.conv_1x3_f2_04(x_1k)
        x_1k_03 = self.conv_1x5_f4_04(x_1k)
        x_1k_04 = self.maxpool_1x3(x_1k)
        x_1k_04 = self.conv_1x1_f8_04_02(x_1k_04)
        x_1k = torch.cat((x_1k_01, x_1k_02, x_1k_03, x_1k_04), 1) #concat across channel dim
        x_1k = self.bn_f_05(x_1k)

        if self.act_type=='glu':
           x_1k = self.glu(x_1k)
        elif self.act_type=='leak_relu':
           x_1k = self.leak_relu_01(x_1k)
        else:   
           x_1k = self.relu(x_1k)
        x_1k = self.avepool_1x3(x_1k)
        #print('1xk l2:', x_1k.size())
        x_1k = self.l_normalize(x_1k, (1,3)) #normalize across frequence

        #--- kx1
        x_k1_01 = self.conv_1x1_f8_05_01(xm)
        x_k1_02 = self.conv_3x1_f2_05(xm)
        x_k1_03 = self.conv_5x1_f4_05(xm)
        x_k1_04 = self.maxpool_3x1(xm)
        x_k1_04 = self.conv_1x1_f8_05_02(x_k1_04)
        x_k1 = torch.cat((x_k1_01, x_k1_02, x_k1_03, x_k1_04), 1) #concat across channel dim
        x_k1 = self.bn_f_06(x_k1)
        #print('kx1 l1:', x_k1.size())

        x_k1_01 = self.conv_1x1_f8_06_01(x_k1)
        x_k1_02 = self.conv_3x1_f2_06(x_k1)
        x_k1_03 = self.conv_5x1_f4_06(x_k1)
        x_k1_04 = self.maxpool_3x1(x_k1)
        x_k1_04 = self.conv_1x1_f8_06_02(x_k1_04)
        x_k1 = torch.cat((x_k1_01, x_k1_02, x_k1_03, x_k1_04), 1) #concat across channel dim
        x_k1 = self.bn_f_07(x_k1)

        if self.act_type=='glu':
           x_k1 = self.glu(x_k1)
        elif self.act_type=='leak_relu':
           x_k1 = self.leak_relu_01(x_k1)
        else:   
           x_k1 = self.relu(x_k1)

        x_k1 = self.avepool_3x1(x_k1)
        #print('kx1 l2:', x_k1.size())
        x_k1 = self.l_normalize(x_k1, (1,3)) #normalize across frequence

        #------- add all branches
        o_x = torch.add(x_sc, x_kk)   #add
        o_x = torch.add(o_x, x_1k)   #add
        o_x = torch.add(o_x, x_k1)   #add
        if self.act_type=='glu':                       #activation
           o_x = self.glu(o_x)
        elif self.act_type=='leak_relu':
           o_x = self.leak_relu_01(o_x)
        else:   
           o_x = self.relu(o_x)

        if self.is_act == 1:
            o_x = self.maxpool_2x2(o_x)              #pooling and reduce size
            #print('o_x:', x_k1.size())

            if self.drop_rate==1:
               o_x = self.drop_01(o_x)
            elif self.drop_rate==2:
               o_x = self.drop_02(o_x)
            elif self.drop_rate==3:
               o_x = self.drop_03(o_x)
            else:   
               o_x = self.drop_04(o_x)

            o_x = self.l_normalize(o_x, (1,3))
        
        return o_x

#--------------------------------------------------- CLASS: Double Inception
class Doub_Inc(nn.Module):
    def __init__(self, i_ch, o_ch, act_type, drop_rate, is_act):
        super(Doub_Inc, self).__init__()             
        #setup para
        self.is_act = is_act
        self.act_type = act_type
        self.drop_rate = drop_rate
        # activation
        self.relu         = nn.ReLU()
        self.leak_relu_01 = nn.LeakyReLU(negative_slope=0.1)
        self.glu          = nn.GLU(dim=1)
        # dropout
        self.drop_01      = nn.Dropout(p=0.1)
        self.drop_02      = nn.Dropout(p=0.15)
        self.drop_03      = nn.Dropout(p=0.2)
        self.drop_04      = nn.Dropout(p=0.25)
        # pooling 
        self.maxpool_3x3  = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1)) #no reduce size 
        self.maxpool_2x4  = nn.MaxPool2d(kernel_size=(2,4), stride=(2,4)) #reduce size
        self.maxpool_2x2  = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) #reduce size
        self.avepool_2x2  = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)) #reduce size
        # batchnorm
        self.bn_f_00         = nn.BatchNorm2d(num_features = i_ch,   eps=1e-05, momentum=0.1)
        self.bn_f_01         = nn.BatchNorm2d(num_features = o_ch,   eps=1e-05, momentum=0.1)
        self.bn_f_02         = nn.BatchNorm2d(num_features = o_ch,   eps=1e-05, momentum=0.1)
        #convolution
        o_ch_8 = int(o_ch/8)
        o_ch_4 = int(o_ch/4)
        o_ch_2 = int(o_ch/2)

        self.conv_1x1_f8_01_01  = nn.Conv2d(i_ch, o_ch_8, kernel_size=(1,1), stride=1, padding='same') 
        self.conv_1x1_f8_01_02  = nn.Conv2d(i_ch, o_ch_8, kernel_size=(1,1), stride=1, padding='same') 
        self.conv_3x3_f2_01  = nn.Conv2d(i_ch, o_ch_2, kernel_size=(3,3), stride=1, padding='same') 
        self.conv_1x4_f4_01  = nn.Conv2d(i_ch, o_ch_4, kernel_size=(4,1), stride=1, padding='same') 

        self.conv_1x1_f8_02_01  = nn.Conv2d(o_ch, o_ch_8, kernel_size=(1,1), stride=1, padding='same') 
        self.conv_1x1_f8_02_02  = nn.Conv2d(o_ch, o_ch_8, kernel_size=(1,1), stride=1, padding='same') 
        self.conv_3x3_f2_02  = nn.Conv2d(o_ch, o_ch_2, kernel_size=(3,3), stride=1, padding='same') 
        self.conv_1x4_f4_02  = nn.Conv2d(o_ch, o_ch_4, kernel_size=(4,1), stride=1, padding='same') 

        self.apply(self._init_weights)

    def _init_weights(self, module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                module.weight.data.normal_(mean=0.0, std=0.1)
                module.bias.data.normal_(mean=0.0, std=0.1)

    def l_normalize(self, x, axes_sel):
        mean  = x.mean(dim=axes_sel)
        var   = x.std(dim=axes_sel)
        [nS, nC, nF, nT] = x.size()

        mean = torch.reshape(mean, (nS, 1, nF, 1))
        var  = torch.reshape(mean, (nS, 1, nF, 1))

        x_std = (x - mean)/torch.sqrt(var)
        x_nor = 0.6*x + x_std
        return x_nor

    def forward(self, i_x):   #f/8:f/2:f/4:f/8
        #--- kxk
        #i_x = self.bn_f_00(i_x)
        x_kk_01 = self.conv_1x1_f8_01_01(i_x)
        x_kk_02 = self.conv_3x3_f2_01(i_x)
        x_kk_03 = self.conv_1x4_f4_01(i_x)
        x_kk_04 = self.maxpool_3x3(i_x)
        x_kk_04 = self.conv_1x1_f8_01_02(x_kk_04)
        x_kk = torch.cat((x_kk_01, x_kk_02, x_kk_03, x_kk_04), 1) #concat across channel dim
        x_kk = self.bn_f_01(x_kk)

        x_kk_01 = self.conv_1x1_f8_02_01(x_kk)
        x_kk_02 = self.conv_3x3_f2_02(x_kk)
        x_kk_03 = self.conv_1x4_f4_02(x_kk)
        x_kk_04 = self.maxpool_3x3(x_kk)
        x_kk_04 = self.conv_1x1_f8_02_02(x_kk_04)

        x_kk = torch.cat((x_kk_01, x_kk_02, x_kk_03, x_kk_04), 1) #concat across channel dim
        x_kk = self.bn_f_02(x_kk)
        #print(x_kk.size())

        #activation
        if self.act_type=='glu':
           x_kk = self.glu(x_kk)
        elif self.act_type=='leak_relu':
           x_kk = self.leak_relu_01(x_kk)
        else:   
           x_kk = self.relu(x_kk)
        if self.is_act == 1:
            #pooling
            x_kk = self.maxpool_2x2(x_kk)
            #drop
            if self.drop_rate==1:
               x_kk = self.drop_01(x_kk)
            elif self.drop_rate==2:
               x_kk = self.drop_02(x_kk)
            elif self.drop_rate==3:
               x_kk = self.drop_03(x_kk)
            else:   
               x_kk = self.drop_04(x_kk)
        #freq normalization
        o_x = self.l_normalize(x_kk, (1,3))
        #print(x_kk.size())
        #print('================ X01')
        o_x = x_kk
        
        return o_x
