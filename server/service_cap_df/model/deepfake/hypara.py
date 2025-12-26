import numpy as np
import os

class hypara(object):

    def __init__(self):

        #---- Create Label dic
        self.label_dict     = dict(bonafide=1, spoof=0)


        #---- Para for generator and training
        self.eps   = np.spacing(1)
        self.nF_aud    = 64     #Frequency resolution
        self.nT_aud    = 64     #Time resolution
        self.nC_aud    = 3      #Channel resolution
        self.new_dim   = 64

        self.batch_size    = 2000   
        self.start_batch   = 0     
        self.learning_rate = 1e-4  
        self.is_arg        = False
        self.check_every   = 5
        self.class_num     = len(self.label_dict)
        self.epoch_num     = 30
