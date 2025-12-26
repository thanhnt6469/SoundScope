import numpy as np
import os

class hypara_feature(object):

    def __init__(self):

        #---- Original Dataset Directory
        self.audio_data_train_dir  = '/var/data/storage/datasets/audio/speech/06_ASVspooving/2019_chal/LA/ASVspoof2019_LA_train/flac'
        self.audio_label_train_dir = '/var/data/storage/datasets/audio/speech/06_ASVspooving/2019_chal/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
      
        self.audio_data_dev_dir    = '/var/data/storage/datasets/audio/speech/06_ASVspooving/2019_chal/LA/ASVspoof2019_LA_dev/flac'
        self.audio_label_dev_dir   = '/var/data/storage/datasets/audio/speech/06_ASVspooving/2019_chal/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

        self.audio_data_eval_dir   = '/var/data/storage/datasets/audio/speech/06_ASVspooving/2019_chal/LA/ASVspoof2019_LA_eval/flac'
        self.audio_label_eval_dir  = '/var/data/storage/datasets/audio/speech/06_ASVspooving/2019_chal/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

        #---- Para for generating spectrogram
        #01/ log-mel
        self.mel_n_bin = 64
        self.mel_n_win = 1024 
        self.mel_n_fft = 2048
        self.mel_f_min = 0
        self.mel_f_max = None
        self.mel_htk   = False
        self.mel_n_hop = 512
        
        #02/ cqt
        self.cqt_bins_per_octave = 24
        self.cqt_n_bin = 64
        self.cqt_f_min = 10
        self.cqt_n_hop = 512 

        #03/ other para
        self.eps   = np.spacing(1)
        self.fs    = 16000 #Sample rate
        self.nT    = 64    #Time resolution
        self.nF    = 64   #Frequency resolution
