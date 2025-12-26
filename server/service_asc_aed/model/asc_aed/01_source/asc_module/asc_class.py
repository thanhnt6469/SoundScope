#import urllib.request
#from flask import Flask, flash, request, redirect, url_for, render_template
#from werkzeug.utils import secure_filename
#import numpy as np
#import moviepy.editor as mp
#from pydub import AudioSegment
#import tablib
#import csv

#import operator
#import re
#import sys 


import numpy as np
import librosa

import os
import argparse
#from hypara import *
import json 
from scipy import io
from model_cnn import *


class hypara(object):

    def __init__(self):

        # ----File directories
        self.model_dir     = './01_source/asc_module/01_golden_model/'
        self.util_dir      = './01_source/asc_module/02_util/'

        #---- Para for generating spectrograma
        self.fs = 32000
        #01/ log-mel
        self.mel_n_bin = 128
        self.mel_n_win = 2048 
        self.mel_n_fft = 4096
        self.mel_f_min = 10
        self.mel_f_max = self.fs/2
        self.mel_htk   = False
        self.mel_n_hop = 1024

        ##02/ gam
        self.gam_n_bin = 128
        self.gam_n_win = 2048 
        self.gam_n_fft = 4096
        self.gam_f_min = 10
        self.gam_f_max = self.fs/2
        self.gam_htk   = False
        self.gam_n_hop = 1024
        #
        ##03/ cqt
        self.cqt_bins_per_octave = 24
        self.cqt_n_bin = 128
        self.cqt_f_min = 10
        self.cqt_n_hop = 1024
        

        #---- Para for inference process
        #self.learning_rate = 1e-3
        self.class_num  = 15
        self.group_num  = 8
        #self.label_dict = dict(airport=0, bus=1, metro=2, metro_station=3, park=4, public_square=5, shopping_mall=6, street_pedestrian=7, street_traffic=8, tram=9, firework=10, sport=11, music=12, noise=13, riot=14)

        self.class_mem_15 = ['in_airport', 'on bus', 'on metro', 'in metro station', 'in park', 'in public_square', 'in shopping mall', 'on pedestrian street', 'on traffic street', 'on tram',  'in music event', 'in firework event', 'in sport atmosphere', 'in very noise street', 'in riot']
        self.class_mem_8 = ['indoor', 'on transportation', 'outdoor',  'in music event', 'in firework event', 'in sport atmosphere', 'in very noise street', 'in riot']
        

        #---- general para
        self.eps     = np.spacing(1)
        self.dur_eva = 10 #second
        #self.dur_hop = 5 #second
        self.nS     = 1
        self.nF     = 128
        self.nT     = 256
        self.nC     = 3
        self.general_res_fs = 32000

class ASC_component(object):  
    '''
     This class is used for Accoustic Scene Classificaton
     Methods:
        1/delta:     Delta computation on 2-d input
        2/spec_gen:  Generate spectrogram from wav input
        3/asc_reg:   sound scene classifcation
     Notes:
        1/ All parameters are from 'hypara' class above
    ''' 
     
    def __init__(self, dur_hop):
        self.parameters = hypara()
        self.dur_hop = dur_hop

    def get_default_device(self):
       if torch.cuda.is_available():
           return torch.device('cuda')
       else:
           return torch.device('cpu')

    def to_device(self, data, device):
       if isinstance(data, (list,tuple)):
           return [to_device(x, device) for x in data]
       return data.to(device, non_blocking=True)


    def audio_pre_process(self, seg_dur, audio_file_dir):
        '''
         input:  + audio_file_dir: audio recording file directory (wav, mp3, ...) 
                 + seg_dur:  segment duration splitted from the entire audio recording
         usage:  extract start and stop time index of audio segment
         output: 
                + org_wav: audio as an array (1 channel)
                + org_fs:  original fs
                + segment_info_dict: a dict storing start and stop index of audio segments, start and stop time index of audio segments
        '''

        #Check audio file format
        if audio_file_dir.split('.')[-1]!='wav' and audio_file_dir.split('.')[-1]!='mp3':
            with open(os.path.join("general_report.txt"), "a") as text_file:
                text_file.write('\n----------ERROR: The audio file {} is incorrect format. It should be mp3 or wav --> EXIT\n'.format(file_name))

        #load audio without resample
        org_wav, org_fs = librosa.load(audio_file_dir, sr=None) #set sr=None to not use default resample rate 'sr=22050Hz'

        #Selec 1 channel
        if org_wav.ndim >= 2:
            wav  = org_wav[:, 0] 
        else:
            wav = org_wav

        #Check the length of the recording
        if len(wav)/org_fs < self.parameters.dur_eva:
            with open(os.path.join("general_report.txt"), "a") as text_file:
                text_file.write('\n----------WARNING: The audio recording must be equal or larger than {} seconds --> DUPLICATE wav file\n'.format(self.parameters.dur_eva))
            
            while True:
                if len(wav)/org_fs < self.parameters.dur_eva:
                    wav = np.concatenate((wav, wav))
                else:
                    wav = wav[0:self.parameters.dur_eva*org_fs]
                    break

        split_num= int((len(wav)/org_fs-self.parameters.dur_eva)/self.dur_hop) + 1
        #print(split_num, len(wav), org_fs)
        segment_info_dict = {}
        for ind_dur in range(0, split_num):
            #if ind_dur == split_num-1:
            #    str_ind_dur = (len(wav)/org_fs-self.parameters.dur_eva)
            #    end_ind_dur = len(wav)/org_fs
            #else:
            str_ind_dur = ind_dur*self.dur_hop
            end_ind_dur = (str_ind_dur + self.parameters.dur_eva)
            #print(str_ind_dur, end_ind_dur)
            segment_info_dict['segment_'+str(ind_dur)] = [int(str_ind_dur*org_fs), int(end_ind_dur*org_fs), str_ind_dur, end_ind_dur]
         
        #print(segment_info_dict) 
        return wav, org_fs, segment_info_dict

    #def delta(self, X_in):
    #    '''
    #     X_in:   2-D input data (nF:nT), nF: Frequency resolution, nT: Time resolution 
    #     usage:  compuate delta of 2-D input
    #     return: X_out: 2-D output data (nF:nT), nF: Frequency resolution, nT: Time resolution  
    #    '''
    #    X_out = (X_in[:,:,2:,:] - X_in[:,:,:-2,:])/10.0
    #    X_out = X_out[:,:,1:-1,:] + (X_in[:,:,4:,:] - X_in[:,:,:-4,:])/5.0

    #    return X_out

    def delta(self, X_in): #for input format: nF:nT
        '''
         X_in:   2-D input data (nF:nT), nF: Frequency resolution, nT: Time resolution 
         usage:  compuate delta of 2-D input
         return: X_out: 2-D output data (nF:nT), nF: Frequency resolution, nT: Time resolution  
        '''
        X_out = (X_in[:,2:] - X_in[:,:-2])/10.0
        X_out = X_out[:,1:-1] + (X_in[:,4:] - X_in[:,:-4])/5.0
        return X_out


    def spec_gen(self, i_audio_segment, org_fs):
        '''
         i_audio_segment:     input audio segment as an array 
         usage:               generate Mel spectrogram from audio
         return:              + mel_spec: spectrogram (nS:nF:nT:nC), nS:segment number, nF: Frequency resolution, nT: Time resolution, nC: Channel resolution
                              + gam_spec: spectrogram (nS:nF:nT:nC), nS:segment number, nF: Frequency resolution, nT: Time resolution, nC: Channel resolution
                              + cqt_spec: spectrogram (nS:nF:nT:nC), nS:segment number, nF: Frequency resolution, nT: Time resolution, nC: Channel resolution
                              + nS=1, nF=128,nT=256,nC=3
        '''

        res_wav   = librosa.core.resample(i_audio_segment, orig_sr=org_fs, target_sr=self.parameters.general_res_fs) + self.parameters.eps
        #print(len(res_wav), org_fs, self.parameters.general_res_fs, len(i_audio_segment))
        stft_spec = librosa.core.stft(res_wav, 
                                      n_fft      = self.parameters.mel_n_fft,
                                      win_length = self.parameters.mel_n_win,
                                      hop_length = self.parameters.mel_n_hop,
                                      center     = True
                                     )
        stft_spec = np.abs(stft_spec)

        # correct stft spec
        freqs = librosa.core.fft_frequencies(sr=self.parameters.general_res_fs, n_fft=self.parameters.mel_n_fft)
        stft_spec  = librosa.perceptual_weighting(stft_spec ** 2, freqs, ref=1.0, amin=1e-10, top_db=80.0)
        
        # mel
        mel  = librosa.feature.melspectrogram(S      = stft_spec, 
                                              sr     = self.parameters.general_res_fs, 
                                              n_mels = self.parameters.mel_n_bin, 
                                              fmax   = self.parameters.mel_f_max
                                             )

        # reshape

        #mel delta
        mel_delta       = self.delta(mel)
        mel_delta_delta = self.delta(mel_delta)
        ### mel_spec        = np.concatenate((mel[:,:,4:-4,:],mel_delta[:,:,2:-2,:], mel_delta_delta),axis=-1) #for tensorflow

        [nFreq, nTime] = np.shape(mel)
        mel = np.reshape(mel, (1, 1, nFreq, nTime))

        [nFreq, nTime] = np.shape(mel_delta)
        mel_delta = np.reshape(mel_delta, (1, 1, nFreq, nTime))

        [nFreq, nTime] = np.shape(mel_delta_delta)
        mel_delta_delta = np.reshape(mel_delta_delta, (1, 1, nFreq, nTime))


        mel_spec = np.concatenate((mel[:,:,:,4:-4], mel_delta[:,:,:,2:-2], mel_delta_delta),axis=1)

        ### #--- GAM
        ### gam_filter_str = io.loadmat(self.parameters.util_dir+'cof_4096_32000.mat')
        ### gam_filter     = gam_filter_str['data']
        ### gam = 1/self.parameters.gam_n_fft*np.matmul(gam_filter,stft_spec)
        ### 
        ### [nFreq, nTime] = np.shape(gam)
        ### gam = np.reshape(gam, (1, nFreq, nTime, 1))

        ### gam_delta = self.delta(gam)
        ### gam_delta_delta = self.delta(gam_delta)
        ### gam_spec = np.concatenate((gam[:,:,4:-4,:],gam_delta[:,:,2:-2,:], gam_delta_delta),axis=-1)

        ### #---- CQT
        ### cqt = np.abs(librosa.cqt(res_wav,
        ###                         sr              = self.parameters.general_res_fs,
        ###                         fmin            = self.parameters.cqt_f_min,
        ###                         n_bins          = self.parameters.cqt_n_bin,
        ###                         bins_per_octave = self.parameters.cqt_bins_per_octave,
        ###                         hop_length      = self.parameters.cqt_n_hop
        ###                         ))


        ### # correct stft spec
        ### freqs = librosa.cqt_frequencies(cqt.shape[0], fmin=self.parameters.cqt_f_min)
        ### cqt  = librosa.perceptual_weighting(cqt**2, freqs, ref=np.max)

        ### # scale [0:1]
        ### #cqt = (cqt - np.min(cqt)) / (np.max(cqt) - np.min(cqt))

        ### # reshape
        ### [nFreq, nTime] = np.shape(cqt)
        ### cqt = np.reshape(cqt, (1, nFreq, nTime, 1))

        ### cqt_delta = self.delta(cqt)
        ### cqt_delta_delta = self.delta(cqt_delta)
        ### cqt_spec = np.concatenate((cqt[:,:,4:-4,:],cqt_delta[:,:,2:-2,:], cqt_delta_delta),axis=-1)

        #print(np.shape(mel_spec), np.shape(gam_spec), np.shape(cqt_spec))

        ### mel_spec = mel_spec[:,:,0:256,:]
        ### gam_spec = gam_spec[:,:,0:256,:]
        ### cqt_spec = cqt_spec[:,:,0:256,:]
        #print(np.shape(mel_spec), np.shape(gam_spec), np.shape(cqt_spec))

        #print(np.shape(mel_spec), np.shape(gam_spec), np.shape(cqt_spec))

        ### [nS, nF, nT, nC] = np.shape(mel_spec)
        ### if nS!=self.parameters.nS or nF!=self.parameters.nF or nT!=self.parameters.nT or nC!=self.parameters.nC:
        ###     with open(os.path.join("general_report.txt"), "a") as text_file:
        ###         text_file.write('\n----------ERROR: Size of output MEL spectrogram ({}) is incorrect, it should be: 1:128:256:3 --> EXIT\n'.format(np.shape(mel_spec)))

        ### [nS, nF, nT, nC] = np.shape(gam_spec)
        ### if nS!=self.parameters.nS or nF!=self.parameters.nF or nT!=self.parameters.nT or nC!=self.parameters.nC:
        ###     with open(os.path.join("general_report.txt"), "a") as text_file:
        ###         text_file.write('\n----------ERROR: Size of output GAM spectrogram ({}) is incorrect, it should be: 1:128:256:3  --> EXIT\n'.format(np.shape(gam_spec)))

        ### [nS, nF, nT, nC] = np.shape(cqt_spec)
        ### if nS!=self.parameters.nS or nF!=self.parameters.nF or nT!=self.parameters.nT or nC!=self.parameters.nC:
        ###     with open(os.path.join("general_report.txt"), "a") as text_file:
        ###         text_file.write('\n----------ERROR: Size of output CQT spectrogram ({}) is incorrect, it should be: 1:128:256:3 --> EXIT\n'.format(np.shape(cqt_spec)))

        ### return mel_spec, gam_spec, cqt_spec
        return mel_spec


    ### def asc_reg(self, i_mel_spec, i_gam_spec, i_cqt_spec, opt_prod):
    def asc_reg(self, i_mel_spec, opt_prod):
        '''
         input:   +i_mel_spec:  input spectrogram (1:nF=128:nT=256:nC=3), nF: Frequency resolution, nT: Time resolution, nC: Channel resolution
                  +i_gam_spec:  input spectrogram (1:nF=128:nT=256:nC=3), nF: Frequency resolution, nT: Time resolution, nC: Channel resolution
                  +i_cqt_spec:  input spectrogram (1:nF=128:nT=256:nC=3), nF: Frequency resolution, nT: Time resolution, nC: Channel resolution
                   
         usage:   recognize the sound scene 
         return:  predicted_prob: probabilities of 15 sound scene contexts (1:nClass=15) 
        '''

        #model directory

        mel_model_dir = os.path.abspath(os.path.join(os.path.curdir, self.parameters.model_dir,"best_model"))
        ### gam_model_dir = os.path.abspath(os.path.join(os.path.curdir, self.parameters.model_dir,"gam.h5"))
        ### cqt_model_dir = os.path.abspath(os.path.join(os.path.curdir, self.parameters.model_dir,"cqt.h5"))
    
        # reload an available golden model
        ### if os.path.isfile(mel_model_dir) and os.path.isfile(gam_model_dir) and os.path.isfile(cqt_model_dir):
        if os.path.isfile(mel_model_dir):
            mel_model = model_cnn()
            mel_model.load_state_dict(torch.load(mel_model_dir, map_location=torch.device('cpu')))
            ### gam_model = tf.keras.models.load_model(gam_model_dir)
            ### cqt_model = tf.keras.models.load_model(cqt_model_dir)
        else:
            with open(os.path.join("general_report.txt"), "a") as text_file:
                ### text_file.write('\n----------ERROR: Not Available h5 Models (gam, mel, cqt) at: {} --> EXIT \n'.format(self.parameters.model_dir))
                text_file.write('\n----------ERROR: Not Available h5 Models (mel) at: {} --> EXIT \n'.format(self.parameters.model_dir))
            exit()
        #model.summary()
        device     = self.get_default_device()
        mel_model  = self.to_device(mel_model, device)
        mel_model.eval()

        i_mel_spec = torch.tensor(i_mel_spec, dtype=torch.float32)
        i_mel_spec = self.to_device(i_mel_spec, device)

        mel_predicted_prob = mel_model(i_mel_spec) # 1:nClass  if recognition on entire recording --> 1:nClass
        mel_predicted_prob = mel_predicted_prob.detach().cpu().numpy()

        ### gam_predicted_prob = gam_model.predict(i_gam_spec) # 1:nClass  if recognition on entire recording --> 1:nClass
        ### cqt_predicted_prob = cqt_model.predict(i_cqt_spec) # 1:nClass  if recognition on entire recording --> 1:nClass
        #print(np.shape(predicted_prob))

        if   opt_prod == 'mel':
            predicted_prob = mel_predicted_prob
        ### elif opt_prod == 'gam':
        ###     predicted_prob = gam_predicted_prob
        ### elif opt_prod == 'cqt':
        ###     predicted_prob = cqt_predicted_prob
        ### elif opt_prod == 'ens_mean':
        ###     predicted_prob = mel_predicted_prob+gam_predicted_prob+cqt_predicted_prob  #Mean ensemble
        ### elif opt_prod == 'ens_prod':
        ###     predicted_prob = mel_predicted_prob*gam_predicted_prob*cqt_predicted_prob   #Product ensemble
        else:
            with open(os.path.join("general_report.txt"), "a") as text_file:
                text_file.write('----------ERROR:  Option \'{}\' to select predicted probability is incorrect, It must be: \'{}\', \'{}\', \'{}\', \'{}\', or  \'{}\' \n'.format(opt_prod, 'mel', 'gam', 'cqt', 'ens_mean', 'ens_prod'))
                
        predicted_prob = 100*predicted_prob/np.sum(predicted_prob) #1:nClass

        pred_dict_15c = {}
        for i in range(0, self.parameters.class_num):
            pred_dict_15c[self.parameters.class_mem_15[i]] = round(float(predicted_prob[:,i]),3)

        pred_dict_8c = {}
        for i in range(0, self.parameters.group_num):
            if i==0:
                pred_dict_8c[self.parameters.class_mem_8[i]] = round(float(predicted_prob[:,0] + predicted_prob[:,3] + predicted_prob[:,6]),3)
            elif i==1:    
                pred_dict_8c[self.parameters.class_mem_8[i]] = round(float(predicted_prob[:,1] + predicted_prob[:,2] + predicted_prob[:,9]),3)
            elif i==2:
                pred_dict_8c[self.parameters.class_mem_8[i]] = round(float(predicted_prob[:,4] + predicted_prob[:,5] + predicted_prob[:,7] + predicted_prob[:,8]),3)
            else:    
                pred_dict_8c[self.parameters.class_mem_8[i]] = round(float(predicted_prob[:,i+7]),3)

        return pred_dict_15c, pred_dict_8c, predicted_prob

    def asc_extr_emb(self, i_mel_spec, i_gam_spec, i_cqt_spec, opt_emb):
        '''
         input:   +i_mel_spec:  input spectrogram (1:nF=128:nT=256:nC=3), nF: Frequency resolution, nT: Time resolution, nC: Channel resolution
                  +i_gam_spec:  input spectrogram (1:nF=128:nT=256:nC=3), nF: Frequency resolution, nT: Time resolution, nC: Channel resolution
                  +i_cqt_spec:  input spectrogram (1:nF=128:nT=256:nC=3), nF: Frequency resolution, nT: Time resolution, nC: Channel resolution
                   
         usage:   extract embeddings   
         return:  embeddings  
        '''

        #model directory
        mel_model_dir = os.path.abspath(os.path.join(os.path.curdir, self.parameters.model_dir,"mel.h5"))
        gam_model_dir = os.path.abspath(os.path.join(os.path.curdir, self.parameters.model_dir,"gam.h5"))
        cqt_model_dir = os.path.abspath(os.path.join(os.path.curdir, self.parameters.model_dir,"cqt.h5"))
    
        # reload an available golden model
        if os.path.isfile(mel_model_dir) and os.path.isfile(gam_model_dir) and os.path.isfile(cqt_model_dir):
            mel_model = tf.keras.models.load_model(mel_model_dir)
            gam_model = tf.keras.models.load_model(gam_model_dir)
            cqt_model = tf.keras.models.load_model(cqt_model_dir)
            #with open(os.path.join(stored_dir,"valid_acc_log.txt"), "a") as text_file:
            #    text_file.write("\n\n\n Latest model is loaded from: {} \n".format(best_model_dir))
        else:
            with open(os.path.join("general_report.txt"), "a") as text_file:
                text_file.write('\n----------ERROR: Not Available h5 Models (gam, mel, cqt) at: {} --> EXIT \n'.format(self.parameters.model_dir))
            exit()
        #mel_model.summary()
    
        for layer in mel_model.layers:
            if layer.name == 'activation_20':
                mel_emb_layer = layer.output                   #successive_outputs = [layer.output for layer in model.layers[1:]
                break
        for layer in gam_model.layers:
            if layer.name == 'activation_20':
                gam_emb_layer = layer.output                   #successive_outputs = [layer.output for layer in model.layers[1:]
                break
        for layer in cqt_model.layers:
            if layer.name == 'activation_20':
                cqt_emb_layer = layer.output                   #successive_outputs = [layer.output for layer in model.layers[1:]
                break
                    
        vis_mel_model = tf.keras.models.Model(inputs = mel_model.input, outputs = mel_emb_layer)
        vis_gam_model = tf.keras.models.Model(inputs = gam_model.input, outputs = gam_emb_layer)
        vis_cqt_model = tf.keras.models.Model(inputs = cqt_model.input, outputs = cqt_emb_layer)

        mel_embedding = vis_mel_model.predict(i_mel_spec)
        gam_embedding = vis_gam_model.predict(i_gam_spec)
        cqt_embedding = vis_cqt_model.predict(i_cqt_spec)

        if opt_emb=='mel':
            embedding = mel_embedding
        elif opt_emb=='gam':    
            embedding = mel_embedding
        elif opt_emb=='cqt':    
            embedding = mel_embedding
        elif opt_emb=='emb_add':    
            embedding = mel_embedding+gam_embedding+cqt_embedding
        elif opt_emb=='emb_concat':    
            embedding = np.concatenate((mel_embedding, gam_embedding, cqt_embedding), 1)
        else:    
            with open(os.path.join("general_report.txt"), "a") as text_file:
                text_file.write('----------ERROR:  Option \'{}\' to select embedding type is incorrect, It must be: \'{}\', \'{}\', \'{}\', \'{}\', or  \'{}\' \n'.format(opt_emb, 'mel', 'gam', 'cqt', 'emb_add', 'emb_concat'))

        return embedding

    def get_pred_prob(self, audio_file_dir, json_out_dir, opt_prob) :
        '''
         input:   +audio_file_dir:  audio file directory 
                  +json_out_dir:  diretory to store output json file which reports predicted probabilities for each audio segment 
                   
         usage:   recognize the sound scene 
         return:  json file contains predcited probabilities of 15 sound scene contexts 
        '''
        with open(os.path.join("general_report.txt"), "a") as text_file:
            text_file.write('\n----------INFO: ASC: Get label and predicted probabilities of file: {}...\n'.format(audio_file_dir))

        if not os.path.exists(json_out_dir):
            os.makedirs(json_out_dir)

        #Pre-processing audio
        new_wav, org_fs, segment_info_dict = self.audio_pre_process(self.parameters.dur_eva, audio_file_dir)
        file_dir, file_name = os.path.split(audio_file_dir)

        #Recognize on each audio segments
        json_rp15_dict = {}
        json_rp8_dict = {}
        json_rp15_ent_dict={}
        json_rp8_ent_dict={}
        for i in range(0,len(segment_info_dict)):
            with open(os.path.join("general_report.txt"), "a") as text_file:
                text_file.write('----------INFO: ASC: Process on Segment {} ...\n'.format(i))
            st_pt   = segment_info_dict['segment_'+str(i)][0]
            ed_pt   = segment_info_dict['segment_'+str(i)][1]
            seg_wav = new_wav[st_pt:ed_pt]

            ### mel_spec, gam_spec, cqt_spec  = self.spec_gen(seg_wav, org_fs)
            ### pred_dict_15c, pred_dict_8c, predicted_prob = self.asc_reg(mel_spec, gam_spec, cqt_spec, opt_prob)
            mel_spec = self.spec_gen(seg_wav, org_fs)
            pred_dict_15c, pred_dict_8c, predicted_prob = self.asc_reg(mel_spec, opt_prob)

            json_rp15_dict['segment_'+str(i)] = [segment_info_dict['segment_'+str(i)][2], segment_info_dict['segment_'+str(i)][3], pred_dict_15c]
            json_rp8_dict['segment_'+str(i)] = [segment_info_dict['segment_'+str(i)][2], segment_info_dict['segment_'+str(i)][3], pred_dict_8c]

            if i==0:
                avg_predicted_prob = predicted_prob
            else:
                avg_predicted_prob = avg_predicted_prob + predicted_prob


        avg_predicted_prob  = avg_predicted_prob/len(segment_info_dict)       
        avg_predicted_prob_8c = np.array([avg_predicted_prob[:,0] +avg_predicted_prob[:,3] + avg_predicted_prob[:,6],\
                                 avg_predicted_prob[:,1] +avg_predicted_prob[:,2] + avg_predicted_prob[:,9],\
                                 avg_predicted_prob[:,4] +avg_predicted_prob[:,5] + avg_predicted_prob[:,7] + avg_predicted_prob[:,8],\
                                 avg_predicted_prob[:,10],\
                                 avg_predicted_prob[:,11],\
                                 avg_predicted_prob[:,12],\
                                 avg_predicted_prob[:,13],\
                                 avg_predicted_prob[:,14]])

        ### pred_dict_15c_ent = {}
        ### for i in range(0, self.parameters.class_num):
        ###     pred_dict_15c_ent[self.parameters.class_mem_15[i]] = round(float(avg_predicted_prob[:,i]), 3)

        ### pred_dict_8c_ent = {}
        ### for i in range(0, self.parameters.group_num):
        ###     pred_dict_8c_ent[self.parameters.class_mem_8[i]] = round(float(avg_predicted_prob_8c[i,:]), 3)

        ### json_rp15_ent_dict['all_recording'] = [pred_dict_15c_ent]
        ### json_rp8_ent_dict['all_recording']  = [pred_dict_8c_ent]

        #report probabilities on 15 classes for each segment
        json_output_file1 = os.path.join(json_out_dir, file_name.split('.')[0]+'_ASC_label_15class.json')
        with open(json_output_file1, "w") as outfile1:
            json.dump(json_rp15_dict, outfile1)

        ### #report probabilities on 15 classes for all audio recording
        ### json_output_file2 = os.path.join(json_out_dir, file_name.split('.')[0]+'_ASC_label_15class_entire.json')
        ### with open(json_output_file2, "w") as outfile2:
        ###     json.dump(json_rp15_ent_dict, outfile2)

        #report probabilities on 8 classes for each segment
        json_output_file3 = os.path.join(json_out_dir, file_name.split('.')[0]+'_ASC_label_8class.json')
        with open(json_output_file3, "w") as outfile3:
            json.dump(json_rp8_dict, outfile3)

        ### #report probabilities on 8 classes for all audio recording
        ### json_output_file4 = os.path.join(json_out_dir, file_name.split('.')[0]+'_ASC_label_8class_entire.json')
        ### with open(json_output_file4, "w") as outfile4:
        ###     json.dump(json_rp8_ent_dict, outfile4)

    def get_embeddings(self, audio_file_dir, npy_out_dir, opt_emb) :
        '''
         input:   +audio_file_dir:  audio file directory 
                  +json_out_dir:  diretory to store output json file which reports predicted probabilities for each audio segment 
                   
         usage:   recognize the sound scene 
         return:  json file contains predcited probabilities of 15 sound scene contexts 
        '''

        with open(os.path.join("general_report.txt"), "a") as text_file:
            text_file.write('\n----------INFO: ASC: Get embedding ...\n')

        if not os.path.exists(npy_out_dir):
            os.makedirs(npy_out_dir)

        #Pre-processing audio
        new_wav, org_fs, segment_info_dict = self.audio_pre_process(self.parameters.dur_eva, audio_file_dir)
        file_dir, file_name = os.path.split(audio_file_dir)

        #Recognize on each audio segments
        for i in range(0,len(segment_info_dict)):
            with open(os.path.join("general_report.txt"), "a") as text_file:
                text_file.write('----------INFO: ASC: Process on Segment {} ...\n'.format(i))
            st_pt   = segment_info_dict['segment_'+str(i)][0]
            ed_pt   = segment_info_dict['segment_'+str(i)][1]
            seg_wav = new_wav[st_pt:ed_pt]

            mel_spec, gam_spec, cqt_spec  = self.spec_gen(seg_wav, org_fs)
            single_emb = self.asc_extr_emb(mel_spec, gam_spec, cqt_spec, opt_emb)
            if i == 0:
                seq_emb= single_emb
                seq_sta_ind = np.reshape(segment_info_dict['segment_'+str(i)][2], (1,1))
                seq_end_ind = np.reshape(segment_info_dict['segment_'+str(i)][3], (1,1))

            else:
                seq_emb = np.concatenate((seq_emb, single_emb),0)
                seq_sta_ind = np.concatenate((seq_sta_ind, np.reshape(segment_info_dict['segment_'+str(i)][2], (1,1))),0)
                seq_end_ind = np.concatenate((seq_end_ind, np.reshape(segment_info_dict['segment_'+str(i)][3], (1,1))),0)

        file_des = os.path.join(npy_out_dir, file_name.split('.')[0]+'_ASC_emb')
        np.savez(file_des, seq_emb=seq_emb, seq_sta_ind=seq_sta_ind, seq_end_ind=seq_end_ind)
        #return seq_emb

