import os
#import urllib.request
#from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
#import moviepy.editor as mp
#from pydub import AudioSegment
#import tablib
import csv

import json
import operator

import re
import sys
import librosa

#----- generator
#import tensorflow as tf


#------- Add by Lam Pham: For torch
from natsort import natsorted, ns
from inference import *
from utilities import *

#-------- Add by Lam Pham: For statistics
#---for Statistics
from all_label import *

import json 

class hypara(object):

    def __init__(self):

        #---- general para
        self.eps     = np.spacing(1)
        self.dur_eva = 10 #second
        self.general_res_fs = 32000

class AED_component(object):  
    '''
     This class is used for Accoustic Event Detection 
    ''' 
     
    def __init__(self, dur_hop):
        self.parameters = hypara()
        self.dur_hop = dur_hop

    def audio_pre_process(self, seg_dur, audio_file_dir, file_ext):
        '''
         input:  + audio_file_dir: audio recording file directory (wav, mp3, ...) 
                 + seg_dur:  segment duration splitted from the entire audio recording
                 + file_ext: extention of audio file which should be 'wav' or 'mp3' --> match Librosa toolbox
         usage:  extract start and stop time index of audio segment
         output: 
                + org_wav: audio data as an array (1 channel)
                + org_fs:  original fs
                + segment_info_dict: a dict data storing start and stop index of audio segments, start and stop time index of audio segments
        '''

        #Check audio file format
        if file_ext!='wav' and file_ext!='mp3':
            with open(os.path.join("general_report.txt"), "a") as text_file:
                text_file.write('\n----------ERROR: The audio file {} is incorrect format. It should be mp3 or wav --> EXIT\n'.format(file_name))

        #Load audio recording
        org_wav, org_fs = librosa.load(audio_file_dir, sr=None) #set sr=None to not use default resample rate 'sr=22050Hz'

        #Selec 1 channel
        if org_wav.ndim >= 2:
            wav  = org_wav[:, 0] 
        else:
            wav = org_wav

        #Check the length of the audio recording
        if len(wav)/org_fs < self.parameters.dur_eva:
            with open(os.path.join("general_report.txt"), "a") as text_file:
                text_file.write('\n----------WARNING: The audio recording must be equal or larger than {} seconds --> DUPLICATE wav file \n'.format(self.parameters.dur_eva))

            while True:
                if len(wav)/org_fs < self.parameters.dur_eva:
                    wav = np.concatenate((wav, wav))
                else:
                    wav = wav[0:self.parameters.dur_eva*org_fs]
                    break

        split_num= int(( (len(wav)/org_fs) - self.parameters.dur_eva)/self.dur_hop) + 1

        #Return information of audio segments 
        segment_info_dict = {}
        for ind_dur in range(0, split_num):
            #if ind_dur == split_num-1:
            #    str_ind_dur = (len(wav)/org_fs-self.parameters.dur_eva)
            #    end_ind_dur = len(wav)/org_fs
            #else:
            str_ind_dur = ind_dur*self.dur_hop
            end_ind_dur = (str_ind_dur + self.parameters.dur_eva)
            segment_info_dict['segment_'+str(ind_dur)] = [int(str_ind_dur*org_fs), int(end_ind_dur*org_fs), str_ind_dur, end_ind_dur]
         
        return wav, org_fs, segment_info_dict

    def get_aed_info(self, audio_file_dir, out_dir, opt_info) : #may be we have top-event here
        '''
         input:   +audio_file_dir:  audio file directory 
                  +out_dir:  diretory to store output json file which reports predicted probabilities for each audio segment 
                   
         usage:   recognize the sound scene 
         return:  json file contains predcited probabilities of 10 sound events which present the best predicted probabilities  
        '''
        #Report
        with open(os.path.join("general_report.txt"), "a") as text_file:
            text_file.write('\n----------INFO: AED: Get audio event label, probabilities, and embeddings ...\n')

        #Create outdir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        #Read audio file
        file_dir, file_name = os.path.split(audio_file_dir)
        file_ext = file_name.split('.')[-1]
        new_wav, org_fs, segment_info_dict = self.audio_pre_process(self.parameters.dur_eva, audio_file_dir, file_ext)

        #On each audio segments
        json_rp_dict = {}
        for i in range(0,len(segment_info_dict)):
            #Info report
            with open(os.path.join("general_report.txt"), "a") as text_file:
                text_file.write('----------INFO: AED: Process on Segment {} ...\n'.format(i))

            #Get information for each segment: audio data, time start, time stop    
            st_pt   = segment_info_dict['segment_'+str(i)][0]
            ed_pt   = segment_info_dict['segment_'+str(i)][1]
            seg_wav = new_wav[st_pt:ed_pt]
            
            #Resample 
            res_wav   = librosa.core.resample(seg_wav, orig_sr=org_fs, target_sr=self.parameters.general_res_fs) 

            #Call pre-trained aed model and get label
            _, _, label_dict, embedding = audio_tagging(seg_wav)
            embedding = np.reshape(embedding, (1,-1))

            #Solve the label
            json_rp_dict['segment_'+str(i)] = [segment_info_dict['segment_'+str(i)][2], segment_info_dict['segment_'+str(i)][3], label_dict]

            #Solve the embedding
            if i == 0:
                seq_emb= embedding
                seq_sta_ind = np.reshape(segment_info_dict['segment_'+str(i)][2], (1,1))
                seq_end_ind = np.reshape(segment_info_dict['segment_'+str(i)][3], (1,1))

            else:
                seq_emb = np.concatenate((seq_emb, embedding),0)
                seq_sta_ind = np.concatenate((seq_sta_ind, np.reshape(segment_info_dict['segment_'+str(i)][2], (1,1))),0)
                seq_end_ind = np.concatenate((seq_end_ind, np.reshape(segment_info_dict['segment_'+str(i)][3], (1,1))),0)

       
        #export the embedding

        if opt_info == 'emb' or opt_info == 'both':
            npz_output_file = os.path.join(out_dir, file_name.split('.'+file_ext)[0]+'_AED_emb')
            np.savez(npz_output_file, seq_emb=seq_emb, seq_sta_ind=seq_sta_ind, seq_end_ind=seq_end_ind)
            
        #export the label
        if opt_info == 'label' or opt_info == 'both':
            json_output_file = os.path.join(out_dir, file_name.split('.'+file_ext)[0]+'_AED_lable.json')
            with open(json_output_file, "w") as outfile:
                json.dump(json_rp_dict, outfile)

