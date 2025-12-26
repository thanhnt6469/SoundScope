#sys.path.insert(0, '/home/cug/ldp7/.local/lib/python2.7/site-packages/librosa/')
import os
import sys
import re
import numpy as np
import librosa
import soundfile as sf
import argparse
import scipy

#----- import hyper-parameters for extracting spectrogram
from hypara_feature import *
from pathlib import Path

project_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(project_dir)
#------------------------------------------------------------ FUNCTIONS DEFINE -----------------------------

def extract_linear_weight(sr, n_fft, n_filter=128, fmin=0.0, fmax=None, dtype=np.float32):

    if fmax is None:
        fmax = float(sr) / 2
    # Initialize the weights
    n_filter = int(n_filter)
    weights = np.zeros((n_filter, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of liner bands - uniformly spaced between limits
    linear_f = np.linspace(fmin, fmax, n_filter + 2)

    fdiff = np.diff(linear_f)
    ramps = np.subtract.outer(linear_f, fftfreqs)

    for i in range(n_filter):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    return weights

def deltas(X_in):  #delta across time dim
    X_out = (X_in[:,:,:, 2:] - X_in[:,:,:, :-2])/10.0
    X_out = X_out[:,:,:, 1:-1] + (X_in[:,:,:, 4:] - X_in[:,:,:, :-4])/5.0
    return X_out

def get_data_mel(indir, outdir, is_del):
   
   #---- Create directory for storing
   store_dir = outdir
   # --- Delete previous store mel
   if os.path.exists(store_dir):
        cmd = f'rm -rf {store_dir}'
        os.system(cmd)
   # -- Create store dir for current file
   if not os.path.exists(store_dir):
      os.makedirs(store_dir)

   #---- Get info from hypara_feature file
   fs    = hypara_feature().fs
   n_bin = hypara_feature().mel_n_bin
   n_win = hypara_feature().mel_n_win 
   n_hop = hypara_feature().mel_n_hop 
   n_fft = hypara_feature().mel_n_fft 
   f_min = hypara_feature().mel_f_min 
   f_max = hypara_feature().mel_f_max 
   htk   = hypara_feature().mel_htk   
   eps   = hypara_feature().eps   
   nT    = hypara_feature().nT
   nF    = hypara_feature().nF

#    audio_data_dir  = project_dir + "/input"
   audio_data_dir  = indir


   #get the list of machine from audio dataset
   org_file_list = os.listdir(audio_data_dir)
   file_list=[]
   for i in range(0,len(org_file_list)):
      isHidden=re.match("\.",org_file_list[i])
      if (isHidden is None):
         file_list.append(org_file_list[i])

   
   #---- Generate mel/log-mel spectrogram for each file in file_name_list
   #xxx = 0
   for file_name in file_list:

       file_open  = os.path.abspath(os.path.join(audio_data_dir, file_name))

       #--- Read and resample
       org_wav, org_fs = librosa.load(file_open)
       if org_fs != 16000:
           org_wav = librosa.resample(org_wav, orig_sr=org_fs, target_sr=fs)
       wav1 = org_wav + eps
       #print(np.shape(wav1))

       while True:
          nTime = len(wav1)
          if nTime < fs*2:
             wav1 = np.concatenate((wav1, wav1), -1)
          else:
             break

       #--- Get lfcc
       stft = librosa.core.stft(wav1, 
                                n_fft      = n_fft,
                                win_length = n_win,
                                hop_length = n_hop,
                                center     = True
                               )
       abs_stft = np.abs(stft)  
       lin_weight = extract_linear_weight(sr=fs, n_fft=n_fft, n_filter=nF, fmin=f_min, fmax=fs/2)
       lin_spec = np.dot(lin_weight, abs_stft)
       lin_spec = librosa.power_to_db(lin_spec) # 10log()
       #lfcc = scipy.fftpack.dct(lin_spec, axis=0, type=2, norm='ortho') # dct on row/freq
       lfcc = scipy.fftpack.dct(lin_spec, axis=1, type=2, norm='ortho') # dct on col/time

     
       #--- reshape and delta 
       [nFreq, nTime] = np.shape(lfcc)
       #lfcc = np.reshape(lfcc, (1, nFreq, nTime, 1))  #nSxnFxnTxnC  : channel dim is final for Tensor/Keras
       lfcc = np.reshape(lfcc, (1, 1, nFreq, nTime)) #nS:nC:nF:nT  : channel dim is the second for Torch
       #print(np.shape(lfcc))
       #exit()

       #--- delta
       if is_del == 'yes':
           lfcc_delta = deltas(lfcc)
           lfcc_delta_delta = deltas(lfcc_delta)
           lfcc = np.concatenate((lfcc[:,:,:, 4:-4],lfcc_delta[:,:,:, 2:-2], lfcc_delta_delta),axis=1) #concat across channel 

       #--- Check time dimension is larger than 64 (2.1s ~ 16000 Hz)
       [nSample, nChannel, nFreq, nTime] = np.shape(lfcc)
       if nTime < nT:
           while True:
                lfcc =  np.concatenate((lfcc, lfcc), axis=-1)
                [nSample, nChannel, nFreq, nTime] = np.shape(lfcc)
                if nTime > nT:
                    break

       #--- Split into 2 second segment
       split_num = 2 + np.floor((nTime-nT)*2/nT) # overlapping
       for m in range(int(split_num)):
           if m == split_num - 1:
               tStop  = nTime
               tStart = nTime - nT
           else:
               tStart = int(m*nT/2)
               tStop  = tStart + nT
           one_image = lfcc[:,:,:,tStart:tStop]

           if m == 0:
               mul_images = one_image
           else:
               mul_images = np.concatenate((mul_images, one_image), axis=0)

       #--store file if eval subset
       save_file  = file_name.split('.')[0] 
       o_data     = mul_images
       file_des   = os.path.abspath(os.path.join(store_dir, save_file))
       np.save(file_des, o_data)

def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s --subset 'train' --outdir '11_mel' --delta 'yes'",
        description="--subset: subset from dataset --> 'train' or 'dev' or 'eval'"
                    "--outdir: Out directory to store spectrogram;"
                    "--delta: Apply delta on spectrogram or not, use terms of 'yes/no'"
    )
    parser.add_argument(
        "--indir", required=True,
        help='Choose out directory'
    )
    parser.add_argument(
        "--outdir", required=True,
        help='Choose out directory'
    )
    parser.add_argument(
        "--delta", required=True,
        help='Apply delta on spectrogram: yes/no'
    )
    return parser

def main():
    print("-------- PARSER ...")
    parser = init_argparse()
    args   = parser.parse_args()
    get_data_mel(args.indir, args.outdir, args.delta)

#------------------------------------------------------------ MAIN FUNCTION -----------------------------
if __name__ == "__main__":
    main()

