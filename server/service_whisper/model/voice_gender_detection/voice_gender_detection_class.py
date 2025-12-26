import numpy as np
import os
import json
from sklearn.cluster import AgglomerativeClustering
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from pathlib import Path
import sys
from natsort import natsorted, ns
from dotenv import load_dotenv
import yaml
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample
from huggingface_hub import PyTorchModelHubMixin

## This script is based on the https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
#-----------------------------------------------------------------------------
class SEModule(nn.Module):
    def __init__(self, channels : int , bottleneck : int = 128) -> None:
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input : torch.Tensor) -> torch.Tensor:
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes : int, planes : int, kernel_size : Optional[int] = None, dilation : Optional[int] = None, scale : int = 8) -> None:
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 
    

class ECAPA_gender(nn.Module, PyTorchModelHubMixin):
    def __init__(self, C : int = 1024):
        super(ECAPA_gender, self).__init__()
        self.C = C
        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.fc7 = nn.Linear(192, 2)
        self.pred2gender = {0 : 'male', 1 : 'female'}

    def logtorchfbank(self, x : torch.Tensor) -> torch.Tensor:
        # Preemphasis
        flipped_filter = torch.FloatTensor([-0.97, 1.]).unsqueeze(0).unsqueeze(0).to(x.device)
        x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        x = F.conv1d(x, flipped_filter).squeeze(1)

        # Melspectrogram
        x = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80).to(x.device)(x) + 1e-6
        
        # Log and normalize
        x = x.log()   
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return x
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.logtorchfbank(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.fc7(x)
        
        return x
    
    def load_audio(self, path : str) -> torch.Tensor:
        audio, sr = torchaudio.load(path)
        if sr != 16000:
            audio = resample(audio, sr, 16000)
        return audio
    
    def predict(self, audio : torch.Tensor, device: torch.device) -> torch.Tensor:
        audio = self.load_audio(audio)
        audio = audio.to(device)
        self.eval()

        with torch.no_grad():
            output = self.forward(audio)
            _, pred = output.max(1)
        return self.pred2gender[pred.item()]
#---------------------------------------------------------------------------------------------------

class VoiceGenderDetection:
    def __init__(self, project_dir, output_dir, segment_dir):
        """
        Initialize the VoiceGenderDetection class.
        :param project_dir: Directory of current project
        :param segment_dir: Directory containing audio segments for input files (processed through Whisper S2T)
        :param output_dir: Directory where output json for voice gender detection will be saved.
        """
        self.project_dir = project_dir
        self.segment_dir = segment_dir
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(project_dir, "model", "configs", "model_config.yaml"), "r") as f:
            config = yaml.safe_load(f)
        self.model = ECAPA_gender.from_pretrained(config['VOICE_GENDER_CLASSIFIER']).to(self.device)
        self.model.eval()
        self.diarization_inference = None
        if not os.path.exists(self.segment_dir):
            raise FileNotFoundError("Can not find the directory containing extracted audio segments.")
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError('Can not find the directory containing output previously defined.')


    def get_n_speakers(self, folder_name):
        """
        Get number of speakers from speaker_count.json
        :param output_dir
        """ 
        try:
            with open(os.path.join(self.output_dir, folder_name,'speaker_count.json'),'r') as file:
                n_speakers = json.load(file)[0]
            
            return n_speakers
        except Exception as e:
            print("Error getting number of speakers: ", e)


    def get_diarization_result(self, folder_name):
        """
        Perform voice gender detection based on diarization result to avoid redundant run for duplicated speakers
        """
        try:
            with open(os.path.join(self.output_dir, folder_name,'speaker_diarization.json'),'r') as file:
                diarization_result = json.load(file)
            
            return diarization_result
        except Exception as e:
            print("Error getting diarization result: ", e)



    def perform_on_audio_file(self, folder_name):
        """
        Process a single folder containing audio segments.

        :param folder_name: Name of the folder containing all audio segments of a single .wav file.
        """
        folder_path = os.path.join(self.segment_dir, folder_name)
        if not os.path.isdir(folder_path):
            return
        
        # Get diarization inference for this file
        self.diarization_inference = self.get_diarization_result(folder_name=folder_name)
        if not self.diarization_inference: # No human speech
            with open(os.path.join(self.output_dir, folder_name, 'voice_gender.json'), "w") as outfile:
                json.dump([None], outfile)
            return 
        else:
            segment_list = os.listdir(folder_path)
            sorted_segment_list = sorted(
                segment_list,
                key=lambda x: float(x[len(folder_name) + 1:].split('_')[0])
            )
            audio_result = []
            seen_speakers = {}   # Store gender of previously seen speakers.
            gender = None
            for segment_id, segment in enumerate(sorted_segment_list):
                segment_path = os.path.join(folder_path, segment)
                split_list = segment[len(folder_name) + 1:].split("_")
                start_time = float(split_list[0])
                end_time = float(split_list[1])
                # Define segment result in advanced for this segment
                segment_result = {
                    "id": segment_id,
                    "start": start_time,
                    "end": end_time,
                    "gender": None
                }
                # Get speaker value of diarization result
                speaker_val = next((item["speaker"] for item in self.diarization_inference if item["start"] == start_time \
                                and item["end"] == end_time), None)
                
                # If encounter seen speakers --> Get store gender value withoyt calling model
                if speaker_val in seen_speakers.keys(): 
                    gender = seen_speakers[speaker_val]
                else:
                    try:
                        # Process segment of new speaker
                        # Process each segment
                        signal, fs =torchaudio.load(segment_path)
                        if signal.shape[0] > 1:  # Average channels to one-channel
                            signal = signal.mean(dim=0, keepdim=True)
                        torchaudio.save(segment_path, signal, sample_rate=fs)
                        
                        with torch.no_grad():
                            gender = self.model.predict(segment_path, device=self.device)
                            # print("gender", gender)
                        # Update new speaker's gender to seen_speaker
                        seen_speakers[speaker_val] = gender
                    except Exception as e:
                        print(f"Error processing segment {segment}: {e}")
                
                # Update gender to segment_result
                segment_result['gender'] = gender
                audio_result.append(segment_result)
                

            output_folder = os.path.join(self.output_dir, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, 'voice_gender.json')
            # print("Seen speaker", seen_speakers)
            # Save voice gender result
            with open(output_file, "w") as outfile:
                json.dump(audio_result, outfile)


    def perform_on_all_audio_files(self):
        """
        Process all folders in the segment directory.
        """
        folder_list = os.listdir(self.segment_dir)
        try:
            for folder_name in sorted(folder_list):
                print(f"Processing folder: {folder_name}")
                self.perform_on_audio_file(folder_name)
        except Exception as e:
            print(f"Error processing file {folder_name}.wav: {e}")