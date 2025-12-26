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

class SpeakerDiarization:
    def __init__(self, project_dir, output_dir, segment_dir):
        """
        Initialize the SpeakerDiarization class.
        :param project_dir: Directory of current project
        :param segment_dir: Directory containing audio segments for input files (processed through Whisper S2T)
        :param output_dir: Directory where output json for speaker diarization will be saved.
        """
        self.project_dir = project_dir
        self.segment_dir = segment_dir
        self.output_dir = output_dir        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model and tokenizer
        with open(os.path.join(project_dir, "model" ,"configs", "model_config.yaml"), "r") as f:
            config = yaml.safe_load(f)
        self.encoder = EncoderClassifier.from_hparams(source=config['DIARIZER_EMBED'])
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


    def perform_on_audio_file(self, folder_name):
        """
        Process a single folder containing audio segments.

        :param folder_name: Name of the folder containing all audio segments of a single .wav file.
        """
        # --- Process directory------
        folder_path = os.path.join(self.segment_dir, folder_name)
        if not os.path.isdir(folder_path):
            return
        # ------Directory to save output-----------
        output_folder = os.path.join(self.output_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, 'speaker_diarization.json')
        # Get number of speakers
        n_speakers = self.get_n_speakers(folder_name=folder_name)
        # ----------Check special case--------------
        if n_speakers == 0: # No human speech
            with open(output_file, "w") as outfile:
                json.dump([None], outfile)
            return 
        else:
            segment_list = os.listdir(folder_path)
            sorted_segment_list = sorted(
                segment_list,
                key=lambda x: float(x[len(folder_name) + 1:].split('_')[0])
            )
            audio_result = []
            embeddings = []
            for segment_id, segment in enumerate(sorted_segment_list):
                segment_path = os.path.join(folder_path, segment)
                split_list = segment[len(folder_name) + 1:].split("_")
                start_time = float(split_list[0])
                end_time = float(split_list[1])
                try:
                    # Process each segment
                    signal, fs =torchaudio.load(segment_path)
                    segment_embed = self.encoder.encode_batch(signal)  # (n, 1, 192) for n-channel waveform
                    # Remove the second dimension (1) 
                    segment_embed = segment_embed.squeeze(1)    # (n, 192)
                    # Average across the first dimension 
                    segment_embed = segment_embed.mean(dim=0, keepdim=True) # (1, 192)
                    # print(segment_embed, segment_embed)
                    # print(segment_embed.shape)

                    # Define segment result in advanced
                    segment_result = {
                        "id": segment_id,
                        "start": start_time,
                        "end": end_time,
                        "speaker": None
                    }
                    audio_result.append(segment_result)
                    embeddings.append(segment_embed)
                except Exception as e:
                    print(f"Error processing segment {segment}: {e}")
        
            # Gather embedding 
            embeddings = np.concatenate(embeddings, axis=0)
            if embeddings.shape[0] == 1 and len(audio_result) == 1:    # Only one segment
                audio_result[0]['speaker'] = "SPEAKER 1"
                with open(output_file, "w") as outfile:
                    json.dump(audio_result, outfile)
                return 

            # Perform clustering
            cluster_algo = AgglomerativeClustering(n_clusters=n_speakers).fit(embeddings)
            f_label = cluster_algo.labels_

            # Assign label to audio_result
            assert len(audio_result) == len(f_label)
            for idx, segment_result in enumerate(audio_result):
                segment_result['speaker'] = "SPEAKER " + str(f_label[idx] + 1)

            # Save diarization result
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