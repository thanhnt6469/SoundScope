import os
import re
import json
from pathlib import Path
import torchaudio
from transformers import AutoModel, PreTrainedTokenizerFast
from natsort import natsorted
import torch
import argparse
import sys
from dotenv import load_dotenv
import yaml

class AudioCaptioning:
    def __init__(self, project_dir, input_dir, output_dir):
        self.project_dir = project_dir
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model and tokenizer
        with open(os.path.join(project_dir, "model", "configs", "model_config.yaml"), "r") as f:
            config = yaml.safe_load(f)
        # self.model_name = os.getenv("AUDIO_CAPTIONING_MODEL_CLOTHO")
        self.model_name = config['AUDIO_CAPTIONING_MODEL_CLOTHO']
        self.tokenizer_name = config['AUDIO_CAPTIONING_TOKENIZER']
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.tokenizer_name)
        self.tmp_output_dir = None

    def perform_model(self, audio_file_path):
        # Inference on a single audio file
        try:
            wav, sr = torchaudio.load(audio_file_path)
            
            # Check if audio is empty or too short
            if wav.size(1) == 0:
                return "No audio content detected"
            
            wav = torchaudio.functional.resample(wav, sr, self.model.config.sample_rate)
            if wav.size(0) > 1:
                wav = wav.mean(0).unsqueeze(0)
            
            # Check if resampled audio is still valid
            if wav.size(1) == 0:
                return "Audio too short after processing"

            with torch.no_grad():
                word_idxs = self.model(audio=wav, audio_length=[wav.size(1)])

            caption = self.tokenizer.decode(word_idxs[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error in perform_model for {audio_file_path}: {e}")
            return f"Error generating caption: {str(e)}"

    def process_audio_file(self, audio_file, target_length):
        audio_file_path = os.path.join(self.input_dir, audio_file)
        audio_name = audio_file[:-4]
        print(f"Processing file: {audio_name}")
        segment_results = []

        self.tmp_output_dir = "./tmp_segments"
        if not os.path.exists(self.tmp_output_dir):
            os.makedirs(self.tmp_output_dir)
        try:
            waveform, sample_rate = torchaudio.load(audio_file_path)
            audio_length_sec = waveform.size(1) / sample_rate
            if audio_length_sec <= target_length:  # If audio_length <= 10 second
                print("Audio captioning- Audio length < 10 seconds --> Padding to 10 seconds")
                # print("waveform:", waveform.shape)
                padded_audio = self.pad_audio(waveform, sample_rate, target_length=10)
                # print("padded audio", padded_audio.shape)
                tmp_padded_path = os.path.join(self.tmp_output_dir, f"{audio_name}.wav")
                torchaudio.save(tmp_padded_path, padded_audio, sample_rate)
                caption = self.perform_model(tmp_padded_path)
                segment_results.append({"id":0, "start": 0,"end": 10, "caption": caption})

            else:
                segment_samples = target_length * sample_rate
                num_segments = waveform.size(1) // segment_samples

                for i in range(num_segments + 1):
                    start = i * segment_samples   # start (samples)
                    end = start + segment_samples 

                    segment = waveform[:, start:end] if end < waveform.shape[1] else waveform[:, start:]
                    
                    # Check if segment is too short (less than 1 second)
                    segment_length_sec = segment.size(1) / sample_rate
                    if segment_length_sec < 1.0:
                        # Skip segments that are too short
                        print(f"Skipping segment {i}: too short ({segment_length_sec:.2f}s)")
                        continue
                    
                    # Pad segment if it's shorter than target_length
                    if segment_length_sec < target_length:
                        segment = self.pad_audio(segment, sample_rate, target_length)

                    segment_name = f"{audio_name}_segment{i}"
                    tmp_segment_path = os.path.join(self.tmp_output_dir, f"{segment_name}.wav")

                    try:
                        torchaudio.save(tmp_segment_path, segment, sample_rate)
                        caption = self.perform_model(tmp_segment_path)

                        start_time = i * target_length
                        end_time = (i + 1) * target_length if (i + 1) * target_length <= audio_length_sec else audio_length_sec
                        segment_infor = {"id": i, "start": start_time, "end": end_time, "caption": caption}
                        
                        segment_results.append(segment_infor)
                    except Exception as seg_error:
                        print(f"Error processing segment {i}: {seg_error}")
                        # Continue with next segment instead of failing completely
                        continue
            
            save_dir = os.path.join(self.output_dir, audio_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir ,"audio_captioning.json"), "w") as outfile:
                json.dump(segment_results, outfile)

        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            # Create empty result instead of failing completely
            save_dir = os.path.join(self.output_dir, audio_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, "audio_captioning.json"), "w") as outfile:
                json.dump([], outfile)


    def pad_audio(self, audio_tensor, sample_rate, target_length = 10):
        """
        Pads audio with silence to reach the target length in seconds.
        Handles both single-channel and multi-channel audio.
        """
        current_length = audio_tensor.size(1)
        target_length_samples = target_length * sample_rate  # Target length in samples
        
        if current_length < target_length_samples:
            # Calculate the required padding
            padding_length = target_length_samples - current_length
            
            # Create padding 
            padding = torch.zeros(audio_tensor.size(0), padding_length)  # (channels, samples)
            audio_tensor = torch.cat((audio_tensor, padding), dim=1)
        
        return audio_tensor

    def process_all_files(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("New output directory created.")

        org_file_list = os.listdir(self.input_dir)
        file_list = [f for f in org_file_list if not re.match("\\.", f)]
        file_list = natsorted(file_list)

        for audio_file in file_list:
            self.process_audio_file(audio_file, target_length=10)
        
        # Delete tmp segment folder after done
        cmd = f'rm -rf {self.tmp_output_dir}'
        os.system(cmd)

def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s --",
        description="--project_dir: project directory"
                    "--input_dir: Out directory containing all audio input files;"
                    "--output_dir: Directory to store output of the task"
    )
    parser.add_argument(
        "--project_dir", required=True,
        help='Directory of project'
    )
    parser.add_argument(
        "--input_dir", required=True,
        help='Directory containing input files'
    )
    parser.add_argument(
        "--output_dir", required=True,
        help='Output directory'
    )
    return parser

if __name__ == "__main__":
    print("-------- AUDIO CAPTIONING -----------")
    parser = init_argparse()
    args   = parser.parse_args()
    processor = AudioCaptioning(project_dir=args.project_dir, input_dir = args.input_dir, output_dir=args.output_dir)
    processor.process_all_files()