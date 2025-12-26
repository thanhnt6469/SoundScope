import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
import torch
import torchaudio
from pyannote.audio import Pipeline
import whisper
import numpy as np
from natsort import natsorted


class Speech2Text_LID:
    def __init__(self, project_dir, input_dir, output_dir, segment_dir, model_type = 'tiny'):
        # Initialize paths and load environment variables
        self.project_dir = project_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.segment_dir = segment_dir
        load_dotenv()
        self.pyannote_key = os.getenv("PYANNOTE_KEY")
        if not self.pyannote_key:
            raise ValueError("PYANNOTE_KEY is not set in the environment or .env file.")
        
        # Set HuggingFace token as environment variable (for huggingface_hub)
        os.environ["HF_TOKEN"] = self.pyannote_key
        os.environ["HUGGING_FACE_HUB_TOKEN"] = self.pyannote_key
        
        # Create new directory (if not available)
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.segment_dir):
            os.makedirs(self.segment_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Pyannote pipeline
        try:
            print(f"Loading Pyannote pipeline with key: {self.pyannote_key[:10]}..." if self.pyannote_key else "PYANNOTE_KEY is None!")
            # Try multiple methods to authenticate
            try:
                # Method 1: New API with token parameter
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.pyannote_key
                )
            except (TypeError, Exception) as e1:
                try:
                    # Method 2: Old API with use_auth_token
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=self.pyannote_key
                    )
                except Exception as e2:
                    # Method 3: Use environment variable (HF_TOKEN should be set above)
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )
            if self.pipeline is None:
                raise ValueError("Failed to load Pyannote pipeline. Pipeline is None.")
            self.pipeline = self.pipeline.to(self.device)
            print("Pyannote pipeline loaded successfully!")
        except Exception as e:
            print(f"ERROR loading Pyannote pipeline: {e}")
            print("Make sure:")
            print("1. PYANNOTE_KEY is set in .env file")
            print("2. You have accepted user conditions at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("3. Your token has access to the pipeline")
            raise ValueError(f"Failed to load Pyannote pipeline: {e}")

        self.language_map = {
            "en": "English", "fr": "French", "es": "Spanish", "de": "German",
            "zh": "Chinese (Simplified)", "ja": "Japanese", "ru": "Russian",
            "pt": "Portuguese", "it": "Italian", "ko": "Korean", "hi": "Hindi"
        }

        self.model_type = model_type
        self.run_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.whisper_model = whisper.load_model(self.model_type, self.run_device)
        if self.whisper_model is not None:
            print("Loaded model successuflly")

    def count_speakers(self, audio_file, spk_max=5, spk_min=0):
        """
        Count number of speakers in a .wav file
        """
        diarization = self.pipeline(audio_file, min_speakers=spk_min, max_speakers=spk_max)
        speaker_count = set()
        for _, _, speaker in diarization.itertracks(yield_label=True):
            speaker_count.add(speaker)
        
        print("n_speakers:", len(speaker_count))
        return len(speaker_count)

    def perform_s2t_on_audio_file(self, audio_file):
        """
        Perform S2T + LID for one single .wav file
        """
        audio_file_path = os.path.join(self.input_dir, audio_file)
        audio_file_name = audio_file[:-4]
        print(f"Processing file: {audio_file}")
        # First load, audio file -- Get metadata
        waveform, sr = torchaudio.load(audio_file_path)
        duration = round(waveform.size(1) / sr, 2)

        # --------Count speakers
        n_speakers = self.count_speakers(audio_file_path)
        output_dir = os.path.join(self.output_dir, audio_file_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save metadata for all cases
        metadata = {"filename": audio_file_name,
                    "duration": duration,
                    "sample_rate": 16000,
                    }
        with open(os.path.join(output_dir, 'metadata.json'), "w") as outfile:
            json.dump([metadata], outfile)
        
        if n_speakers == 0: # No speaker detected
            print(f"File name: {audio_file_name}: No human speech detected.")
            with open(os.path.join(output_dir, 's2t.json'), "w") as outfile:
                json.dump(["No human speech"], outfile)
            with open(os.path.join(output_dir, 'lid.json'), "w") as outfile:
                json.dump(["None"], outfile)
            with open(os.path.join(output_dir, 'speaker_count.json'), "w") as outfile:
                json.dump([n_speakers], outfile)
            return 
        else:
            # Detect speakers
            segment_dir_with_audio_name = os.path.join(self.segment_dir, audio_file_name)
            os.makedirs(segment_dir_with_audio_name, exist_ok=True)
            print(f"Created folder to save segments for {audio_file_name} at {segment_dir_with_audio_name}.")

            try:
                # waveform, sr = torchaudio.load(audio_file_path)
                wp_results = self.whisper_model.transcribe(audio_file_path)
                for ide in range(len(wp_results['segments'])):
                    for key in ['seek', 'tokens', 'compression_ratio', 'temperature', 'avg_logprob', 'no_speech_prob']:
                        wp_results['segments'][ide].pop(key, None)

                for segment in wp_results["segments"]:
                    start_time = np.round(segment["start"], 2)
                    end_time = np.round(segment["end"], 2)
                    if end_time > waveform.size(1) / sr:  # Avoid end_time > audio_length (second)
                        end_time = np.round(waveform.size(1) / sr, 2)
                    # Rouding for displaying
                    segment['start'] = start_time
                    segment['end'] = end_time

                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    audio_segment_name = f"{audio_file_name}_{start_time}_{end_time}_.wav"
                    audio_segment = waveform[:, start_sample:end_sample]

                    torchaudio.save(os.path.join(segment_dir_with_audio_name, audio_segment_name), audio_segment, sr)

            except Exception as e:
                print(f"Error processing file {audio_file}: {e}")
                return

            print("S2T results", wp_results['segments'])
            print("Language detected:", self.language_map.get(wp_results.get('language', 'unknown'), "Unknown"))

        # Save results
        with open(os.path.join(output_dir, 's2t.json'), "w") as outfile:
            json.dump(wp_results['segments'], outfile)

        detected_language = self.language_map.get(wp_results.get('language', 'unknown'), "Unknown")
        with open(os.path.join(output_dir, 'lid.json'), "w") as outfile:
            json.dump([detected_language], outfile)
        with open(os.path.join(output_dir, 'speaker_count.json'), "w") as outfile:
            json.dump([n_speakers], outfile)

    def perform_s2t_on_all_files(self):
        org_file_list = os.listdir(self.input_dir)
        # print("org_file_list", org_file_list)
        file_list = [f for f in org_file_list if not f.startswith('.')]
        file_list = natsorted(file_list)

        for audio_file in file_list:
            self.perform_s2t_on_audio_file(audio_file)

        


# # Usage example
# if __name__ == "__main__":
#     project_dir = str(Path(__file__).resolve().parent.parent.parent)
#     audio_processor = Speech2Text_LID(project_dir, model_type="tiny", run_device="cpu")
#     audio_processor.perform_s2t_on_all_files()