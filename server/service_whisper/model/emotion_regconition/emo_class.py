import os
import json
from pathlib import Path
import torch
from speechbrain.inference.interfaces import foreign_class
from dotenv import load_dotenv
import yaml

class EmotionRecognition:
    def __init__(self, project_dir, output_dir, segment_dir):
        """
        Initialize the EmotionRecognition class.
        :param project_dir: Directory of current project
        :param segment_dir: Directory containing audio segments for input files (processed through Whisper S2T)
        :param output_dir: Directory where output files will be saved.
        :param label_mapping: Dictionary mapping model labels to descriptive labels.
        """
        self.project_dir = project_dir
        self.segment_dir = segment_dir
        self.output_dir = output_dir
        self.label_mapping = {'neu': "neutral", 'ang': "angry", 'hap': "happy", 'sad': "sad"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model and tokenizer
        with open(os.path.join(project_dir, "model" ,"configs", "model_config.yaml"), "r") as f:
            config = yaml.safe_load(f)
        # self.classifier = foreign_class(
        #     source=os.getenv("EMOTION_REG_MODEL"), 
        #     pymodule_file="custom_interface.py", 
        #     classname=os.getenv("EMOTION_REG_CLASSNAME")
        # )
        self.classifier = foreign_class(
            source=config['EMOTION_REG_MODEL'], 
            pymodule_file="custom_interface.py", 
            classname=config['EMOTION_REG_CLASSNAME']
        )

        if not os.path.exists(self.segment_dir):
            raise FileNotFoundError("Can not find the directory containing extracted audio segments.")
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError('Can not find the directory containing output previously defined.')
        
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
    
    def emo_reg_audio_file(self, folder_name):
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
            for segment_id, segment in enumerate(sorted_segment_list):
                segment_path = os.path.join(folder_path, segment)
                split_list = segment[len(folder_name) + 1:].split("_")
                start_time = float(split_list[0])
                end_time = float(split_list[1])
                try:
                    out_prob, score, index, text_lab = self.classifier.classify_file(segment_path)
                    segment_result = {
                        "id": segment_id,
                        "start": start_time,
                        "end": end_time,
                        "emotion": self.label_mapping[text_lab[0]]
                    }
                    audio_result.append(segment_result)
                except Exception as e:
                    print(f"Error processing segment {segment}: {e}")

            output_folder = os.path.join(self.output_dir, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, 'emo_reg.json')

            with open(output_file, "w") as outfile:
                json.dump(audio_result, outfile)


    def emo_reg_all_audio_files(self):
        """
        Process all folders in the segment directory.
        """
        try:
            folder_list = os.listdir(self.segment_dir)
            for folder_name in sorted(folder_list):
                print(f"Processing folder: {folder_name}")
                self.emo_reg_audio_file(folder_name)
        except Exception as e:
            print(f"Error processing file {folder_name}.wav: {e}")


# # Example usage
# if __name__ == "__main__":
#     project_dir = str(Path(__file__).resolve().parent.parent.parent)
#     segment_dir = os.path.join(project_dir, 's2t_segments')
#     output_dir = os.path.join(project_dir, 'final_output')

#     emotion_recognition = EmotionRecognition(segment_dir=segment_dir, output_dir=output_dir)
#     emotion_recognition.run()
