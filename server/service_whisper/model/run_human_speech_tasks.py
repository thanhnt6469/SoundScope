# ---------- This script import needed modules for human-speech tasks ----------------
from pathlib import Path
import argparse
from speech_to_text_lid.s2t_lid_class import Speech2Text_LID
from emotion_regconition.emo_class import EmotionRecognition
from speaker_diarization.speaker_diarization_class import SpeakerDiarization
from voice_gender_detection.voice_gender_detection_class import *
import time
import os


def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s Running human-speech tasks---",
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
    parser.add_argument(
        "--segment_dir", required=True,
        help='Directory containing temp Whisper segments'
    )
    parser.add_argument(
        "--model_type", required=False, default="tiny",
        help='Whisper version'
    )
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args   = parser.parse_args()
    #===================== Speech2Text + Language Detection=============
    s2t_lid = Speech2Text_LID(project_dir=args.project_dir, input_dir=args.input_dir, output_dir=args.output_dir, \
                            segment_dir=args.segment_dir ,model_type=args.model_type)
    s2t_lid.perform_s2t_on_all_files()
    # ===================== Speaker Diarization===============
    spk_dia = SpeakerDiarization(project_dir=args.project_dir, output_dir=args.output_dir,segment_dir=args.segment_dir)
    spk_dia.perform_on_all_audio_files()
    # ===================Speaker Emotion Recogition ==================
    emotion_regconition = EmotionRecognition(project_dir=args.project_dir, output_dir=args.output_dir, segment_dir=args.segment_dir)
    emotion_regconition.emo_reg_all_audio_files()
    # =================== Speaker Voice Gender Detection
    voice_gender = VoiceGenderDetection(project_dir=args.project_dir, output_dir=args.output_dir,segment_dir=args.segment_dir)
    voice_gender.perform_on_all_audio_files()   

    # Detele folder containing temp whisper segments
    cmd = f'rm -rf {args.segment_dir}'
    os.system(cmd)