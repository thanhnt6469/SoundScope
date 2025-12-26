import os
import re
import sys
import json
import random
import pandas as pd
from pathlib import Path
from natsort import natsorted
import warnings
warnings.filterwarnings("ignore")
import random
import argparse
sys.path.append('./01_source/asc_module')
sys.path.append('./01_source/asc_module/02_models')
from asc_class import *
sys.path.append('./01_source/aed_module')
sys.path.append('./01_source/aed_module/utils')
sys.path.append('./01_source/aed_module/pytorch')
sys.path.append('./01_source/aed_module/metadata')
from aed_class import *
from asc_class import ASC_component
from aed_class import AED_component


class ASCAEDPipeline:
    def __init__(self, project_dir, input_dir, output_dir):
        self.project_dir = project_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("Can not find the directory containing input audio files.")
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError('Can not find the directory containing output previously defined.')
        self.tmp_output_dir = "./11_output"
        self.full_event_file = './full_event_dict.csv'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("New output directory created for ASC_AED.")

        if not os.path.exists(self.tmp_output_dir):
            os.makedirs(self.tmp_output_dir)

    def _get_file_list(self):
        """
        Get list of audio files from input directory
        """
        org_file_list = os.listdir(self.input_dir)
        file_list = [file for file in org_file_list if not re.match(r"\.", file)]
        return natsorted(file_list)

    def _process_audio_file(self, audio_file):
        """
        Get ASC-AED result from one audio file
        """
        audio_file_path = os.path.join(self.input_dir, audio_file)
        audio_name, _ = os.path.splitext(audio_file)
        audio_tmp_output_dir = os.path.join(self.tmp_output_dir, audio_name)

        if not os.path.exists(audio_tmp_output_dir):
            os.makedirs(audio_tmp_output_dir)

        asc_comp_inst = ASC_component(dur_hop=10)
        aed_comp_inst = AED_component(dur_hop=10)
        #-- set option
        opt_prod = 'mel'
        opt_emb  = 'emb_concat'
        opt_info = 'label'
        asc_comp_inst.get_pred_prob(audio_file_path, audio_tmp_output_dir, opt_prod)
        aed_comp_inst.get_aed_info(audio_file_path, audio_tmp_output_dir, opt_info)

        return self._generate_summary(audio_tmp_output_dir, audio_name)

    def _generate_summary(self, audio_tmp_output_dir, audio_name):
        """
        Generate ASC-ASC summary for one audio file
        """
        sel_audio_event = []
        sel_audio_scene = []

        for json_file in os.listdir(audio_tmp_output_dir):
            with open(os.path.join(audio_tmp_output_dir, json_file), "r") as json_rd:
                json_data = json.load(json_rd)

                if re.search("_AED_lable.json", json_file):
                    for sub_segment, values in json_data.items():
                        event_sel = [event for event, score in values[2].items() if score > 0.2]
                        sel_audio_event.append(event_sel)

                elif re.search("_ASC_label_8class.json", json_file):
                    for sub_segment, values in json_data.items():
                        scene_dict = values[2]
                        sel_audio_scene.append([values[0], values[1], max(zip(scene_dict.values(), scene_dict.keys()))[1]])
        
        # Create summary for one audio file given extracted sound scene and sound event
        summary = self._create_summary(sel_audio_event, sel_audio_scene)
        # Dir containing results of
        output_path = os.path.join(self.output_dir, audio_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(os.path.join(output_path, 'asc_aed.json'), "w") as outfile:
            json.dump(summary, outfile)

        return summary

    def _create_summary(self, sel_audio_event, sel_audio_scene):
        """
        Function to create summary given asc-aed labels
        """
        full_event_pd = pd.read_csv(self.full_event_file)
        summary = []

        for i_seg in range(len(sel_audio_scene)):
            # segment_summary = f"From {sel_audio_scene[i_seg][0]}s to {sel_audio_scene[i_seg][1]}s, sound events of: "
            segment_summary = {"id": i_seg,
                               "start":sel_audio_scene[i_seg][0],
                               "end":sel_audio_scene[i_seg][1],
                               "background_scene": None,
                               "sound_events": None}
            bad_event_count = 0

            for ind, event in enumerate(sel_audio_event[i_seg]):
                alarming_note = ""
                event_rating = full_event_pd.loc[full_event_pd["Event"] == event, "Rate"].iloc[0]

                if event_rating.startswith("l2"):
                    alarming_note = "(Alarming sound)"
                    bad_event_count += 1
                elif event_rating.startswith("l3"):
                    alarming_note = "(Violent sound)"
                    bad_event_count += 1

                # separator = "; " if ind == len(sel_audio_event[i_seg]) - 1 else ", "
                # segment_summary += f"{event}{alarming_note}{separator}"
                segment_summary['sound_events'] = f"{event}{alarming_note}"
            if sel_audio_scene[i_seg][2] == "in riot" and bad_event_count == 0:
                # segment_summary += "sound scene of: in door."
                segment_summary['background_scene'] = "in door"
            else:
                # segment_summary += f"sound scene of: {sel_audio_scene[i_seg][2]}."
                segment_summary['background_scene'] = sel_audio_scene[i_seg][2]
            summary.append(segment_summary)

        return summary

    def clean_temp_files(self):
        """
        Clear temporary files after processing
        """
        if os.path.exists(self.tmp_output_dir):
            for root, dirs, files in os.walk(self.tmp_output_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.tmp_output_dir)

    def perform_asc_aed(self):
        """
        Run full ASC-AED pipeline
        """
        file_list = self._get_file_list()

        for audio_file in file_list:
            print(f"Processing file: {audio_file}")
            summary = self._process_audio_file(audio_file)
            print(summary)

        self.clean_temp_files()

def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s --subset 'train' --outdir '11_mel' --delta 'yes'",
        description="--subset: subset from dataset --> 'train' or 'dev' or 'eval'"
                    "--outdir: Out directory to store spectrogram;"
                    "--delta: Apply delta on spectrogram or not, use terms of 'yes/no'"
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
    print("-------- ASC-AED -----------")
    parser = init_argparse()
    args   = parser.parse_args()
    pipeline = ASCAEDPipeline(project_dir=args.project_dir, input_dir=args.input_dir, output_dir= args.output_dir)
    pipeline.perform_asc_aed()
