import os
import numpy as np
import re
import sys 
import librosa
import json
import os
import time
from pathlib import Path
import pandas as pd
from PIL import Image
from natsort import natsorted, ns
import random
sys.path.append('./01_source/asc_module')
sys.path.append('./01_source/asc_module/02_models')
from asc_class import *

sys.path.append('./01_source/aed_module')
sys.path.append('./01_source/aed_module/utils')
sys.path.append('./01_source/aed_module/pytorch')
sys.path.append('./01_source/aed_module/metadata')
from aed_class import *

#============================xxxxxxxxxDEPRECATED SCRIPTxxxxxxxxxxxxxxxx==============================

# Resolve the absolute path of the input folder based on the script's location
project_dir = str(Path(__file__).resolve().parent.parent.parent)
input_dir = project_dir +  "/input2"
final_output_dir = project_dir +  "/final_output"
#-------------------------------------------------------------------------------------------------------
def main():

    # input_dir = './01_input'

    # final_output_dir = '12_output'
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
        print("New output dir for ASC_AED created.")

    org_file_list = os.listdir(input_dir)
    file_list = []  #remove .file
    for i in range(0,len(org_file_list)):
       isHidden=re.match("\.",org_file_list[i])
       if (isHidden is None):
          file_list.append(org_file_list[i])
    natsorted(file_list)
    file_num = len(file_list)

    for audio_file in file_list:
        tmp_output_dir = './11_output'
        if not os.path.exists(tmp_output_dir):
            os.makedirs(tmp_output_dir)

        audio_file_path = os.path.join(input_dir, audio_file)

        audio_name, audio_ext = os.path.splitext(audio_file)
        audio_tmp_output_dir = os.path.join(tmp_output_dir, audio_name)
        if not os.path.exists(audio_tmp_output_dir):
            os.makedirs(audio_tmp_output_dir)

        #-- call instances
        asc_comp_inst = ASC_component(dur_hop=10)
        aed_comp_inst = AED_component(dur_hop=10)

        #-- set option
        opt_prod = 'mel'
        opt_emb  = 'emb_concat'
        opt_info = 'label'

        #-- predict and export
        asc_comp_inst.get_pred_prob(audio_file_path, audio_tmp_output_dir, opt_prod)
        aed_comp_inst.get_aed_info(audio_file_path,  audio_tmp_output_dir, opt_info)


        #---Solve the audio
        sel_audio_event = []
        sel_audio_scene = []
        json_file_list = os.listdir(audio_tmp_output_dir)
        for json_file in json_file_list:
            with open(os.path.join(audio_tmp_output_dir, json_file), "r") as json_rd:
               json_data = json.load(json_rd)

               #----------- Acoustic Event
               if re.search("_AED_lable.json", json_file):
                   for sub_segment, values in json_data.items():
                       event_sel = []
                       event_dict = values[2]
                       for event, score in event_dict.items():
                           if score > 0.2:
                               event_sel.append(event)
                       sel_audio_event.append(event_sel)

               #----------- Acoustic Scene
               elif re.search("_ASC_label_8class.json", json_file):
                   for sub_segment, values in json_data.items():
                       scene_dict = values[2]
                       sel_audio_scene.append([values[0], values[1], max(zip(scene_dict.values(), scene_dict.keys()))[1]])
        
        #print(sel_audio_event)
        #print(sel_audio_scene)

        summary = []
        full_event_dir = './full_event_dict.csv'
        full_event_pd = pd.read_csv(full_event_dir)
        bad_event_count = 0
        for i_seg in range(0, len(sel_audio_scene)):
            sum_aud = 'From '+str(sel_audio_scene[i_seg][0])+'s to '+str(sel_audio_scene[i_seg][1])+'s, sound events of: '
            bad_event_count = 0
            for ind in range(0, len(sel_audio_event[i_seg])):
                alarming_note = ""
               # Check if the event in alarming list
                event_rating = full_event_pd.loc[full_event_pd["Event"] == str(sel_audio_event[i_seg][ind]), "Rate"].iloc[0] 

                if event_rating.startswith("l2"):
                    alarming_note = "(Alarming sound)"
                    bad_event_count += 1
                elif event_rating.startswith("l3"):
                    alarming_note = "(Violent sound)"
                    bad_event_count += 1
            
                if ind == len(sel_audio_event[i_seg]) - 1:
                    sum_aud = sum_aud + sel_audio_event[i_seg][ind] + alarming_note + '; '
                else:
                    sum_aud = sum_aud + sel_audio_event[i_seg][ind] + alarming_note + ", "

            # Add sound scene information - Adjust wrong 'in_riot' label
            if sel_audio_scene[i_seg][2] == "in riot" and bad_event_count == 0:
                sum_aud = sum_aud + "sound scene of: " + random.choice(["in door"]) + "."
            else:
                sum_aud = sum_aud + "sound scene of: " + sel_audio_scene[i_seg][2] + "."

            summary.append(sum_aud)

        with open(os.path.join(final_output_dir, audio_name+'.json'), "w") as outfile:
            json.dump(summary, outfile)

        print(summary)
        cmd = 'rm -rf ./11_output'
        os.system(cmd)
     
   
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()        


