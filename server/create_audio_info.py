import json
import os
##---------------------------- THIS SCRIPT CREATE FINAL JSON FILE AFTER PERFORMING ALL TASK-----------------


def create_task_centric_info(output_dir, save_dir):
    """
    Create JSON file based on tasks
    :param output_dir: Directory containing all JSON files of all tasks
    :param save_dir: Directory to save final JSON
    Example:
                {
            "audio_file": "audio_filename",
            "duration": "10s",
            "asc_aed": {
                "timeline": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "sound_events": ["Speech", "Narration", "Monologue", "Male speech", "Man speaking"],
                    "sound_scene": "Indoor"
                }
                ]
            },
            "audio_captioning": {
                "timeline": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "caption": "A person is speaking"
                }
                ]
            },
            "deepfake_detection": {
                "status": "Real"
            },
            "emotion_recognition": {
                "timeline": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.48,
                    "emotion": "Neutral"
                }
                ]
            },
            "speech_to_text": {
                "timeline": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.48,
                    "text": "It is your chair, I used to sit in this chair."
                }
                ]
            },
            "speaker_diarization": {
                "timeline": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.48,
                    "speaker": "SPEAKER 1"
                }
                ]
            },
            "voice_gender_detection": {
                "timeline": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.48,
                    "gender": "Female"
                }
                ]
            }
            }
    """
    # Initialize the dictionary for the Task-Centric Format
    task_centric_data = {
        "metadata": {},
        "sound event and background scene": None,
        "audio captioning": None,
        "deepfake detection": [],
        "language detection": [],
        "text transcription": None,
        "emotion recognition": None,
        "speaker count": [],
        "speaker diarization": None,
        "voice gender detection": None
    }

    # Map file names to the respective task keys
    file_to_task_mapping = {
        "metadata.json": "metadata",
        "asc_aed.json": "sound event and background scene",
        "audio_captioning.json": "audio captioning",
        "deepfake.json": "deepfake detection",
        "lid.json": "language detection",
        "s2t.json": "text transcription",
        "emo_reg.json": "emotion recognition",
        "speaker_count.json": "speaker count",
        "speaker_diarization.json": "speaker diarization",
        "voice_gender.json": "voice gender detection"
    }
    for folder in os.listdir(output_dir):
        folder_dir = os.path.join(output_dir, folder)
        # Process each JSON task file
        for task_file in os.listdir(folder_dir):
            task_file_path = os.path.join(folder_dir, task_file)
            if task_file in file_to_task_mapping and task_file.endswith(".json"):
                #Get task key
                task_key = file_to_task_mapping[task_file]
                with open(task_file_path, "r") as file:
                    data = json.load(file)
                    # Adjust specialc cases
                    if task_key == "metadata":
                        task_centric_data[task_key] = data[0]            # audio_name, duration, sr
                    elif task_key in ["deepfake detection", "language detection", "speaker count"]:
                        task_centric_data[task_key] = [data[0]]  # Fake or real
                    else:
                        # Remove "id" key from each item
                        for item in data:
                            item.pop('id', None)
                        task_centric_data[task_key] = data
        
        print(task_centric_data)
        # Write the consolidated data to the output file
        save_path = os.path.join(save_dir, folder, "task_centric_infor")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, "final_info.json"), "w") as output:
            json.dump([task_centric_data], output, indent=4)
        print(f"Save final information of file {folder} in {save_path}")
    print(f"Task-Centric Format of saved done!")




def create_timeline_centric_infor(output_dir, save_dir):
    """
    Create JSON file based on timeline 
    :param output_dir: Directory containing all JSON files of all tasks
    :param save_dir: Directory to save final JSON
    Examples:
    {
    "audio_file": "audio_filename_placeholder",
    "duration": "10s",
    "timeline": [
        {
        "start": 0.0,
        "end": 3.48,
        "asc_aed": {
            "sound_events": ["Speech", "Narration", "Monologue", "Male speech", "Man speaking"],
            "sound_scene": "Indoor"
        },
        "audio_captioning": "A person is speaking",
        "deepfake_detection": "Real",
        "emotion_recognition": "Neutral",
        "speech_to_text": "It is your chair, I used to sit in this chair.",
        "speaker_diarization": "SPEAKER 1",
        "voice_gender_detection": "Female"
        },
        {
        "start": 3.48,
        "end": 10.0,
        "asc_aed": {
            "sound_events": ["Speech", "Narration", "Monologue", "Male speech", "Man speaking"],
            "sound_scene": "Indoor"
        },
        "audio_captioning": "A person is speaking",
        "deepfake_detection": "Real"
        }
    ]
    }
    """

    # Map file names to the respective task keys
    file_to_task_mapping = {
        "metadata.json": "metadata",
        "asc_aed.json": "sound event and sound scene",
        "audio_captioning.json": "audio captioning",
        "deepfake.json": "deepfake detection",
        "lid.json": "language detection",
        "s2t.json": "text transcription",
        "emo_reg.json": "emotion recognition",
        "speaker_count.json": "speaker count",
        "speaker_diarization.json": "speaker diarization",
        "voice_gender.json": "voice gender detection"
    }
    for folder in os.listdir(output_dir):
        print("Processing folder: ", folder)
        folder_dir = os.path.join(output_dir, folder)
        if not os.path.isdir(os.path.join(output_dir, folder)):
            continue 
        task_data = {}
        # Process each JSON task file
        for task_file in os.listdir(folder_dir):
            task_file_path = os.path.join(folder_dir, task_file)
            if task_file in file_to_task_mapping and task_file.endswith(".json"):
                # Get task key
                task_key = file_to_task_mapping[task_file]
                with open(task_file_path, "r") as file:
                    task_data[task_key] = json.load(file)


        total_infor = {"metadata": None, "acoustics information": None, "human speech information": None, "other information": None}    
        # Define metadata
        try:
            total_infor['metadata'] = task_data['metadata']
        except Exception as e:
            print("Can not get meatadata: ", e)
            total_infor['metadata'] = None
        """
        Define:
        - metadata: audio name, duration, sampling rate
        - acoustics information: For Type 1 tasks (based on 10-second segments)
        - +) sound scene, sound events, audio captioning
        - human speech information: For Type 2 tasks (based on segment defined by S2T)
        - +) Speech-to-text, speaker diarization, emotion recognition, voice gender detection
        - other information: For Type 3 tasks (non-timeline-based tasks)
        - +) Deepfake detection, language detection, speaker count
        """
        # ---------Define fixed 10-second intervals for asc_aed and audio_captioning tasks (Type 1- Acoustic-based timeline)
        try:
            acoustics_info = task_data['audio captioning'] # Define timeline of audio_captioning as base, then add result of asc-aed later
            for idx, item in enumerate(task_data['audio captioning']):
                # Remove 'id' field
                item.pop('id', 0)
                if idx == len(task_data['audio captioning']) - 1:  
                    # asc_aed's result skip the last segment, so replace the last <10-second by the last 10-second
                    acoustics_info[idx]['background_scene'] = task_data['sound event and sound scene'][idx-1]['background_scene']
                    acoustics_info[idx]['sound_events'] = task_data['sound event and sound scene'][idx-1]['background_scene']
                else:
                    # Add asc_aed entries to acoustics infor
                    acoustics_info[idx]['background_scene'] = task_data['sound event and sound scene'][idx]['background_scene']
                    acoustics_info[idx]['sound_events'] = task_data['sound event and sound scene'][idx]['background_scene']
            total_infor["acoustics information"] = acoustics_info
        except Exception as e:
            print("Can not get acoustics information: ", e)
            total_infor["acoustics information"] = None

        # Define timeline intervals based on the speech-to-text task    (Type 2: Human-speech-based timeline)
        try:
            human_speech_infor = task_data['text transcription'] # Define timeline of S2T as baseline
            for idx, item in enumerate(task_data['text transcription']):
                # remove 'id' field
                item.pop('id', 0)
                # diarization
                try:
                    human_speech_infor[idx]['speaker'] = task_data['speaker diarization'][idx]['speaker']
                except Exception as e:
                    human_speech_infor[idx]['speaker'] = None
                try:
                    # emotion
                    human_speech_infor[idx]['emotion'] = task_data['emotion recognition'][idx]['emotion']
                except Exception as e:
                    human_speech_infor[idx]['emotion'] = None
                try:
                    # voice gender
                    human_speech_infor[idx]['gender'] = task_data['voice gender detection'][idx]['gender']
                except Exception as e:
                    human_speech_infor[idx]['gender'] = None

            total_infor['human speech information'] = human_speech_infor
        except Exception as e:
            print("Can not get human speech information: ", e)
            total_infor['human speech information'] = None
    
        # Define non-timeline-based tasks
        other_infor = {}
        list_of_none_timeline_tasks = ["speaker count", "language detection", "deepfake detection"]
        try:
            for task in list_of_none_timeline_tasks:
                other_infor[task] = task_data[task][0]

            total_infor['other information'] = other_infor
        except Exception as e:
            print("Can not get other information: ", e)
            total_infor['other information'] = None

        # print("Total infor", total_infor)     
        save_path = os.path.join(save_dir, folder, "timeline_centric_info")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, "final_info.json"), "w") as output:
            json.dump([total_infor], output, indent=4)
        print(f"Save final information of file {folder} in {save_path}")
    print(f"Timeline-Centric Format of saved done!")   
    