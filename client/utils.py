import time
import json

def generate_system_prompt(audio_data):
    metadata = audio_data['information']
    labels = audio_data['labels']

    # prompt = "You are a chatbot designed to perform information retrieval tasks based on provided information related to an audio recording." + \
    #          "\nGiven the following extracted information about an audio recording: \n"

    prompt = ""

    # Add metadata
    prompt += f"'duration': {metadata['duration']} seconds\n"
    prompt += f"'sample_rate': {metadata['sample_rate']} Khz\n"
    prompt += f"'channels': {metadata['channels']}\n"

    # Add labels
    for label, data in labels.items():
        prompt += f"'{label}':\n"
        if isinstance(data, dict):
            for time_range, value in data.items():
                prompt += f"- [{time_range}]: {repr(value)}\n"
        elif isinstance(data, list):
            prompt += f"- {repr(data)}\n"
        else:
            prompt += f"- {repr(data)}\n"
        prompt += "\n"

    return prompt

def allowed_file(filename):
    """
    Check wav input format
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav']

# Streamed response emulator
def response_generator(str):
    for word in str.split():
        yield word + " "
        time.sleep(0.1)


def merge_json_files(asc_aed_response, whisper_response, cap_df_response):
    """
    Summary information from 3 json files of 3 services
    asc_aed_response: asc_aed
    whisper_response: whisper_based
    cap_df_response: cap_df
    """
    total_info = {"metadata": None, "acoustics_information": None, "human_speech_information": None, "other_information": None}
    
    # Get total metadata
    metadata = whisper_response.get("whisper_based", [{}])[0].get("metadata", [{}])
    total_info["metadata"] = metadata

    # Extract acoustics information from file1.json
    asc_aed_info = asc_aed_response.get("asc_aed", [{}])
    captionin_information = cap_df_response.get("cap_df", [{}])[0].get("audio_captioning", [{}])
    acoustics_info = captionin_information

    for idx, item in enumerate(acoustics_info):
        # Remove 'id' field
        item.pop('id', 0)
        if idx == len(acoustics_info) - 1:  
            # asc_aed's result skip the last segment, so replace the last <10-second by the last 10-second
            acoustics_info[idx]['background_scene'] = asc_aed_info[idx-1]['background_scene']
            acoustics_info[idx]['sound_events'] = asc_aed_info[idx-1]['sound_events']
        else:
            # Add asc_aed entries to acoustics infor
            acoustics_info[idx]['background_scene'] = asc_aed_info[idx]['background_scene']
            acoustics_info[idx]['sound_events'] = asc_aed_info[idx]['background_scene']

    total_info["acoustics_information"] = acoustics_info


    # --------------Extract human speech information---------------------
    try:
        whisper_data = whisper_response.get("whisper_based", [])
        print(f"DEBUG: whisper_data type: {type(whisper_data)}, length: {len(whisper_data) if whisper_data else 0}")
        
        if whisper_data and len(whisper_data) > 0:
            human_speech_info = whisper_data[0].copy()
            print(f"DEBUG: human_speech_info keys: {human_speech_info.keys()}")
            
            human_speech_info.pop("metadata", None)  # Remove metadata from human speech info

            # Get S2T, Diarization
            human_speech_text = human_speech_info.get("human speech information", [])
            print(f"DEBUG: human_speech_text type: {type(human_speech_text)}, length: {len(human_speech_text) if isinstance(human_speech_text, list) else 'N/A'}")
            
            # Get LID - handle both list and dict formats
            human_speech_lang = human_speech_info.get("language detection", None)
            if isinstance(human_speech_lang, list) and len(human_speech_lang) > 0:
                # If it's a list, take first element
                human_speech_lang = human_speech_lang[0] if isinstance(human_speech_lang[0], (str, dict)) else human_speech_lang
            elif isinstance(human_speech_lang, list) and len(human_speech_lang) == 0:
                human_speech_lang = None
            print(f"DEBUG: language detection: {human_speech_lang}")
            
            # Get speaker count - handle both list and int formats
            human_speech_count = human_speech_info.get("number of speakers", None)
            if isinstance(human_speech_count, list) and len(human_speech_count) > 0:
                human_speech_count = human_speech_count[0]
            elif isinstance(human_speech_count, list) and len(human_speech_count) == 0:
                human_speech_count = None
            print(f"DEBUG: number of speakers: {human_speech_count}")
            
            # Get human speech information from file2.json
            human_speech_info_merged = {
                "human speech information": human_speech_text if human_speech_text else [],
                "language detection": human_speech_lang,
                "number of speakers": human_speech_count
            }
        else:
            print("DEBUG: whisper_data is empty or None")
            human_speech_info_merged = {
                "human speech information": [],
                "language detection": None,
                "number of speakers": None
            }
    except Exception as e:
        import traceback
        print(f"ERROR processing whisper data: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        human_speech_info_merged = {
            "human speech information": [],
            "language detection": None,
            "number of speakers": None
        }

    total_info["human_speech_information"] = human_speech_info_merged


    # ------------Extract other information from file3.json
    cap_df_data = cap_df_response.get("cap_df", [{}])
    if cap_df_data and len(cap_df_data) > 0:
        deepfake_info = cap_df_data[0].get("deepfake_detection", None)
        # deepfake_info can be ["fake"], ["real"], or None
        # If it's None or empty list, set to None
        if deepfake_info is None or (isinstance(deepfake_info, list) and len(deepfake_info) == 0):
            deepfake_info = None
        elif isinstance(deepfake_info, list) and len(deepfake_info) > 0:
            # Extract the first element (should be "fake" or "real")
            deepfake_info = deepfake_info[0] if isinstance(deepfake_info[0], str) else deepfake_info
    else:
        deepfake_info = None

    total_info["other_information"] = {
        "deepfake detection": deepfake_info
    }
    
    return total_info

    