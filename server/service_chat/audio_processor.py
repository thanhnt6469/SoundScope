"""
Audio Processing Module
Tích hợp xử lý audio từ các services (ASC-AED, Whisper, CAP-DF)
"""
import httpx
import asyncio
import logging
from typing import Dict, Any, Tuple
from io import BytesIO

logger = logging.getLogger(__name__)


async def fetch_audio_response(client: httpx.AsyncClient, url: str, file_name: str, file_contents: bytes) -> Dict[str, Any]:
    """Gửi request tới audio processing service."""
    try:
        files = {"file": (file_name, file_contents, "audio/wav")}
        response = await client.post(url, files=files, timeout=300.0)  # 5 minutes timeout
        return response.json()
    except Exception as e:
        logger.error(f"Error calling {url}: {e}")
        return {}


async def process_audio_file(file_name: str, file_contents: bytes, 
                            asc_aed_url: str, whisper_url: str, cap_df_url: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Xử lý audio file bằng cách gọi 3 services song song.
    
    Returns:
        Tuple of (asc_aed_response, whisper_response, cap_df_response)
    """
    timeout = httpx.Timeout(300.0)  # 5 minutes timeout
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            asyncio.create_task(fetch_audio_response(client, asc_aed_url, file_name, file_contents)),
            asyncio.create_task(fetch_audio_response(client, whisper_url, file_name, file_contents)),
            asyncio.create_task(fetch_audio_response(client, cap_df_url, file_name, file_contents)),
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        asc_aed_response = responses[0] if not isinstance(responses[0], Exception) else {}
        whisper_response = responses[1] if not isinstance(responses[1], Exception) else {}
        cap_df_response = responses[2] if not isinstance(responses[2], Exception) else {}
        
        if isinstance(responses[0], Exception):
            logger.error(f"ASC-AED error: {responses[0]}")
        if isinstance(responses[1], Exception):
            logger.error(f"Whisper error: {responses[1]}")
        if isinstance(responses[2], Exception):
            logger.error(f"CAP-DF error: {responses[2]}")
    
    return asc_aed_response, whisper_response, cap_df_response


def merge_audio_analysis(asc_aed_response: Dict[str, Any], 
                        whisper_response: Dict[str, Any], 
                        cap_df_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge kết quả từ 3 services thành format chuẩn.
    Giống logic trong client/utils.py
    """
    total_info = {
        "metadata": None, 
        "acoustics_information": None, 
        "human_speech_information": None, 
        "other_information": None
    }
    
    # Get metadata
    try:
        metadata = whisper_response.get("whisper_based", [{}])[0].get("metadata", [{}])
        total_info["metadata"] = metadata
    except:
        total_info["metadata"] = None
    
    # Extract acoustics information
    try:
        asc_aed_info = asc_aed_response.get("asc_aed", [{}])
        captioning_information = cap_df_response.get("cap_df", [{}])[0].get("audio_captioning", [{}])
        acoustics_info = captioning_information
        
        for idx, item in enumerate(acoustics_info):
            item.pop('id', None)
            if idx == len(acoustics_info) - 1:
                if idx > 0 and len(asc_aed_info) > idx - 1:
                    acoustics_info[idx]['background_scene'] = asc_aed_info[idx-1].get('background_scene', '')
                    acoustics_info[idx]['sound_events'] = asc_aed_info[idx-1].get('sound_events', [])
            else:
                if len(asc_aed_info) > idx:
                    acoustics_info[idx]['background_scene'] = asc_aed_info[idx].get('background_scene', '')
                    acoustics_info[idx]['sound_events'] = asc_aed_info[idx].get('sound_events', [])
        
        total_info["acoustics_information"] = acoustics_info
    except Exception as e:
        logger.error(f"Error processing acoustics: {e}")
        total_info["acoustics_information"] = []
    
    # Extract human speech information
    try:
        whisper_data = whisper_response.get("whisper_based", [])
        
        if whisper_data and len(whisper_data) > 0:
            human_speech_info = whisper_data[0].copy()
            human_speech_info.pop("metadata", None)
            
            human_speech_text = human_speech_info.get("human speech information", [])
            
            human_speech_lang = human_speech_info.get("language detection", None)
            if isinstance(human_speech_lang, list) and len(human_speech_lang) > 0:
                human_speech_lang = human_speech_lang[0] if isinstance(human_speech_lang[0], (str, dict)) else human_speech_lang
            elif isinstance(human_speech_lang, list) and len(human_speech_lang) == 0:
                human_speech_lang = None
            
            human_speech_count = human_speech_info.get("number of speakers", None)
            if isinstance(human_speech_count, list) and len(human_speech_count) > 0:
                human_speech_count = human_speech_count[0]
            elif isinstance(human_speech_count, list) and len(human_speech_count) == 0:
                human_speech_count = None
            
            human_speech_info_merged = {
                "human speech information": human_speech_text if human_speech_text else [],
                "language detection": human_speech_lang,
                "number of speakers": human_speech_count
            }
        else:
            human_speech_info_merged = {
                "human speech information": [],
                "language detection": None,
                "number of speakers": None
            }
        
        total_info["human_speech_information"] = human_speech_info_merged
    except Exception as e:
        logger.error(f"Error processing human speech: {e}")
        total_info["human_speech_information"] = {
            "human speech information": [],
            "language detection": None,
            "number of speakers": None
        }
    
    # Extract other information
    try:
        cap_df_data = cap_df_response.get("cap_df", [{}])
        if cap_df_data and len(cap_df_data) > 0:
            deepfake_info = cap_df_data[0].get("deepfake_detection", None)
            if deepfake_info is None or (isinstance(deepfake_info, list) and len(deepfake_info) == 0):
                deepfake_info = None
            elif isinstance(deepfake_info, list) and len(deepfake_info) > 0:
                deepfake_info = deepfake_info[0] if isinstance(deepfake_info[0], str) else deepfake_info
        else:
            deepfake_info = None
        
        total_info["other_information"] = {
            "deepfake detection": deepfake_info
        }
    except Exception as e:
        logger.error(f"Error processing other info: {e}")
        total_info["other_information"] = {
            "deepfake detection": None
        }
    
    return total_info

