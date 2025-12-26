import json
import os



def create_human_speech_information(output_dir, save_dir):
    """
    Create JSON file based on human speech information (S2T, Speaker diarization, Voice gender detection, Emotion recognition, Language detection)
    :param output_dir: Directory containing all JSON files of all tasks
    :param save_dir: Directory to save final JSON
    Examples:
    {
    "metadata": {
        "audio_file": "audio_filename_placeholder",
        "duration": "10s",
        "sample_rate": 16000
    }
    "human_speech": [
        {
        "start": 0.0,
        "end": 3.48,
        "speaker": "SPEAKER 1",
        "text": "It is your chair, I used to sit in this chair."
        "emotion": "Neutral"
        "gender": "male"
        },
        {
        "start": 3.48,
        "end": 10.0,
        "speaker": "SPEAKER 1",
        "text": "It is your chair, I used to sit in this chair."
        "emotion": "Neutral"
        "gender": "male"
        }
    ]
    }
    """
   
        # Map file names to the respective task keys
    file_to_task_mapping = {
        "metadata.json": "metadata",
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

        human_tasks_infor = {"metadata": None, "human speech information": None, "language detection": None, "number of speakers": None}
        # Assign metadata
        try:
            human_tasks_infor['metadata'] = task_data['metadata']
        except Exception as e:
            print("Error processing metadata: ", e)
            human_tasks_infor['metadata'] = None
        # Assign speaker count
        try:
            human_tasks_infor['number of speakers'] = task_data['speaker count']
        except Exception as e:
            print("Error processing speaker count: ", e)
        
        # Assign language detection 
        try:
            human_tasks_infor['language detection'] = task_data['language detection']
        except Exception as e:
            print("Error processing language detection: ", e)
            human_tasks_infor['language detection'] = None

        # Assign other human speech tasks    (Type 2: Human-speech-based timeline)
        try:
            human_speech_infor = task_data['text transcription'] # Define timeline of S2T as baseline
            for idx, item in enumerate(task_data['text transcription']):
                
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

            human_tasks_infor['human speech information'] = human_speech_infor

        except Exception as e:
            print("Error processing human speech information: ", e)
        
        save_path = os.path.join(save_dir, folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, "human_speech_info.json"), "w") as output:
            json.dump([human_tasks_infor], output, indent=4)
        print(f"Save final information of file {folder} in {save_path}")
    


def load_packages_to_remove():
    # List of packages from the --index-url to remove
    packages_to_remove = {
        "certifi", "charset-normalizer", "cmake", "colorama", "fbgemm-gpu", "filelock", "fsspec", 
        "idna", "intel-openmp", "iopath", "jinja2", "lightning-utilities", "lit", "markupsafe", 
        "mkl", "mpmath", "mypy-extensions", "networkx", "numpy", "nvidia-cublas-cu11", 
        "nvidia-cuda-cupti-cu11", "nvidia-cuda-nvrtc-cu11", "nvidia-cuda-runtime-cu11", 
        "nvidia-cudnn-cu11", "nvidia-cufft-cu11", "nvidia-curand-cu11", "nvidia-cusolver-cu11", 
        "nvidia-cusparse-cu11", "nvidia-nccl-cu11", "nvidia-nccl-cu12", "nvidia-nvtx-cu11", 
        "packaging", "pillow", "portalocker", "pyre-extensions", "pytorch-triton", 
        "pytorch-triton-rocm", "pytorch-triton-xpu", "requests", "setuptools", "sympy", "tbb", 
        "torch", "torch-cuda80", "torch-model-archiver", "torch-tb-profiler", "torch-tensorrt", 
        "torchao", "torchaudio", "torchcodec", "torchcsprng", "torchdata", "torchmetrics", 
        "torchrec", "torchrec-cpu", "torchserve", "torchtext", "torchtune", "torchvision", 
        "tqdm", "triton", "typing-extensions", "typing-inspect", "urllib3", "xformers"
    }
    return packages_to_remove

def clean_requirements(input_file, output_file):
    packages_to_remove = load_packages_to_remove()
    
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            # Strip whitespace and comments
            line = line.strip()
            if not line or line.startswith("#"):
                f_out.write(line + "\n")
                continue
            
            # Extract package name (before any version specifier like ==, >=, etc.)
            package_name = line.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0].strip()
            
            # Write line only if package is not in the removal list
            if package_name.lower() not in packages_to_remove:
                f_out.write(line + "\n")