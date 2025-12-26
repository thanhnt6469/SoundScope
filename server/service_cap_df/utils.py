import json
import os



def create_captioning_df_information(output_dir, save_dir):
    """
    Create JSON file based on captioning-deepfake service (**Captioning and Deepfake Detection**)
    :param output_dir: Directory containing all JSON files of all tasks
    :param save_dir: Directory to save final JSON
    Examples:
    {
    "audio_captioning": [
        {
        "start": 0.0,
        "end": 3.48,
        "caption: "A person is speaking"
        "gender": "male"
        },
        {
        "start": 3.48,
        "end": 10.0,
        "caption: "A person is speaking"
        "gender": "male"
        }
        ]
    "deepfake_detection": ['Real']
    }
    """
   
    # Map file names to the respective task keys
    file_to_task_mapping = {
        "audio_captioning.json": "audio captioning",
        "deepfake.json": "deepfake detection"
    }
  
    # First, collect all task data from all folders
    # Group folders by base name (e.g., "3" and "3. Actual Test 1B" both have base "3")
    folder_groups = {}
    all_folders = os.listdir(output_dir)
    
    for folder in all_folders:
        folder_dir = os.path.join(output_dir, folder)
        if not os.path.isdir(folder_dir):
            continue
        
        # Get base name (part before first dot, or full name if no dot)
        base_name = folder.split('.')[0] if '.' in folder else folder
        
        if base_name not in folder_groups:
            folder_groups[base_name] = []
        folder_groups[base_name].append(folder)
    
    # Process each group of folders
    for base_name, folders in folder_groups.items():
        print(f"Processing base name: {base_name}, folders: {folders}")
        task_data = {}
        
        # Collect task data from all folders in this group
        for folder in folders:
            folder_dir = os.path.join(output_dir, folder)
            # Process each JSON task file in this folder
            if os.path.isdir(folder_dir):
                for task_file in os.listdir(folder_dir):
                    task_file_path = os.path.join(folder_dir, task_file)
                    if task_file in file_to_task_mapping and task_file.endswith(".json"):
                        # Get task key
                        task_key = file_to_task_mapping[task_file]
                        try:
                            with open(task_file_path, "r") as file:
                                task_data[task_key] = json.load(file)
                                print(f"Found {task_key} in folder {folder}")
                        except Exception as e:
                            print(f"Error reading {task_file_path}: {e}")

        cap_df_infor = {"audio_captioning": None , "deepfake_detection": None}
        try:
            cap_df_infor['audio_captioning'] = task_data['audio captioning']
        except:
            cap_df_infor['audio_captioning'] = None

        try:
            cap_df_infor['deepfake_detection'] = task_data['deepfake detection']
        except:
            cap_df_infor['deepfake_detection'] = None

        # Save to all folders with this base name (or use the longest folder name)
        # Use the longest folder name as the primary save location
        primary_folder = max(folders, key=len) if folders else base_name
        save_path = os.path.join(save_dir, primary_folder)
        print("SAVE path: ", save_path)
        if not os.path.exists(save_path):
            print("Create folder: ", save_path)
            os.makedirs(save_path)
        with open(os.path.join(save_path, "captioning_deepfake.json"), "w") as output:
            json.dump([cap_df_infor], output, indent=4)
        print(f"Save final information of file {primary_folder} in {save_path}")
        
        # Also save to other folders in the group for consistency
        for folder in folders:
            if folder != primary_folder:
                alt_save_path = os.path.join(save_dir, folder)
                if not os.path.exists(alt_save_path):
                    os.makedirs(alt_save_path)
                with open(os.path.join(alt_save_path, "captioning_deepfake.json"), "w") as output:
                    json.dump([cap_df_infor], output, indent=4)
        
