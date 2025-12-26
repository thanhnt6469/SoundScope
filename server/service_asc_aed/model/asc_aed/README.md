# 01_general_audio_summary_batchmode

This repo is an application for audio summary. The input are audio files (wav or mp3). The output are JSON files in which audio information (sound scene and sound event) in the audio files is described in text.

In this repo, input audio files are stored in the folder '01_input' (some wav files are available in this folder). Output JSON files are stored in the folder '12_output' (some JSON files, which are corresponding wav files, are available in this folder).

When the tool is called to execute, all audio files in the folder '01_input' are processed. For each audio file in the folder '01_input', we obtain JSON file in the folder '12_output' with the same name. Therefore, this tool can automatically run with a large number of input audio files.

A docker image is also available on the docker hub to pull and use.

### NOTE:  The time recording of audio file is not over 100 seconds. The audio file should be 10 second. The code is only for running on CPU

## I/ Local development setup

### Prerequisites:
* python (tested with 3.12.4) with conda 
* pip

### Installation
1. Create a conda environment
```
conda create --name ait-st-aud python==3.12.4
```
2. Activate conda environment
```
conda activate ait-st-aud
```
3. Install requirements
```
pip install --upgrade pip
pip install -r requirement.txt
```
4. Process with available wav files in the folder '01_input'
```
python asc_aed_detect.py 
```

### Processing audio files
1. Activate conda environment
```
conda activate ait-st-aud
```
2. Copy audio files into the folder '01_input'
```
cp <path>/*.wav  01_intput/
```
3. Process and check the JSON ouptut files in the folder '12_output'
```
python asc_aed_detect.py 
```

## II/ Dockerized setup

Check the setting in files: `Dockerfile`

### Build docker image
You can first build the docker image with the name: ait-st-aud
```
sudo docker build -t ait-st-aud .
```

### Execute the tool with the docker image name: ait-st-aud
1. Create the local folders:  '01_input', '12_output'
```
mkdir 01_input
mkdir 12_output 
```
2. Copy audio files to the local folder '01_input'
```
cp <path>/*.wav  01_intput/
```
3. Execute the container
```
sudo docker run -v "$(pwd)/01_input:/app/01_input" -v "$(pwd)/12_output:/app/12_output" ait-st-aud
```
5. Check the output JSON files in the local folder '12_output'


### Execute the tool with the available docker image on the docker hub: lamphamait/ait-st-aud
1. login your docker account
```
sudo docker login -u "your user name" -p "your password" docker.io 
```
2. Pull the available docker image with the name: lamphamait/ait-st-aud
```
sudo docker pull lamphamait/ait-st-aud
```
3. Execute all steps of the upper section (Execute the tool with the docker image name: lamphamait/ait-st-aud)
