#!/bin/bash

# Enable error handling
set -e

echo "Downloading cached models..."

# Load PYANNOTE_KEY from .env file
export PYANNOTE_KEY=$(grep -E '^PYANNOTE_KEY=' .env | cut -d '=' -f2- | tr -d '"')

# Run Python script to predownload models
python <<EOF
from pyannote.audio import Pipeline
from model import ECAPA_gender

# Download pyannote speaker diarization model
Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='${PYANNOTE_KEY}')

# Download gender classifier model
ECAPA_gender.from_pretrained('JaesungHuh/voice-gender-classifier')

# If you want to include the SpeechBrain model later, uncomment below:
# from speechbrain.inference.interfaces import foreign_class
# classifier = foreign_class(
#     source='speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
#     pymodule_file='custom_interface.py',
#     classname='CustomEncoderWav2vec2Classifier'
# )
EOF

echo "✅ All Huggingface models cached successfully!"


