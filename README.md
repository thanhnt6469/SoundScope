# SoundScope 🎧

An audio analysis app that can detect sounds, transcribe speech, identify speakers, and even detect deepfakes.

## What it does

- **Sound Detection**: Identifies background scenes (street, park, mall...) and sound events (car horn, dog bark, gunshot...)
- **Speech-to-Text**: Transcribes what people say in the audio
- **Speaker Diarization**: Figures out who said what
- **Emotion & Gender Detection**: Detects speaker's emotion and gender
- **Language Detection**: Identifies the language being spoken
- **Deepfake Detection**: Checks if the audio is real or AI-generated
- **Audio Captioning**: Generates a description of what's happening in the audio
- **Chat with AI**: Ask questions about the audio and get answers

## Project Structure

```
├── server/                 # Backend services
│   ├── service_asc_aed/   # Sound scene & event detection
│   ├── service_whisper/   # Speech processing (STT, emotion, gender...)
│   ├── service_cap_df/    # Audio captioning & deepfake detection
│   └── service_chat/      # Chat API with Groq LLM
├── client/                # Streamlit web app
├── flutter_app/           # Mobile app (Android/iOS)
└── docker-compose.yml     # Run everything with Docker
```

## Quick Start

### 1. Setup environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 2. Run with Docker

```bash
docker-compose up --build
```

### 3. Or run manually

**Server (Chat API):**
```bash
cd server/service_chat
pip install -r requirements_chat.txt
python chat_api.py
```

**Flutter App:**
```bash
cd flutter_app
flutter pub get
flutter run
```

## Tech Stack

- **Backend**: FastAPI, PyTorch, Whisper, Groq LLM
- **Frontend**: Flutter (mobile), Streamlit (web)
- **AI Models**: Custom models for ASC/AED, Whisper for STT, SpeechBrain for emotion

## Notes

- Get Groq API key (free): https://console.groq.com/
- Get Pyannote access token (for speaker diarization): https://huggingface.co/pyannote/speaker-diarization-3.1
- Audio services need GPU for best performance (CPU works too, just slower)
