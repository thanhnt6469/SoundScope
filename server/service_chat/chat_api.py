"""
FastAPI Chat Service
Endpoint để xử lý chat với Groq LLM streaming
Giữ nguyên logic từ Streamlit, chỉ expose qua API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import yaml
from groq import Groq
from dotenv import load_dotenv
import asyncio
from pathlib import Path
import shutil
import uuid
from datetime import datetime
import logging
import httpx
from io import BytesIO

# Setup logging với format rõ ràng
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Force reconfiguration
)
logger = logging.getLogger(__name__)

# Đảm bảo log hiển thị ra console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# Import audio processing functions (after logger is defined)
try:
    from audio_processor import process_audio_file, merge_audio_analysis
    AUDIO_PROCESSING_AVAILABLE = True
    logger.info("✅ Audio processing module loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️  audio_processor module not found: {e}. Audio processing will be disabled.")
    AUDIO_PROCESSING_AVAILABLE = False
    # Define dummy functions to avoid errors
    async def process_audio_file(*args, **kwargs):
        return {}, {}, {}
    def merge_audio_analysis(*args, **kwargs):
        return {}

# URLs for audio processing services
# Có thể config qua environment variables
AUDIO_SERVICE_ASC_AED = os.getenv('AUDIO_SERVICE_ASC_AED', 'http://localhost:8000/process_audio/')
AUDIO_SERVICE_WHISPER = os.getenv('AUDIO_SERVICE_WHISPER', 'http://localhost:8001/process_audio/')
AUDIO_SERVICE_CAP_DF = os.getenv('AUDIO_SERVICE_CAP_DF', 'http://localhost:8002/process_audio/')

# Load environment variables
load_dotenv()

app = FastAPI(title="Chat API Service")

# Request logging middleware - Log tất cả requests
@app.middleware("http")
async def log_requests(request, call_next):
    """Log tất cả requests đến server"""
    start_time = datetime.now()
    client_ip = request.client.host if request.client else "unknown"
    
    # Force print để đảm bảo hiển thị
    print(f"\n🌐 {request.method} {request.url.path} - From: {client_ip}")
    logger.info(f"🌐 {request.method} {request.url.path} - From: {client_ip}")
    import sys
    sys.stdout.flush()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    print(f"✅ {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}s")
    logger.info(f"✅ {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}s")
    sys.stdout.flush()
    
    return response

# CORS middleware để Flutter app có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục lưu file upload
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Giới hạn kích thước file (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Initialize Groq client
# Note: Requires groq>=0.11.0 for httpx compatibility
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Load LLM config
# Try multiple paths to find config file
LLM_CONFIG_PATHS = [
    Path(__file__).parent.parent.parent / "client" / "configs" / "llm_config.yaml",
    Path(__file__).parent / "llm_config.yaml",
    Path("client/configs/llm_config.yaml"),
]

llm_config = {
    'LLM_MODEL': "llama-3.1-8b-instant",
    'TEMPERATURE': 0.3,
    'MAX_TOKENS': 512,
    'TOP_P': 0.5,
    'STOP': None,
    'STREAM': True,
}

# Try to load from file
for config_path in LLM_CONFIG_PATHS:
    if config_path.exists():
        try:
            with open(config_path, "r") as file:
                loaded_config = yaml.safe_load(file)
                if loaded_config:
                    llm_config.update(loaded_config)
            break
        except Exception:
            continue

# Load system prompt
# SYSTEM_PROMPT structure from client/prompts/sample_prompt.py
# Format: [{"Intro": "...", "Outro": "..."}]
SYSTEM_PROMPT = [
    {
        "Intro": "You are a specialized assistant designed to provide natural, concise answers about a given audio recording based on the extracted information provided.\n\n"
        "The extracted information includes:\n"
        "- metadata: Basic audio file information (duration, sample_rate, channels)\n"
        "- acoustics_information: Background scenes and sound events detected in the audio\n"
        "- human_speech_information: Contains:\n"
        "  * 'human speech information': Array of speech segments with text transcription, speaker ID, emotion, and gender\n"
        "  * 'language detection': Detected language(s) in the audio\n"
        "  * 'number of speakers': Total number of speakers detected\n"
        "- other_information: Contains:\n"
        "  * 'deepfake detection': Result can be 'fake', 'real', or None (if not available)\n\n",
        "Outro": (
            "\nRules for Interaction:\n"
            "1. First, check if the question relates to the extracted information from the audio recording. If it doesn't, politely say that the question is beyond your scope and invite a relevant one.\n"
            "2. When answering questions about:\n"
            "   - Speech content: Use the 'human speech information' array, specifically the 'text' field in each segment\n"
            "   - Language: Use the 'language detection' field\n"
            "   - Speakers: Use the 'speaker' field in 'human speech information' and 'number of speakers'\n"
            "   - Emotions: Use the 'emotion' field in 'human speech information'\n"
            "   - Gender: Use the 'gender' field in 'human speech information'\n"
            "   - Deepfake detection: Use the 'deepfake detection' field in 'other_information'. If it's 'fake', say the audio is detected as fake/deepfake. If it's 'real', say the audio is detected as real/authentic. If it's None, say the information is not available.\n"
            "3. Provide short, direct, and natural answers, as if you've heard the audio yourself.\n"
            "4. If a field is None or empty, clearly state that the information is not available.\n"
            "5. Do not provide explanations about which part of data was mentioned, or additional details unless explicitly requested.\n"
            "6. Keep responses casual yet professional, concise, and engaging, avoiding technical terms.\n"
        )
    }
]


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    audio_analysis_data: Optional[Dict[str, Any]] = None  # Extracted audio information


class ChatResponseChunk(BaseModel):
    content: str
    finish_reason: Optional[str] = None


def build_system_prompt(audio_analysis_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Build system prompt from template và audio analysis data
    Giống logic trong Streamlit
    Bỏ metadata (duration, sample_rate, channels) - không quan trọng
    """
    if audio_analysis_data is None:
        logger.warning("⚠️  No audio analysis data - using default prompt")
        return "You are a helpful assistant. The user has not uploaded any audio file yet. Please ask them to upload an audio file first."
    
    # Tạo bản copy và loại bỏ metadata
    filtered_analysis = audio_analysis_data.copy()
    if 'metadata' in filtered_analysis:
        del filtered_analysis['metadata']
        logger.info("🗑️  Removed metadata from audio analysis (not important for LLM)")
    
    # Format JSON properly for LLM to read
    json_data_formatted = json.dumps(filtered_analysis, indent=2, ensure_ascii=False)
    
    # Build system content
    system_content = (
        SYSTEM_PROMPT[-1]["Intro"] + 
        "\n\nExtracted Audio Information:\n" + 
        json_data_formatted + 
        "\n\n" + 
        SYSTEM_PROMPT[-1]["Outro"]
    )
    
    logger.info(f"✅ System prompt built successfully ({len(system_content)} chars, metadata removed)")
    return system_content


def stream_groq_response(messages_history: List[Dict[str, str]]):
    """
    Stream response from Groq LLM
    Generator function để yield chunks (SSE format)
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages_history,
            model=llm_config['LLM_MODEL'],
            temperature=llm_config['TEMPERATURE'],
            max_tokens=llm_config['MAX_TOKENS'],
            top_p=llm_config['TOP_P'],
            stop=llm_config['STOP'],
            stream=True,  # Enable streaming
        )
        
        # Stream chunks in SSE format
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                # SSE format: "data: content\n\n"
                yield f"data: {chunk.choices[0].delta.content}\n\n"
            
            # Check if finished
            if chunk.choices[0].finish_reason:
                yield f"data: [FINISH_REASON:{chunk.choices[0].finish_reason}]\n\n"
                break
                
    except Exception as e:
        yield f"data: [ERROR:{str(e)}]\n\n"


@app.get("/")
async def root():
    """Root endpoint - Test server connectivity"""
    logger.info("🔍 Root endpoint accessed")
    print("🔍 Root endpoint accessed - SERVER IS RUNNING!")
    import sys
    sys.stdout.flush()
    return {
        "message": "Chat API Service - Streaming LLM Chat",
        "status": "running",
        "endpoints": {
            "chat_stream": "/chat/stream",
            "chat": "/chat",
            "upload": "/upload (GET for info, POST for upload)",
            "files": "/files/{file_id}"
        }
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Chat endpoint với streaming response
    - Nhận messages history và audio analysis data
    - Build system prompt
    - Gọi Groq LLM với streaming
    - Trả về streaming response (SSE format)
    """
    try:
        import sys
        
        # Log request info
        print("\n" + "="*60)
        print("💬 CHAT STREAM REQUEST RECEIVED")
        print("="*60)
        logger.info("💬 Chat stream request received")
        
        # Log audio analysis data
        if request.audio_analysis_data:
            print("✅ Audio analysis data PRESENT:")
            logger.info("✅ Audio analysis data PRESENT")
            
            # Log summary
            audio_data = request.audio_analysis_data
            has_metadata = audio_data.get('metadata') is not None
            has_acoustics = audio_data.get('acoustics_information') is not None
            has_speech = audio_data.get('human_speech_information') is not None
            has_other = audio_data.get('other_information') is not None
            
            print(f"   - Metadata: {'✅' if has_metadata else '❌'}")
            print(f"   - Acoustics: {'✅' if has_acoustics else '❌'}")
            print(f"   - Speech: {'✅' if has_speech else '❌'}")
            print(f"   - Other info: {'✅' if has_other else '❌'}")
            logger.info(f"   - Metadata: {'✅' if has_metadata else '❌'}")
            logger.info(f"   - Acoustics: {'✅' if has_acoustics else '❌'}")
            logger.info(f"   - Speech: {'✅' if has_speech else '❌'}")
            logger.info(f"   - Other info: {'✅' if has_other else '❌'}")
            
            # Log speech segments count
            if has_speech:
                speech_info = audio_data.get('human_speech_information', {})
                speech_segments = speech_info.get('human speech information', []) if isinstance(speech_info, dict) else []
                print(f"   - Speech segments: {len(speech_segments)}")
                logger.info(f"   - Speech segments: {len(speech_segments)}")
                
                # Log first few speech texts
                if speech_segments and len(speech_segments) > 0:
                    first_texts = [seg.get('text', '') for seg in speech_segments[:3] if isinstance(seg, dict)]
                    print(f"   - First texts: {first_texts[:2]}")
                    logger.info(f"   - First texts: {first_texts[:2]}")
        else:
            print("❌ Audio analysis data NOT PRESENT")
            logger.warning("❌ Audio analysis data NOT PRESENT - Groq will not have audio context")
        
        sys.stdout.flush()
        
        # Build messages history
        messages_history = []
        
        # Add system prompt
        system_content = build_system_prompt(request.audio_analysis_data)
        
        # Log system prompt length
        print(f"📝 System prompt length: {len(system_content)} characters")
        logger.info(f"📝 System prompt length: {len(system_content)} characters")
        
        # Log first 500 chars of system prompt for debugging
        if len(system_content) > 500:
            print(f"📝 System prompt preview (first 500 chars):\n{system_content[:500]}...")
            logger.info(f"📝 System prompt preview (first 500 chars):\n{system_content[:500]}...")
        else:
            print(f"📝 System prompt:\n{system_content}")
            logger.info(f"📝 System prompt:\n{system_content}")
        
        sys.stdout.flush()
        
        messages_history.append({"role": "system", "content": system_content})
        
        # Add conversation history (excluding system messages, will be added above)
        for msg in request.messages:
            if msg.role != "system":  # Skip system messages, we handle them separately
                messages_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        print(f"📨 Total messages to Groq: {len(messages_history)}")
        logger.info(f"📨 Total messages to Groq: {len(messages_history)}")
        print("="*60 + "\n")
        sys.stdout.flush()
        
        # Return streaming response
        return StreamingResponse(
            stream_groq_response(messages_history),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Chat stream error: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_non_stream(request: ChatRequest):
    """
    Chat endpoint không streaming (fallback)
    """
    try:
        # Build messages history
        messages_history = []
        
        # Add system prompt
        system_content = build_system_prompt(request.audio_analysis_data)
        messages_history.append({"role": "system", "content": system_content})
        
        # Add conversation history
        for msg in request.messages:
            if msg.role != "system":
                messages_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Call Groq (non-streaming)
        chat_completion = groq_client.chat.completions.create(
            messages=messages_history,
            model=llm_config['LLM_MODEL'],
            temperature=llm_config['TEMPERATURE'],
            max_tokens=llm_config['MAX_TOKENS'],
            top_p=llm_config['TOP_P'],
            stop=llm_config['STOP'],
            stream=False,
        )
        
        return {
            "content": chat_completion.choices[0].message.content,
            "finish_reason": chat_completion.choices[0].finish_reason
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# GET endpoint - Phải định nghĩa TRƯỚC POST
@app.get("/upload")
async def upload_info():
    """
    GET endpoint để hiển thị thông tin về upload endpoint
    """
    # Force print để đảm bảo hiển thị
    print("\n📋📋📋 GET /upload - Upload info requested 📋📋📋")
    logger.info("📋 GET /upload - Upload info requested")
    import sys
    sys.stdout.flush()
    
    return {
        "message": "Upload endpoint - Use POST to upload files",
        "method": "POST",
        "endpoint": "/upload",
        "supported_types": ["audio", "image", "document"],
        "max_size": f"{MAX_FILE_SIZE / 1024 / 1024}MB",
        "audio_processing": "Automatic for audio files (wav, mp3, m4a, etc.)",
        "note": "Use POST method with multipart/form-data to upload files",
        "status": "GET endpoint is working!",
        "server_time": datetime.now().isoformat()
    }


@app.post("/upload", name="upload_file_post")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload file endpoint
    Hỗ trợ: audio (wav, mp3, m4a), image (jpg, png, gif), document (pdf, txt, docx)
    """
    try:
        # Force print và flush để đảm bảo log hiển thị ngay
        import sys
        print("\n" + "=" * 60)
        print("📤📤📤 UPLOAD REQUEST RECEIVED - SERVER PROCESSING 📤📤📤")
        print("=" * 60)
        sys.stdout.flush()
        sys.stderr.flush()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("📤📤📤 UPLOAD REQUEST RECEIVED - SERVER PROCESSING 📤📤📤")
        logger.info("=" * 60)
        
        if file.filename is None:
            logger.error("❌ Filename is None")
            raise HTTPException(status_code=400, detail="Filename is required")
        
        print(f"📁 Filename: {file.filename}")
        print(f"📋 Content-Type: {file.content_type}")
        logger.info(f"📁 Filename: {file.filename}")
        logger.info(f"📋 Content-Type: {file.content_type}")
        sys.stdout.flush()
        
        # Kiểm tra kích thước file
        file_content = await file.read()
        print(f"📦 File size: {len(file_content)} bytes ({len(file_content) / 1024:.2f} KB)")
        logger.info(f"📦 File size: {len(file_content)} bytes ({len(file_content) / 1024:.2f} KB)")
        sys.stdout.flush()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB")
        
        # Tạo unique filename
        file_ext = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Lưu file
        print(f"💾 Saving file to: {file_path}")
        logger.info(f"💾 Saving file to: {file_path}")
        sys.stdout.flush()
        
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        print(f"✅ File saved successfully!")
        logger.info(f"✅ File saved successfully!")
        sys.stdout.flush()
        
        # Xác định file type
        file_type = "other"
        is_audio = False
        if file_ext.lower() in ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']:
            file_type = "audio"
            is_audio = True
        elif file_ext.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            file_type = "image"
        elif file_ext.lower() in ['.pdf', '.txt', '.docx', '.doc']:
            file_type = "document"
        
        # Xử lý audio file nếu là audio
        audio_analysis = None
        if is_audio and AUDIO_PROCESSING_AVAILABLE:
            try:
                logger.info("🎵 Audio file detected - Starting analysis...")
                asc_aed_response, whisper_response, cap_df_response = await process_audio_file(
                    file.filename,
                    file_content,
                    AUDIO_SERVICE_ASC_AED,
                    AUDIO_SERVICE_WHISPER,
                    AUDIO_SERVICE_CAP_DF
                )
                
                # Merge kết quả
                audio_analysis = merge_audio_analysis(asc_aed_response, whisper_response, cap_df_response)
                logger.info("✅ Audio analysis completed!")
                logger.info(f"   Metadata: {audio_analysis.get('metadata') is not None}")
                logger.info(f"   Speech segments: {len(audio_analysis.get('human_speech_information', {}).get('human speech information', []))}")
            except Exception as e:
                logger.error(f"❌ Audio processing failed: {e}")
                logger.exception(e)
                audio_analysis = None
        
        result = {
            "file_id": unique_filename,
            "filename": file.filename,
            "file_type": file_type,
            "file_size": len(file_content),
            "uploaded_at": datetime.now().isoformat(),
            "url": f"/files/{unique_filename}",
            "audio_analysis": audio_analysis  # Thêm kết quả phân tích nếu có
        }
        
        # Log kết quả với cả print và logger để đảm bảo hiển thị
        print(f"💾 File saved to: {file_path}")
        logger.info(f"💾 File saved to: {file_path}")
        
        print("")
        print("✅✅✅ UPLOAD SUCCESSFUL ✅✅✅")
        logger.info("✅✅✅ UPLOAD SUCCESSFUL ✅✅✅")
        
        print(f"   📄 File ID: {unique_filename}")
        print(f"   📁 File Type: {file_type}")
        print(f"   📊 File Size: {len(file_content) / 1024:.2f} KB ({len(file_content) / 1024 / 1024:.2f} MB)")
        logger.info(f"   📄 File ID: {unique_filename}")
        logger.info(f"   📁 File Type: {file_type}")
        logger.info(f"   📊 File Size: {len(file_content) / 1024:.2f} KB ({len(file_content) / 1024 / 1024:.2f} MB)")
        
        if audio_analysis:
            print(f"   🎵 Audio Analysis: ✅ Available")
            logger.info(f"   🎵 Audio Analysis: ✅ Available")
            print(f"      - Metadata: {'✅' if audio_analysis.get('metadata') else '❌'}")
            logger.info(f"      - Metadata: {'✅' if audio_analysis.get('metadata') else '❌'}")
            speech_info = audio_analysis.get('human_speech_information', {})
            speech_segments = speech_info.get('human speech information', []) if isinstance(speech_info, dict) else []
            print(f"      - Speech segments: {len(speech_segments)}")
            logger.info(f"      - Speech segments: {len(speech_segments)}")
        else:
            if is_audio:
                print(f"   🎵 Audio Analysis: ⚠️ Not available (services may not be running)")
                logger.info(f"   🎵 Audio Analysis: ⚠️ Not available (services may not be running)")
        
        print("=" * 60)
        print("")
        logger.info("=" * 60)
        logger.info("")
        
        # Force flush để đảm bảo log hiển thị ngay
        sys.stdout.flush()
        sys.stderr.flush()
        
        return result
        
    except HTTPException as e:
        logger.error(f"❌ HTTP Exception: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {str(e)}")
        logger.exception(e)  # Log full traceback
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@app.get("/files/{file_id}")
async def get_file(file_id: str):
    """
    Download file endpoint
    """
    file_path = UPLOAD_DIR / file_id
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=Path(file_id).name,
        media_type="application/octet-stream"
    )


@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete file endpoint
    """
    file_path = UPLOAD_DIR / file_id
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        return {"message": "File deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

