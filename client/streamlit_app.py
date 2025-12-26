import streamlit as st
import json
from prompts.sample_prompt import SYSTEM_PROMPT
from groq import Groq
from dotenv import load_dotenv
import os
import time
from utils import response_generator, merge_json_files
import yaml
import requests
import base64
import httpx
import asyncio
from copy import deepcopy

load_dotenv()
# Load YAML llm config
with open("./configs/llm_config.yaml", "r") as file:
    llm_config = yaml.safe_load(file)
#------------------Define API -----------------
client = Groq(
    api_key=os.getenv('GROQ_API_KEY'))

FASTAPI_ASC_AED = "http://server_asc_aed:8000/process_audio/"
FASTAPI_WHISPER = "http://server_whisper:8001/process_audio/"
FASTAPI_CAP_DF = "http://server_cap_df:8002/process_audio/"
# FASTAPI_ASC_AED = "http://localhost:8000/process_audio/"
# FASTAPI_WHISPER = "http://localhost:8001/process_audio/"
# FASTAPI_CAP_DF = "http://localhost:8002/process_audio/"

#--------------------------------------------------------------------------------------
# Streamlit Page Configuration
st.set_page_config(
    page_title="An Intelligent Audio Analyzer",
    page_icon="./img/chatbot.jpg",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        
    }
)

def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None

def decorate():
        #-------------------------- START APP--------------------------------------
    title_style = """
        <style>
        .title {
            text-align: center;
            font-size: 45px;
        }
        </style>
        """
    st.markdown(
    title_style,
    unsafe_allow_html=True
    )
    title  = """
    <h1 class = "title" >Aud-Sur: An Audio Analyzer Assistant for
        Audio Surveillance Application</h1>
    </div>
    """
    st.markdown(title,
                unsafe_allow_html=True)
    # Insert custom CSS for glowing effect
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #819ccc,
                0 0 10px #6b89bf,
                0 0 15px #5b7dba,
                0 0 20px #496fb3,
                0 0 25px #365b9c,
                0 0 30px #295096,
                0 0 35px #174391;
            position: relative;
            z-index: -1;
            border-radius: 45px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load and display sidebar image
    img_path = "./img/chatbot.jpg"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.write("**Aud-Sur**: An Audio Analyzer Assistant for Audio Surveillance Application")
    st.sidebar.markdown("---")
    
    # Display basic interactions
    show_basic_info = st.sidebar.checkbox("Show Basic Interactions", value=True)
    if show_basic_info:
        st.sidebar.markdown("""
        ### Basic Interactions
        - **Upload your audio file**: Click the "Browze Files" button to upload your audio file.   
        - **Extract information**: Click "Extract information" button and wait for the information to be extracted.                         
        - **Ask anything about the audio**: Enter your question in the text box.
        """)



async def fetch_response(client, url, file_name, file_contents):
    """Asynchronously send a POST request to a FastAPI service."""
    files = {"file": (file_name, file_contents, "audio/wav")}
    response = await client.post(url, files=files)
    return response.json()

async def process_audio(uploaded_file): 
    """Send requests to multiple FastAPI services in parallel."""  
    timeout = httpx.Timeout(None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Read file contents once
        file_contents = uploaded_file.getvalue()
        file_name = uploaded_file.name  # Get the filename
        
        tasks = [
            asyncio.create_task(fetch_response(client, FASTAPI_ASC_AED, file_name, file_contents)),
            asyncio.create_task(fetch_response(client, FASTAPI_WHISPER, file_name, file_contents)),
            asyncio.create_task(fetch_response(client, FASTAPI_CAP_DF, file_name, file_contents)),
        ]

        responses = await asyncio.gather(*tasks)
    
    return responses



def main():
    decorate()

    # ============ GET INPUT AUDIO ==========================
    uploaded_file = st.file_uploader("Upload your WAV file:", type=["wav"])

    # Display the result
    st.write("Your uploaded wav file: ")
    st.audio(uploaded_file, format = 'audio/wav')

    # ============ PROCESS FILE ========================
    if "json_label" not in st.session_state:
        st.session_state.json_label = None
    if st.button("Extract information") and uploaded_file:
        with st.spinner('Extracting information from input file...'):
            st.session_state.messages = []
            start_time = time.time()

           
            responses = asyncio.run(process_audio(uploaded_file))
            response_asc_aed, response_whisper, response_cap_df = responses
            
            # Debug: Show raw responses (can be commented out in production)
            with st.expander("Debug: Raw API Responses", expanded=False):
                st.write("ASC-AED Response:", response_asc_aed)
                st.write("Whisper Response:", response_whisper)
                st.write("CAP-DF Response:", response_cap_df)
    
            end_time = time.time()
            # Standalize
            final_response = merge_json_files(response_asc_aed, response_whisper, response_cap_df)
            
            # Debug: Show merged response
            with st.expander("Debug: Merged Response", expanded=False):
                st.json(final_response)
            save_json_dir = './save_jsons'
            os.makedirs(save_json_dir, exist_ok=True)  # Ensure the directory exists
            filename = f"{save_json_dir}/{uploaded_file.name[:-4]}.json"
            with open(filename, "w") as f:
                json.dump([final_response], f, indent=4)


            st.session_state.json_label = final_response
            st.success(f"Processing completed! Processing time: {end_time - start_time:.2f}s")


    # ============ PROCESS CONVERSATION ==========================
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app 
    for message in st.session_state.messages:
        if message["role"] == "user":
            # Display user message without avatar
            with st.chat_message("user", avatar="./img/person.jpg"):
                st.markdown(message["content"])
        else:
            # Display assistant message with avatar
            with st.chat_message("assistant", avatar="./img/chatbot.jpg"):
                st.markdown(message["content"])


    # Accept user input
    if prompt := st.chat_input("Enter your question here?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user", avatar="./img/person.jpg"):
            st.markdown(prompt)

        # Prepare conversation history for LLM
        # Check if json_label exists and is not None
        if st.session_state.json_label is None:
            messages_history = [{"role": "system", "content": "You are a helpful assistant. The user has not uploaded any audio file yet. Please ask them to upload an audio file first."}]
        else:
            # Format JSON properly for LLM to read
            json_data_formatted = json.dumps(st.session_state.json_label, indent=2, ensure_ascii=False)
            system_content = SYSTEM_PROMPT[-1]["Intro"] + "\n\nExtracted Audio Information:\n" + json_data_formatted + "\n\n" + SYSTEM_PROMPT[-1]["Outro"]
            messages_history = [{"role": "system", "content": system_content}]
        messages_history.extend(st.session_state.messages)  # Append previous messages 
        messages_history.append({"role": "user", "content": prompt})  # Add current question
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="./img/chatbot.jpg"):
            chat_completion = client.chat.completions.create(
        messages = messages_history,
        model=llm_config['LLM_MODEL'],
        temperature=llm_config['TEMPERATURE'],
        max_tokens=llm_config['MAX_TOKENS'],
        top_p=llm_config['TOP_P'],
        stop=llm_config['STOP'],
        stream=llm_config['STREAM'],
    )
            with st.spinner('Thinking...'):
                
                llm_output_text = chat_completion.choices[0].message.content
            response = st.write_stream(response_generator(llm_output_text))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
if __name__ == "__main__":
    main()

