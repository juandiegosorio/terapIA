import streamlit as st
from streamlit_navigation_bar import st_navbar
# Must be the first Streamlit command
st.set_page_config(
    page_title="Therapy Session Manager",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

import datetime
from pathlib import Path
import tempfile

import os
import torch
import whisper
import streamlit as st
from pydub import AudioSegment


# Fix for torch
torch.classes.__path__ = [] # add this line to manually set it to empty

# Custom styling
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #FFFFFF;
        font-family: Futura, "Trebuchet MS", Arial, sans-serif;
    }
    
    /* Navigation bar container */
    .nav-container {
        background-color: #6B9AC4;
        padding: 0;
        margin: 0;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Navigation button styling */
    .stButton > button {
        background-color: transparent;
        color: white;
        border: none;
        border-radius: 0;
        padding: 1rem 2rem;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
        height: 100%;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .stButton > button:hover {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: none;
    }
    
    /* Active button styling */
    .stButton > button[kind="primary"] {
        background-color: white;
        color: #6B9AC4;
        font-weight: 600;
    }
    
    /* Main content spacing */
    .main-content {
        margin-top: 0;
        padding: 2rem;
    }
    
    /* Ensure content is scrollable */
    .main .block-container {
        max-width: 100%;
        padding-top: 0;
        padding-bottom: 2rem;
    }
    
    /* Fix sidebar */
    [data-testid="stSidebarContent"] {
        background-color: #F7FAFC;
        padding: 2rem 1rem;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Main section styling */
    .main-section {
        background-color: #F8FAFC;
        padding: 20px;
        margin: 0 0 30px 0;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    
    /* Section title styling */
    .section-title {
        color: #1A365D;
        border-bottom: 2px solid #6B9AC4;
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-size: 1.5em;
        font-weight: 600;
    }
    
    /* Input field styling */
    .stTextInput, .stTextArea, .stDateInput {
        background-color: #EDF2F7;
        border: 1px solid #CBD5E0;
        font-family: Futura, "Trebuchet MS", Arial, sans-serif;
    }
    
    /* Regular button styling */
    .regular-button > button {
        background-color: #6B9AC4;
        color: #FFFFFF;
        border-radius: 4px;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s ease;
        font-family: Futura, "Trebuchet MS", Arial, sans-serif;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .regular-button > button:hover {
        background-color: #4A7BA7;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize data directories
DATA_DIR = Path("data")
PATIENTS_FILE = DATA_DIR / "patients.txt"
SESSIONS_DIR = DATA_DIR / "sessions"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = "uploads"
TRANSCRIPT_DIR = "transcripts"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# Initialize session state variables
if 'user_type' not in st.session_state:
    st.session_state.user_type = None


if 'current_transcript' not in st.session_state:
    st.session_state.current_transcript = ""

if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None

# Initialize audio-related session state variables
if 'audio_recording' not in st.session_state:
    st.session_state.audio_recording = None
if 'file_uploader' not in st.session_state:
    st.session_state.file_uploader = None
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

@st.cache_resource
def load_model(model_name: str):
    """Load the selected Whisper model."""
    return whisper.load_model(model_name)

# Modified audio transcription function that can handle both file paths and audio data
def transcribe_audio(audio_input, model_name: str):
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_input: Either a file path (str) or audio data (bytes or numpy array)
        model_name: Name of the Whisper model to use
    """
    model = load_model(model_name)
    
    # If audio_input is already a string (file path), use it directly
    if isinstance(audio_input, str):
        if os.path.exists(audio_input):
            try:
                result = model.transcribe(audio_input)
                return result["text"]
            except Exception as e:
                st.error(f"Error transcribing file path: {e}")
                return "Transcription failed. Please check FFmpeg installation and audio format."
    
    # If audio_input is not a string, it's likely a file-like object or bytes
    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_path = temp_file.name
    
    try:
        # If it's a file-like object with getbuffer method
        if hasattr(audio_input, 'getbuffer'):
            with open(temp_path, 'wb') as f:
                f.write(audio_input.getbuffer())
        # If it's bytes directly
        elif isinstance(audio_input, bytes):
            with open(temp_path, 'wb') as f:
                f.write(audio_input)
        else:
            st.error(f"Unsupported audio input type: {type(audio_input)}")
            os.unlink(temp_path)
            return "Transcription failed. Unsupported audio input type."
            
        # Now transcribe the temporary file
        result = model.transcribe(temp_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return f"Transcription failed: {str(e)}"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@st.cache_data
def convert_to_mp3(input_path: str, output_path: str):
    """Convert an audio file to MP3 format."""
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="mp3")
    return output_path

def get_next_patient_id():
    """Generate the next available patient ID as a number."""
    patients = get_all_patients()
    if not patients:
        return 1
    # Get the highest ID and add 1
    max_id = max(int(p[0]) for p in patients)
    return max_id + 1

def save_patient(patient_name):
    """Save a new patient with an automatically generated ID."""
    # Create patients file if it doesn't exist
    if not PATIENTS_FILE.exists():
        PATIENTS_FILE.write_text("")
    
    # Generate new patient ID
    patient_id = get_next_patient_id()
    
    # Append new patient
    with open(PATIENTS_FILE, "a") as f:
        f.write(f"{patient_id}|{patient_name}\n")
    return True

def get_patient(patient_name):
    if not PATIENTS_FILE.exists():
        return None
    
    with open(PATIENTS_FILE, "r") as f:
        for line in f:
            pid, name = line.strip().split("|")
            if name == patient_name:
                return (pid, name)
    return None

def get_all_patients():
    if not PATIENTS_FILE.exists():
        return []
    
    patients = []
    with open(PATIENTS_FILE, "r") as f:
        for line in f:
            if line.strip():
                pid, name = line.strip().split("|")
                patients.append((pid, name))
    return patients

def save_session(patient_id, session_data, transcript, audio_path=None):
    patient_dir = SESSIONS_DIR / patient_id
    patient_dir.mkdir(exist_ok=True)
    
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = patient_dir / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Save session metadata
    metadata_file = session_dir / "metadata.txt"
    with open(metadata_file, "w") as f:
        f.write(f"Date: {session_data['date']}\n")
        f.write(f"Notes: {session_data['notes']}\n")
        if audio_path:
            # Save audio file as audio.mp3 in the session directory
            audio_file = session_dir / "audio.mp3"
            with open(audio_file, "wb") as f_audio:
                f_audio.write(audio_path.getbuffer())
            # Store the relative path to the audio file
            f.write(f"Audio: audio.mp3\n")
    
    # Save transcript
    transcript_file = session_dir / "transcript.txt"
    with open(transcript_file, "w") as f:
        f.write(transcript)
    
    return session_id

def load_patient_sessions(patient_id):
    patient_session_dir = SESSIONS_DIR / patient_id
    if not patient_session_dir.exists():
        return []
    
    sessions = []
    for session_dir in patient_session_dir.glob("*"):
        if session_dir.is_dir():
            session_data = {}
            metadata_file = session_dir / "metadata.txt"
            with open(metadata_file, "r") as f:
                lines = f.readlines()
                session_data["date"] = lines[0].replace("Date: ", "").strip()
                session_data["notes"] = lines[1].replace("Notes: ", "").strip()
                session_data["id"] = session_dir.name
                # Check if audio path exists
                if len(lines) > 2 and lines[2].startswith("Audio: "):
                    session_data["audio"] = lines[2].replace("Audio: ", "").strip()
                else:
                    session_data["audio"] = None
            
            transcript_file = session_dir / "transcript.txt"
            with open(transcript_file, "r") as f:
                session_data["transcript"] = f.read()
            
            # Add session directory path
            session_data["session_dir"] = str(session_dir)
            
            sessions.append(session_data)
    
    return sorted(sessions, key=lambda x: x["date"], reverse=True)

def therapist_interface():
    st.markdown('<h1 class="section-title">🧑‍⚕️ Therapist Dashboard</h1>', unsafe_allow_html=True)
    
    # Patient management section
    patients = get_all_patients()
    if patients:
        patient_list = "\n".join([f"ID: {p[0]} - Name: {p[1]}" for p in patients])
        st.text_area("📋 Patient List", patient_list, height=100, disabled=True)
    else:
        st.info("No patients registered yet")
    
    with st.expander("➕ Add New Patient"):
        new_patient_name = st.text_input("Patient Name", placeholder="Full Name")
        
        if st.button("Add Patient", use_container_width=True):
            if new_patient_name:
                if save_patient(new_patient_name):
                    st.success(f"Patient {new_patient_name} added successfully!")
                else:
                    st.error("Failed to add patient!")
            else:
                st.warning("Please enter the patient's name")

    st.markdown("---")
    st.markdown('<h2 class="section-title">Create New Session</h2>', unsafe_allow_html=True)
    
    # Get all patients for the selector
    patients = get_all_patients()
    if not patients:
        st.warning("No patients registered yet. Please add a patient first.")
    else:
        # If we have a selected patient from search, use that as default
        default_idx = 0
        if 'selected_patient' in st.session_state and st.session_state.selected_patient is not None:
            for idx, p in enumerate(patients):
                if p[0] == st.session_state.selected_patient[0]:
                    default_idx = idx + 1  # +1 because of the "Select a patient" option
                    break
        
        patient_options = ["Select a patient"] + [f"{p[1]} (ID: {p[0]})" for p in patients]
        selected_patient = st.selectbox("Patient", patient_options, index=default_idx)
        
        if selected_patient == "Select a patient":
            st.warning("Please select a patient")
        else:
            # Session Content
            session_notes = st.text_area("Session Notes", placeholder="Enter session notes here...", height=150, key="session_notes")

            input_option = st.radio("How do you want to add the audio?", options=["Upload a file", "Record audio now"], index=1)
            
            uploaded_file = None

            if input_option == "Record audio now":
                uploaded_file = st.audio_input("Press the button to record", key="audio_input")
            elif input_option == "Upload a file":
                uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "wma", "aac", "flac", "mp4", "flv"], key="file_uploader")

            if uploaded_file and input_option == "Upload a file":
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"File uploaded: {uploaded_file.name}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.audio(file_path)
                
                with col2:
                    available_models = whisper.available_models()
                    model_name = st.selectbox("Choose a Whisper model", available_models, index=available_models.index("base"))
                
                if st.button("Generate Transcript"):
                    with st.spinner("Transcribing..."):
                        transcript_text = transcribe_audio(file_path, model_name)
                        st.text_area("Transcript", transcript_text, height=300)
                        st.download_button("Download Transcript", transcript_text, file_name=uploaded_file.name + ".txt", mime="text/plain")
                        st.success("Transcription complete!")
                        
                        # Store transcript and audio in session state
                        st.session_state.current_transcript = transcript_text
                        st.session_state.current_audio = uploaded_file
            
            elif uploaded_file and input_option == "Record audio now":
                st.success("Recording complete!")
                st.audio(uploaded_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Recording saved to file")
                
                with col2:
                    available_models = whisper.available_models()
                    model_name = st.selectbox("Choose a Whisper model", available_models, index=available_models.index("base"))
                
                if st.button("Generate Transcript"):
                    with st.spinner("Transcribing..."):
                        transcript_text = transcribe_audio(uploaded_file, model_name)
                        st.text_area("Transcript", transcript_text, height=300)
                        st.download_button("Download Transcript", transcript_text, file_name="recording.txt", mime="text/plain")
                        st.success("Transcription complete!")
                        
                        # Store transcript and audio in session state
                        st.session_state.current_transcript = transcript_text
                        st.session_state.current_audio = uploaded_file
            
            # Save button
            if 'current_transcript' in st.session_state and st.session_state.current_transcript:
                if st.button("Save Session", use_container_width=True):
                    # Extract patient ID from selection
                    patient_id = selected_patient.split("(ID: ")[1].rstrip(")")
                    patient_name = selected_patient.split(" (ID:")[0]
                    
                    session_data = {
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "notes": session_notes
                    }
                    
                    if save_session(patient_id, session_data, st.session_state.current_transcript, st.session_state.current_audio):
                        st.success("Session saved successfully!")
                        # Clear only the transcript and audio state
                        st.session_state.current_transcript = None
                        st.session_state.current_audio = None
                    else:
                        st.error("Failed to save session. Please try again.")

    st.markdown("---")
    st.markdown('<h1 class="section-title">🔍 Search for Patient</h1>', unsafe_allow_html=True)

    patient_name = st.text_input("Enter Patient Name", placeholder="Enter existing patient Name")
    
    if patient_name:
        patient = get_patient(patient_name)
        if patient:
            st.write(f"📝 Patient Name: {patient[1]}")
            
            # Set the selected patient in session state
            st.session_state.selected_patient = patient
            
            # Display previous sessions
            st.markdown("### Previous Sessions")
            sessions = load_patient_sessions(patient[0])
            for session in sessions:
                with st.expander(f"📅 Session from {session['date']}", expanded=False):
                    st.markdown("**📝 Notes:**")
                    st.write(session["notes"])
                    st.markdown("**📝 Transcript:**")
                    st.write(session["transcript"])
                    
                    # Check for audio file
                    audio_path = session.get("audio")
                    if audio_path:
                        audio_file = Path(session["session_dir"]) / audio_path
                        if audio_file.exists():
                            st.markdown("**🔊 Audio Recording:**")
                            st.audio(str(audio_file))
                        else:
                            st.warning("Audio file not found")
        else:
            st.error("❌ Patient ID not found!")

def patient_interface():
    st.markdown('<h1 class="section-title">👤 Patient Portal</h1>', unsafe_allow_html=True)
    
    patient_id = st.text_input("Enter your Patient ID", placeholder="Enter your ID")
    
    if patient_id:
        patient = get_patient(patient_id)
        if patient:
            st.write(f"👋 Welcome, {patient[1]}!")
            sessions = load_patient_sessions(patient[0]	)
            if sessions:
                for session in sessions:
                    with st.expander(f"📅 Session from {session['date']}"):
                        st.markdown("**Notes:**")
                        st.write(session["notes"])
                        st.markdown("**Transcript:**")
                        st.write(session["transcript"])
                        if session.get("audio"):
                            try:
                                uploaded_file = Path(session["session_dir"]) / session["audio"]
                                if uploaded_file.exists():
                                    st.markdown("**Audio Recording:**")
                                    st.audio(str(uploaded_file))
                            except Exception:
                                st.warning("Audio file not found")
            else:
                st.info("📭 No sessions found")
        else:
            st.error("❌ Patient ID not found!")

def homepage():
    st.markdown('<h1 class="section-title">Welcome to Therapy Session Manager</h1>', unsafe_allow_html=True)
    
    # Service cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background-color: #F8FAFC; padding: 20px; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 20px;'>
                <h3 style='color: #1A365D; margin-bottom: 15px;'>🎵 Music Therapy</h3>
                <p style='color: #2D3748;'>Music therapy is a form of therapy that uses music to help people with mental health issues. It can be used to help people with depression, anxiety, and other mental health issues.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background-color: #F8FAFC; padding: 20px; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 20px;'>
                <h3 style='color: #1A365D; margin-bottom: 15px;'>💊 psychedelic therapy </h3>
                <p style='color: #2D3748;'>Psychedelic therapy is a form of therapy that uses psychedelic drugs to help people with mental health issues. It can be used to help people with depression, anxiety, and other mental health issues.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background-color: #F8FAFC; padding: 20px; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 20px;'>
                <h3 style='color: #1A365D; margin-bottom: 15px;'>🤖 AI Therapy</h3>
                <p style='color: #2D3748;'>AI therapy is a form of therapy that uses AI to help people with mental health issues. It can be used to help people with depression, anxiety, and other mental health issues.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Additional information
    st.markdown("""
        <div style='background-color: #F8FAFC; padding: 20px; border-radius: 8px; border: 1px solid #E2E8F0; margin-top: 20px;'>
            <h2 style='color: #1A365D; margin-bottom: 15px;'>About Our Service</h2>
            <p style='color: #2D3748;'>TerapIA is a platform that uses AI to help people with mental health issues. It can be used to help people with depression, anxiety, and other mental health issues.</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    
    # Create pages dictionary
    pages = {
        "Home": homepage,
        "Therapist": therapist_interface,
        "Patient": patient_interface
    }
    
    # Add custom CSS for the navigation bar
    st.markdown("""
        <style>
        .nav-container {
            background-color: #1A365D;
            padding: 1rem;
            margin: 0;
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton > button {
            color: #1A365D !important;
            background-color: transparent !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            margin: 0 0.5rem !important;
            font-size: 1rem !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            border-radius: 4px !important;
            width: 100% !important;
        }
        .stButton > button:hover {
            background-color: rgba(255,255,255,0.1) !important;
            color: #54a0e8 !important;
        }
        .stButton > button[kind="primary"] {
            background-color: white !important;
            color: #54a0e8 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create the navigation bar container
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    # Create three columns for the navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🏠 Home Dashboard", 
                    key="nav_home",
                    use_container_width=True,
                    type="primary" if st.session_state.current_page == "Home" else "secondary"):
            st.session_state.current_page = "Home"
            st.rerun()
    
    with col2:
        if st.button("👨‍⚕️ Therapist Portal", 
                    key="nav_therapist",
                    use_container_width=True,
                    type="primary" if st.session_state.current_page == "Therapist" else "secondary"):
            st.session_state.current_page = "Therapist"
            st.rerun()
    
    with col3:
        if st.button("👤 Patient Portal", 
                    key="nav_patient",
                    use_container_width=True,
                    type="primary" if st.session_state.current_page == "Patient" else "secondary"):
            st.session_state.current_page = "Patient"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a small divider
    st.markdown("---")
    
    # Show the selected page
    if st.session_state.current_page in pages:
        pages[st.session_state.current_page]()
    else:
        homepage()

if __name__ == "__main__":
    main()