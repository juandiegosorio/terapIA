import streamlit as st
# Must be the first Streamlit command
st.set_page_config(
    page_title="Gestor de Sesiones de Terapia",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Add Font Awesome CSS
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .icon {
            margin-right: 8px;
            color: #1A365D;
        }
        .nav-icon {
            margin-right: 8px;
        }
    </style>
""", unsafe_allow_html=True)

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
            lines = []
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                try:
                    with open(metadata_file, "r", encoding="latin-1") as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    try:
                        with open(metadata_file, "r", encoding="windows-1252") as f:
                            lines = f.readlines()
                    except UnicodeDecodeError:
                        try:
                            with open(metadata_file, "r", errors='replace') as f:
                                lines = f.readlines()
                        except:
                            lines = ["Date: Error", "Notes: Error", "Audio: Error"]

            if lines:
                try:
                    session_data["date"] = lines[0].replace("Date: ", "").strip()
                    session_data["notes"] = lines[1].replace("Notes: ", "").strip()
                    if len(lines) > 2 and lines[2].startswith("Audio: "):
                        session_data["audio"] = lines[2].replace("Audio: ", "").strip()
                    else:
                        session_data["audio"] = None
                except IndexError:
                    session_data["date"] = "Error"
                    session_data["notes"] = "Error"
                    session_data["audio"] = "Error"
            else:
                session_data["date"] = "No Date"
                session_data["notes"] = "No Notes"
                session_data["audio"] = None

            transcript_file = session_dir / "transcript.txt"
            try:
                with open(transcript_file, "r", errors='replace') as f: # force the use of replace.
                    session_data["transcript"] = f.read()
            except:
                session_data["transcript"] = "Error reading transcript"

            session_data["session_dir"] = str(session_dir)
            sessions.append(session_data)

    return sorted(sessions, key=lambda x: x["date"], reverse=True)

def therapist_interface():
    st.markdown('<h1 class="section-title"><i class="fas fa-user-md icon"></i>Panel del Terapeuta</h1>', unsafe_allow_html=True)
    
    # Patient management section
    patients = get_all_patients()
    if patients:
        patient_list = "\n".join([f"ID: {p[0]} - Nombre: {p[1]}" for p in patients])
        st.text_area('Lista de Pacientes', patient_list, height=100, disabled=True)
    else:
        st.info("No hay pacientes registrados aún")
    
    with st.expander('Agregar Nuevo Paciente'):
        new_patient_name = st.text_input("Nombre del Paciente", placeholder="Nombre Completo")
        
        if st.button("Agregar Paciente", use_container_width=True):
            if new_patient_name:
                if save_patient(new_patient_name):
                    st.success(f"¡Paciente {new_patient_name} agregado exitosamente!")
                else:
                    st.error("¡Error al agregar paciente!")
            else:
                st.warning("Por favor ingrese el nombre del paciente")

    st.markdown("---")
    st.markdown('<h2 class="section-title"><i class="fas fa-calendar-plus icon"></i>Crear Nueva Sesión</h2>', unsafe_allow_html=True)
    
    # Get all patients for the selector
    patients = get_all_patients()
    if not patients:
        st.warning("No hay pacientes registrados. Por favor agregue un paciente primero.")
    else:
        # If we have a selected patient from search, use that as default
        default_idx = 0
        if 'selected_patient' in st.session_state and st.session_state.selected_patient is not None:
            for idx, p in enumerate(patients):
                if p[0] == st.session_state.selected_patient[0]:
                    default_idx = idx + 1  # +1 because of the "Select a patient" option
                    break
        
        patient_options = ["Seleccionar un paciente"] + [f"{p[1]} (ID: {p[0]})" for p in patients]
        selected_patient = st.selectbox("Paciente", patient_options, index=default_idx)
        
        if selected_patient == "Seleccionar un paciente":
            st.warning("Por favor seleccione un paciente")
        else:
            # Session Content
            session_notes = st.text_area("Notas de la Sesión", placeholder="Ingrese las notas de la sesión aquí...", height=150, key="session_notes")

            input_option = st.radio("¿Cómo desea agregar el audio?", options=["Subir un archivo", "Grabar audio ahora"], index=1)
            
            uploaded_file = None

            if input_option == "Grabar audio ahora":
                uploaded_file = st.audio_input("Presione el botón para grabar", key="audio_input")
            elif input_option == "Subir un archivo":
                uploaded_file = st.file_uploader("Subir un archivo de audio", type=["wav", "mp3", "ogg", "wma", "aac", "flac", "mp4", "flv"], key="file_uploader")

            if uploaded_file and input_option == "Subir un archivo":
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"Archivo subido: {uploaded_file.name}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.audio(file_path)
                
                with col2:
                    available_models = whisper.available_models()
                    model_name = st.selectbox("Elija un modelo de Whisper", available_models, index=available_models.index("base"))
                
                if st.button("Generar Transcripción"):
                    with st.spinner("Transcribiendo..."):
                        transcript_text = transcribe_audio(file_path, model_name)
                        st.text_area("Transcripción", transcript_text, height=300)
                        st.download_button("Descargar Transcripción", transcript_text, file_name=uploaded_file.name + ".txt", mime="text/plain")
                        st.success("¡Transcripción completada!")
                        
                        # Store transcript and audio in session state
                        st.session_state.current_transcript = transcript_text
                        st.session_state.current_audio = uploaded_file
            
            elif uploaded_file and input_option == "Grabar audio ahora":
                st.success("¡Grabación completada!")
                st.audio(uploaded_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Grabación guardada en archivo")
                
                with col2:
                    available_models = whisper.available_models()
                    model_name = st.selectbox("Elija un modelo de Whisper", available_models, index=available_models.index("base"))
                
                if st.button("Generar Transcripción"):
                    with st.spinner("Transcribiendo..."):
                        transcript_text = transcribe_audio(uploaded_file, model_name)
                        st.text_area("Transcripción", transcript_text, height=300)
                        st.download_button("Descargar Transcripción", transcript_text, file_name="grabacion.txt", mime="text/plain")
                        st.success("¡Transcripción completada!")
                        
                        # Store transcript and audio in session state
                        st.session_state.current_transcript = transcript_text
                        st.session_state.current_audio = uploaded_file
            
            # Save button
            if 'current_transcript' in st.session_state and st.session_state.current_transcript:
                if st.button("Guardar Sesión", use_container_width=True):
                    # Extract patient ID from selection
                    patient_id = selected_patient.split("(ID: ")[1].rstrip(")")
                    patient_name = selected_patient.split(" (ID:")[0]
                    
                    session_data = {
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "notes": session_notes
                    }
                    
                    if save_session(patient_id, session_data, st.session_state.current_transcript, st.session_state.current_audio):
                        st.success("¡Sesión guardada exitosamente!")
                        # Clear only the transcript and audio state
                        st.session_state.current_transcript = None
                        st.session_state.current_audio = None
                    else:
                        st.error("Error al guardar la sesión. Por favor intente nuevamente.")

    st.markdown("---")
    st.markdown('<h1 class="section-title"><i class="fas fa-search icon"></i>Buscar Paciente</h1>', unsafe_allow_html=True)

    patient_name = st.text_input("Ingrese el Nombre del Paciente", placeholder="Ingrese el nombre del paciente existente")
    
    if patient_name:
        patient = get_patient(patient_name)
        if patient:
            st.write(f'<i class="fas fa-user icon"></i>Nombre del Paciente: {patient[1]}', unsafe_allow_html=True)
            
            # Set the selected patient in session state
            st.session_state.selected_patient = patient
            
            # Display previous sessions
            st.markdown("### Sesiones Anteriores")
            sessions = load_patient_sessions(patient[0])
            for session in sessions:
                with st.expander(f'Sesión del {session["date"]}', expanded=False):
                    st.markdown("**Notas:**")
                    st.write(session["notes"])
                    st.markdown("**Transcripción:**")
                    st.write(session["transcript"])
                    
                    # Check for audio file
                    audio_path = session.get("audio")
                    if audio_path:
                        audio_file = Path(session["session_dir"]) / audio_path
                        if audio_file.exists():
                            st.markdown("**Grabación de Audio:**")
                            st.audio(str(audio_file))
                        else:
                            st.warning("Archivo de audio no encontrado")
        else:
            st.error("❌ ¡Paciente no encontrado!")

def patient_interface():
    st.markdown('<h1 class="section-title"><i class="fas fa-user icon"></i>Portal del Paciente</h1>', unsafe_allow_html=True)
    
    patient_id = st.text_input("Ingrese su Nombre", placeholder="Ingrese su Nombre")
    
    if patient_id:
        patient = get_patient(patient_id)
        if patient:
            st.write(f'<i class="fas fa-hand-sparkles icon"></i>¡Bienvenido, {patient[1]}!', unsafe_allow_html=True)
            sessions = load_patient_sessions(patient[0])
            if sessions:
                for session in sessions:
                    with st.expander(f'Sesión del {session["date"]}'):
                        st.markdown("**Notas:**")
                        st.write(session["notes"])
                        st.markdown("**Transcripción:**")
                        st.write(session["transcript"])
                        if session.get("audio"):
                            try:
                                uploaded_file = Path(session["session_dir"]) / session["audio"]
                                if uploaded_file.exists():
                                    st.markdown("**Grabación de Audio:**")
                                    st.audio(str(uploaded_file))
                            except Exception:
                                st.warning("Archivo de audio no encontrado")
            else:
                st.info("📭 No se encontraron sesiones")
        else:
            st.error("❌ ¡ID de paciente no encontrado!")

def homepage():
    st.markdown('<h1 class="section-title"><i class="fas fa-brain icon"></i>Bienvenido al Gestor de Sesiones de Terapia</h1>', unsafe_allow_html=True)
    
    # Add custom CSS for service cards
    st.markdown("""
        <style>
        .service-cards-container {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .service-card {
            background-color: #F8FAFC;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .service-card h3 {
            color: #1A365D;
            margin-bottom: 15px;
            min-height: 40px;
        }
        .service-card p {
            color: #2D3748;
            margin: 0;
        }
        .service-icon {
            font-size: 2em;
            margin-bottom: 15px;
            color: #1A365D;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Service cards
    st.markdown('<div class="service-cards-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="service-card">
                <div class="service-icon"><i class="fas fa-music"></i></div>
                <h3>Musicoterapia</h3>
                <p>La musicoterapia es una forma de terapia que utiliza la música para ayudar a las personas con problemas de salud mental. Puede ser utilizada para ayudar a personas con depresión, ansiedad y otros problemas de salud mental.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="service-card">
                <div class="service-icon"><i class="fas fa-pills"></i></div>
                <h3>Terapia con Psicodélicos</h3>
                <p>La terapia con psicodélicos es una forma de terapia que utiliza sustancias psicodélicas para ayudar a las personas con problemas de salud mental. Puede ser utilizada para ayudar a personas con depresión, ansiedad y otros problemas de salud mental.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="service-card">
                <div class="service-icon"><i class="fas fa-robot"></i></div>
                <h3>Terapia con IA</h3>
                <p>La terapia con IA es una forma de terapia que utiliza inteligencia artificial para ayudar a las personas con problemas de salud mental. Puede ser utilizada para ayudar a personas con depresión, ansiedad y otros problemas de salud mental.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional information
    st.markdown("""
        <div style='background-color: #F8FAFC; padding: 20px; border-radius: 8px; border: 1px solid #E2E8F0; margin-top: 20px;'>
            <h2 style='color: #1A365D; margin-bottom: 15px;'><i class="fas fa-info-circle icon"></i>Sobre Nuestro Servicio</h2>
            <p style='color: #2D3748;'>TerapIA es una plataforma que utiliza IA para ayudar a las personas con problemas de salud mental. Puede ser utilizada para ayudar a personas con depresión, ansiedad y otros problemas de salud mental.</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Inicio'
    
    # Create pages dictionary
    pages = {
        "Inicio": homepage,
        "Terapeuta": therapist_interface,
        "Paciente": patient_interface
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
        if st.button("Panel Principal", 
                    key="nav_home",
                    use_container_width=True,
                    type="primary" if st.session_state.current_page == "Inicio" else "secondary"):
            st.session_state.current_page = "Inicio"
            st.rerun()
    
    with col2:
        if st.button("Portal del Terapeuta", 
                    key="nav_therapist",
                    use_container_width=True,
                    type="primary" if st.session_state.current_page == "Terapeuta" else "secondary"):
            st.session_state.current_page = "Terapeuta"
            st.rerun()
    
    with col3:
        if st.button("Portal del Paciente", 
                    key="nav_patient",
                    use_container_width=True,
                    type="primary" if st.session_state.current_page == "Paciente" else "secondary"):
            st.session_state.current_page = "Paciente"
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