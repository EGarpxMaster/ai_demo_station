import os
import platform
import cv2
import torch
import gradio as gr
import requests
import gdown
from ultralytics import YOLO
from faster_whisper import WhisperModel
import threading
import time

# ==========================================
# 0. CONFIGURACIÓN Y AUTO-DESCARGAS
# ==========================================
YOLO_MODEL_PATH = "models/yolo_custom.pt"
YOLO_GDRIVE_ID = "1--GDRIVE-ID-HERE--" # Rellenar con ID real

# URLs de los otros servicios (Docker)
BRAIN_URL = os.getenv("BRAIN_URL", "http://localhost:11434")
VOICE_URL = os.getenv("VOICE_URL", "http://localhost:5002")

def download_models():
    if not os.path.exists(YOLO_MODEL_PATH):
        print("[Vision Service] Descargando modelo YOLO personalizado...")
        os.makedirs("models", exist_ok=True)
        try:
            # Reemplazar con lógica real de descarga. 
            # Por ahora usamos el YOLOv8n oficial si falla gdown
            gdown.download(id=YOLO_GDRIVE_ID, output=YOLO_MODEL_PATH, quiet=False)
        except Exception as e:
            print(f"Error descargando desde Drive, usando YOLOv8n base: {e}")
            from ultralytics.utils.downloads import download
            download("https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt", dir="models")
            os.rename("models/yolov8n.pt", YOLO_MODEL_PATH)
    else:
        print("[Vision Service] Modelo YOLO detectado localmente.")

# ==========================================
# 1. AUDITORÍA DE HARDWARE (VRAM)
# ==========================================
def setup_hardware():
    print("[Vision Service] Iniciando Auditoría de Hardware...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Detectada: {gpu_name} ({total_vram:.2f} GB VRAM totales)")
        
        # Limitar la VRAM para YOLO/vision_service al 30% (aprox 2.4GB en una RTX 4060 de 8GB)
        # Dejando el resto para Ollama (LLM) y el servicio de voz TTS
        fraction = 0.30
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        print(f"Límite de VRAM aplicado para Visión: {fraction*100}% (~{total_vram*fraction:.2f} GB)")
    else:
        print("Advertencia: No se detectó GPU CUDA válida. Ejecutando en CPU.")

# ==========================================
# 2. DETECCIÓN DE SO PARA CÁMARA
# ==========================================
def get_camera_source():
    # Detectamos el SO del Host (o contenedor)
    # Nota: Dentro de Docker este siempre será Linux, pero respetamos la arquitectura universal solicitada.
    current_os = platform.system()
    print(f"[Vision Service] OS Detectado: {current_os}")
    
    # Comprobar si estamos en WSL
    is_wsl = 'microsoft-standard' in platform.uname().release.lower()
    
    if current_os == "Windows" or is_wsl:
        source = 0
        print("[Vision Service] Asignando cámara: 0 (Windows API)")
    elif current_os == "Linux":
        # Preferir cámaras externas (usualmente video2 o video1) antes que la integrada (video0)
        if os.path.exists("/dev/video2"):
            source = "/dev/video2"
            print("[Vision Service] Asignando cámara externa: /dev/video2")
        elif os.path.exists("/dev/video1"):
            source = "/dev/video1"
            print("[Vision Service] Asignando cámara externa: /dev/video1")
        elif os.path.exists("/dev/video0"):
            source = "/dev/video0"
            print("[Vision Service] Asignando cámara integrada: /dev/video0 (V4L2 Linux)")
        else:
            source = 0
            print("[Vision Service] /dev/videoX no existe, fallback a 0")
    else:
        source = 0
        print(f"[Vision Service] SO no contemplado, usando cámara 0 por defecto.")
        
    return source

# ==========================================
# 3. LÓGICA CORE (VISIÓN, VOZ & LLM)
# ==========================================
current_frame = None

def capture_thread(source):
    global current_frame
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        success, frame = cap.read()
        if success:
            # Espejar la imagen horizontalmente e invertir colores si es necesario
            frame = cv2.flip(frame, 1)
            current_frame = frame
        time.sleep(0.01)

def get_processed_frame(model):
    global current_frame
    if current_frame is not None:
        results = model(current_frame, verbose=False)
        annotated = results[0].plot()
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return None

def process_audio_chat(audio_path, history, stt_model):
    if not audio_path:
        return history, None, ""
        
    print(f"[Vision Service] Transcribiendo audio: {audio_path}")
    segments, _ = stt_model.transcribe(audio_path, beam_size=5)
    transcription = " ".join([segment.text for segment in segments]).strip()
    
    if "unibot" not in transcription.lower():
        # Wake word no detectada - Rechazo silencioso estilo Alexa
        print("[Vision Service] Ignorando audio: No se detectó 'Unibot'.")
        return history, None, "Esperando comando 'Unibot'..."
    
    # Wake word detectada
    history.append({"role": "user", "content": transcription})
    
    # Prompt de sistema forzando brevedad
    system_prompt = "Eres Unibot, una asistente de demostración de Inteligencia Artificial que siempre escucha y ve todo el entorno de forma continua. Responde de forma casual, carismática y MUY breve (1 o 2 oraciones máximo)."
    
    payload = {
        "model": "llama3.2:1b", 
        "prompt": transcription,
        "system": system_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(f"{BRAIN_URL}/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        bot_reply = response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        bot_reply = f"Error conectando al Cerebro (LLM): {e}"
        
    history.append({"role": "assistant", "content": bot_reply})
    
    # Autogenerar audio de la respuesta
    audio_reply = speak_text(bot_reply)
    
    return history, audio_reply, "Transcrito y procesado."

def speak_text(texto):
    if not texto:
        return None
    try:
        payload = {"text": texto}
        # Petición al endpoint del voice_service
        # Retorna el binario de audio
        response = requests.post(f"{VOICE_URL}/synthesize", json=payload)
        response.raise_for_status()
        
        # Guardar audio temp
        audio_path = "temp_response.wav"
        with open(audio_path, 'wb') as f:
            f.write(response.content)
        return audio_path
    except Exception as e:
        print(f"Error en TTS: {e}")
        return None

# ==========================================
# MAIN Y UI
# ==========================================
if __name__ == "__main__":
    download_models()
    setup_hardware()
    
    cam_source = get_camera_source()
    
    # Cargar YOLO (lo enviamos la GPU si está)
    print("[Vision Service] Cargando modelo YOLO...")
    model = YOLO(YOLO_MODEL_PATH)
    
    # Iniciar captura nativa de cámara en thread
    threading.Thread(target=capture_thread, args=(cam_source,), daemon=True).start()
    
    # Cargar Whisper (STT)
    print("[Vision Service] Cargando modelo Faster-Whisper para reconocimiento de voz...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # compute_type "int8" ensures it fits well in VRAM alongside YOLO and LLM
    stt_model = WhisperModel("tiny", device=device, compute_type="int8" if device=="cuda" else "int8")

    print("[Vision Service] Levantando Panel de Control UI...")
    with gr.Blocks(title="Estación Demostrativa IA", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Estación de IA: Visión, LLM y Voz")
        gr.Markdown("Demostración integral con YOLOv8, Ollama, y TTS corriendo de forma local.")
        
        with gr.Tab("👁️ Visión por Computadora"):
            gr.Markdown("### Detección de objetos en tiempo real")
            # Un solo componente no interactivo
            camera_view = gr.Image(label="Feed Nativo YOLOv8", interactive=False)
            
            # Streaming simulado a ~25 FPS (0.04) para UI más fluida sin WebRTC de Gradio
            timer = gr.Timer(0.04)
            timer.tick(lambda: get_processed_frame(model), inputs=None, outputs=camera_view)

        with gr.Tab("🎙️ Asistente de Voz IA"):
            gr.Markdown("### Habla con Unibot (Dí 'Unibot' al inicio de tu pregunta)")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Micrófono (Habla aquí)")
                    status_text = gr.Textbox(label="Estado STT", interactive=False)
                with gr.Column(scale=1):
                    audio_output = gr.Audio(label="Respuesta de Unibot", autoplay=True)
            
            chatbot = gr.Chatbot(height=300)
            
            # Conexión principal
            audio_input.stop_recording(
                fn=lambda a, h: process_audio_chat(a, h, stt_model),
                inputs=[audio_input, chatbot],
                outputs=[chatbot, audio_output, status_text]
            ).then(
                fn=lambda: None, # Limpiar input de micro
                inputs=None,
                outputs=audio_input
            )

    demo.launch(server_name="0.0.0.0", server_port=7860)
