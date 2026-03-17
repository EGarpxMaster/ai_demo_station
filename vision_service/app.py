import os
import platform
import cv2
import torch
import gradio as gr
import requests
import gdown
from ultralytics import YOLO

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
        if os.path.exists("/dev/video0"):
            source = "/dev/video0"
            print("[Vision Service] Asignando cámara: /dev/video0 (V4L2 Linux)")
        else:
            source = 0
            print("[Vision Service] /dev/video0 no existe, fallback a 0")
    else:
        source = 0
        print(f"[Vision Service] SO no contemplado, usando cámara 0 por defecto.")
        
    return source

# ==========================================
# 3. LÓGICA CORE (VISIÓN & LLM)
# ==========================================
def process_frame(frame, model):
    if frame is None:
        return None
    # Inferencia
    results = model(frame, verbose=False)
    # Dibujar bbox
    annotated_frame = results[0].plot()
    return annotated_frame

def chat_with_guide(user_msg, history):
    if not user_msg:
        return history, ""
    
    # Añadir mensaje de usuario al historial
    history.append((user_msg, ""))
    
    # Preparar payload para Ollama
    # Usamos llama3.2:1b como default ligero
    payload = {
        "model": "llama3.2:1b", 
        "prompt": user_msg,
        "stream": False
    }
    
    try:
        response = requests.post(f"{BRAIN_URL}/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        bot_reply = response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        bot_reply = f"Error conectando al Cerebro (LLM): {e}"
        
    history[-1] = (user_msg, bot_reply)
    return history, ""

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
    
    def inference_loop(image):
        return process_frame(image, model)

    print("[Vision Service] Levantando Panel de Control UI...")
    
    with gr.Blocks(title="Estación Demostrativa IA", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Estación de IA: Visión, LLM y Voz")
        gr.Markdown("Demostración integral con YOLOv8, Ollama, y TTS corriendo de forma local.")
        
        with gr.Row():
            # Panel Izquierdo: Cámara
            with gr.Column(scale=1):
                gr.Markdown("### Visto por la IA")
                # En Gradio 4, Image es streaming nativo desde cámara del cliente
                # Para conectarse a cámara local del servidor (/dev/video0) requiere Custom Component 
                # o un frame en bucle. Usaremos la webcam del navegador web (cliente) para mejor compatibilidad WSL.
                camera_input = gr.Image(sources=["webcam"], streaming=True)
                camera_output = gr.Image()
                
                # Link input -> output
                camera_input.stream(inference_loop, inputs=camera_input, outputs=camera_output)

            # Panel Derecho: Chat y TTS
            with gr.Column(scale=1):
                gr.Markdown("### Guía IA (Chat)")
                chatbot = gr.Chatbot(height=300)
                msg = gr.Textbox(placeholder="Escribe algo a la guía...", label="Mensaje")
                
                # Chat functionality
                msg.submit(chat_with_guide, [msg, chatbot], [chatbot, msg])
                
                # TTS functionality
                gr.Markdown("### Voz")
                tts_btn = gr.Button("🗣️ Hablar última respuesta")
                audio_output = gr.Audio(label="Audio Sintetizado", autoplay=True)
                
                def extract_last_reply(chat_hist):
                    if not chat_hist: return ""
                    return chat_hist[-1][1]
                    
                tts_btn.click(
                    fn=extract_last_reply, 
                    inputs=chatbot, 
                    outputs=msg # Variable temporal
                ).then(
                    fn=speak_text,
                    inputs=msg,
                    outputs=audio_output
                ).then(
                    fn=lambda: "",
                    inputs=None,
                    outputs=msg # Limpiar variable temporal
                )

    demo.launch(server_name="0.0.0.0", server_port=7860)
