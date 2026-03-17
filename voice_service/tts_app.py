import os
import torch
from flask import Flask, request, send_file
import tempfile
import numpy as np
from scipy.io import wavfile

# ==========================================
# 1. AUDITORÍA DE HARDWARE (VRAM)
# ==========================================
print("[Voice Service] Iniciando Auditoría de Hardware...")
if torch.cuda.is_available():
    # Asignamos máximo el 20% de VRAM (aprox 1.6GB en una GPU de 8GB) para el servicio de voz TTS
    fraction = 0.20
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(f"Límite de VRAM aplicado para TTS de Voz: {fraction*100}%")
else:
    print("Advertencia: No se detectó GPU CUDA válida. Ejecutando TTS en CPU.")

app = Flask(__name__)

# ==========================================
# 2. CARGAR MODELO TTS (SILERO)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Voice Service] Cargando modelo Silero TTS en {device}...")

# Silero usa un repositorio de PyTorch Hub para descargar pesos
# Debido a límites de tasa de la API de GitHub (que causan KeyError 'Authorization'),
# descargaremos el repositorio localmente si no existe.
local_repo_dir = '/app/models/silero-tts'
if not os.path.exists(local_repo_dir):
    print("[Voice Service] Descargando repositorio de Silero TTS localmente para evitar rate-limits...")
    import urllib.request
    import zipfile
    import shutil
    
    zip_path = '/app/models/silero.zip'
    url = "https://github.com/snakers4/silero-models/archive/refs/heads/master.zip"
    
    # Descargar
    urllib.request.urlretrieve(url, zip_path)
    print("[Voice Service] Descomprimiendo...")
    
    # Extraer
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/app/models/')
        
    # Renombrar carpeta
    extracted_dir = '/app/models/silero-models-master'
    if os.path.exists(extracted_dir):
        shutil.move(extracted_dir, local_repo_dir)
        
    # Limpiar
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print("[Voice Service] Repositorio Silero TTS preparado localmente.")

try:
    model, example_text = torch.hub.load(
        repo_or_dir=local_repo_dir,
        source='local',
        model='silero_tts',
        language='es',
        speaker='v3_es'
    )
except Exception as e:
    print(f"Error cargando desde local, intentando remoto con trust_repo=True: {e}")
    model, example_text = torch.hub.load(
        repo_or_dir='snakers4/silero-tts',
        model='silero_tts',
        language='es',
        speaker='v3_es',
        trust_repo=True
    )
model.to(device)

def normalize_text(text):
    """Preprocesar texto para que Silero TTS lo lea correctamente."""
    import re
    # Convertir números a palabras en español
    try:
        from num2words import num2words
        text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group()), lang='es'), text)
    except ImportError:
        pass
    # Reemplazar signos de puntuación problemáticos
    text = text.replace('¿', '').replace('¡', '')
    text = text.replace('%', ' por ciento')
    text = text.replace('°C', ' grados')
    text = text.replace('km/h', ' kilómetros por hora')
    text = re.sub(r'[^\w\s\.,;áéíóúüñÁÉÍÓÚÜÑ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text', '')
    if not text:
        return {"error": "Texto vacío"}, 400
    
    try:
        # Normalizar texto: números → palabras, eliminar signos problemáticos
        text = normalize_text(text)
        # Truncar para evitar crasheos del modelo
        text = text[:300] if len(text) > 300 else text
        
        # Generar Audio
        audio = model.apply_tts(
            text=text,
            speaker='es_0',
            sample_rate=48000
        )
        
        # Guardar en temp usando scipy (no requiere TorchCodec)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        # Convertir tensor a numpy y guardar como WAV
        audio_numpy = audio.cpu().numpy()
        # Normalizar a int16
        audio_int16 = (audio_numpy * 32767).astype(np.int16)
        wavfile.write(temp_file.name, 48000, audio_int16)
        
        return send_file(temp_file.name, mimetype='audio/wav')
    except Exception as e:
        print(f"Error generando TTS: {e}")
        return {"error": str(e)}, 500

if __name__ == '__main__':
    print("[Voice Service] Servicio de TTS levantado en puerto 5002")
    app.run(host='0.0.0.0', port=5002)
