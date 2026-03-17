import os
import torch
import torchaudio
from flask import Flask, request, send_file
import tempfile

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
model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-tts',
    model='silero_tts',
    language='es',
    speaker='v3_es'
)
model.to(device)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text', '')
    if not text:
        return {"error": "Texto vacío"}, 400
    
    try:
        # Generar Audio
        # Limitado por defecto por Silero a strings razonables
        # speaker es_0, es_1, es_2 son voces españolas
        audio = model.apply_tts(
            text=text,
            speaker='es_0',
            sample_rate=48000
        )
        
        # Guardar en temp
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        # Guardar tensor a wav local
        # Silero devuelve tensor de 1D, torchaudio requiere 2D [channels, time]
        torchaudio.save(temp_file.name, audio.unsqueeze(0).cpu(), 48000)
        
        return send_file(temp_file.name, mimetype='audio/wav')
    except Exception as e:
        print(f"Error generando TTS: {e}")
        return {"error": str(e)}, 500

if __name__ == '__main__':
    print("[Voice Service] Servicio de TTS levantado en puerto 5002")
    app.run(host='0.0.0.0', port=5002)
