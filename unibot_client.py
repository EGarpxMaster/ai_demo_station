#!/usr/bin/env python3
"""
Unibot Native Client — Estación IA de Demostración
Uso: python3 unibot_client.py [--model /ruta/a/best.pt]
"""
import os
import sys

# cv2 sobreescribe QT_QPA_PLATFORM_PLUGIN_PATH con sus plugins incompatibles.
# Debemos corregirlo ANTES de importar PyQt5.
import cv2 as _cv2_preload  # noqa: F401 — importar solo para que setee la variable

# Localizar el directorio de plugins de PyQt5 y sobrescribir la variable
import importlib.util as _iutil
_pyqt5_spec = _iutil.find_spec("PyQt5")
if _pyqt5_spec and _pyqt5_spec.origin:
    _pyqt5_dir = os.path.dirname(_pyqt5_spec.origin)
    _pyqt5_plugins = os.path.join(_pyqt5_dir, "Qt5", "plugins")
    if os.path.isdir(_pyqt5_plugins):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(_pyqt5_plugins, "platforms")

import cv2
import argparse
import requests
import threading
import subprocess
import time
import tempfile
import wave
import numpy as np
import glob
from collections import deque

# === Suprimir ALSA / JACK stderr spam de forma limpia ===
# El método anterior con ctypes causaba "TypeError: cannot build parameter"
# Usamos redirección de stderr a /dev/null durante las partes críticas
def suppress_stderr():
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    save_fds = [os.dup(1), os.dup(2)]
    os.dup2(null_fds[0], 1)
    os.dup2(null_fds[1], 2)
    yield
    os.dup2(save_fds[0], 1)
    os.dup2(save_fds[1], 2)
    for fd in null_fds + save_fds:
        os.close(fd)

import contextlib
@contextlib.contextmanager
def ignore_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

import pyaudio
from faster_whisper import WhisperModel

# PyQt5 — ya con el plugin path correcto
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QSplitter, QFrame, QSizePolicy,
    QComboBox, QPushButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

# ============================================================
# CONFIGURACIÓN
# ============================================================
BRAIN_URL  = "http://localhost:11434"
VOICE_URL  = "http://localhost:5002"
WAKE_WORDS = [
    "unibot", "unibot", "univot", "unibod", "unibó", "univó", 
    "onibot", "onibot", "unigot", "univoc", "onibote", "unibote",
    "onibolt", "univolt", "puñivo", "omigo", "ohniver", "onibox"
]
SAMPLE_RATE  = 16000
CHUNK        = 1024
SILENCE_SECS = 1.5
AMBIENT_MULT = 3.5

# Estado global compartido entre hilos
ui_state = {
    "listening": True,
    "transcription": "",
    "response": "",
    "status": "Calibrando...",
    "mic_active": False,
    "last_activity": time.time(),
    "fps": 0.0,
    "camera_error": False,
    "calibrating": False,
    "mic_index": None,
    "available_mics": {}
}

# ============================================================
# DARK STYLESHEET
# ============================================================
QSS_STYLE = """
QMainWindow, QWidget {
    background-color: #0a0a14;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Ubuntu', sans-serif;
}

QSplitter::handle {
    background-color: #1c1c2e;
    width: 2px;
}

/* ---- Sidebar ---- */
#sidebar {
    background-color: #0d0d1a;
    border-left: 1px solid #1e1e3a;
}

/* ---- Logo ---- */
#logo {
    font-size: 48px;
    font-weight: 800;
    letter-spacing: 8px;
    color: #00dc82;
    padding: 24px 0 6px 0;
}

#logo_sub {
    font-size: 16px;
    letter-spacing: 3px;
    color: #3a5a4a;
    padding-bottom: 20px;
}

/* ---- Divider ---- */
#divider {
    background-color: #1e1e3a;
    max-height: 1px;
    min-height: 1px;
    margin: 0 20px;
}

/* ---- Stat cards ---- */
#stat_card {
    background-color: #12122a;
    border: 1px solid #1e1e3a;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 14px;
}

#stat_label {
    font-size: 18px;
    letter-spacing: 2px;
    color: #3a4a6a;
    text-transform: uppercase;
}

#stat_value {
    font-size: 26px;
    font-weight: 700;
    color: #e0e0e0;
}

#stat_value_green {
    font-size: 26px;
    font-weight: 700;
    color: #00dc82;
}

#stat_value_cyan {
    font-size: 26px;
    font-weight: 700;
    color: #50c8ff;
}

/* ---- Section titles ---- */
#section_title {
    font-size: 24px;
    letter-spacing: 3px;
    color: #3a4a6a;
    padding: 18px 20px 6px 20px;
    font-weight: bold;
}

/* ---- Text areas ---- */
#text_box {
    background-color: #0f0f22;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 0 20px;
    font-size: 18px;
    color: #b0b8c8;
    line-height: 1.6;
}

#text_box_response {
    background-color: #061a22;
    border: 1px solid #1a3040;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 0 20px;
    font-size: 18px;
    color: #50c8ff;
    line-height: 1.6;
}

/* ---- Dot indicator ---- */
#dot_green { color: #00dc82; font-size: 14px; }
#dot_red   { color: #ff4466; font-size: 14px; }

/* ---- Bottom bar ---- */
#bottom_bar {
    background-color: #080812;
    border-top: 1px solid #1e1e3a;
    padding: 8px 14px;
}

#bottom_text {
    font-size: 14px;
    color: #2a3a4a;
    letter-spacing: 1px;
}

QComboBox {
    background-color: #12122a;
    border: 1px solid #1e1e3a;
    border-radius: 8px;
    padding: 8px 15px;
    color: #e0e0e0;
    font-size: 28px;
}

QComboBox QAbstractItemView {
    background-color: #12122a;
    color: #e0e0e0;
    selection-background-color: #1a3040;
    font-size: 26px;
}

QPushButton#btn_calibrate {
    background-color: #1a1a3a;
    border: 1px solid #3a3a5a;
    border-radius: 8px;
    padding: 10px;
    color: #50c8ff;
    font-weight: bold;
    font-size: 16px;
    margin: 4px 14px;
}

QPushButton#btn_calibrate:hover {
    background-color: #2a2a4a;
    border: 1px solid #50c8ff;
}

QComboBox::drop-down {
    border: 0px;
}

QComboBox QAbstractItemView {
    background-color: #0d0d1a;
    border: 1px solid #1e1e3a;
    selection-background-color: #00dc82;
}
"""

# ============================================================
# AUDIO: TTS — Síntesis de voz
# ============================================================
def speak_text(text: str):
    if not text:
        return
    try:
        ui_state["status"] = "🔊  Hablando..."
        resp = requests.post(f"{VOICE_URL}/synthesize", json={"text": text[:300]}, timeout=15)
        resp.raise_for_status()
        audio_path = "/tmp/unibot_reply.wav"
        with open(audio_path, "wb") as f:
            f.write(resp.content)
        subprocess.run(["aplay", "-q", audio_path])
    except Exception as e:
        print(f"[TTS Error] {e}")
    finally:
        ui_state["status"] = "Escuchando... Di 'Unibot'"
        ui_state["mic_active"] = True

# ============================================================
# LLM: Consulta a Ollama
# ============================================================
def ask_ollama(prompt: str):
    ui_state["status"] = "Pensando..."
    ui_state["mic_active"] = False
    system_prompt = ("Eres Unibot, asistente IA de demostración. "
                     "Responde siempre en español, de forma casual y MUY breve "
                     "(máximo 2 oraciones cortas). Sin listas ni asteriscos.")
    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "system": system_prompt,
        "stream": False
    }
    try:
        resp = requests.post(f"{BRAIN_URL}/api/generate", json=payload, timeout=30)
        reply = resp.json().get("response", "").strip()
        ui_state["response"] = reply
        ui_state["last_activity"] = time.time()
        print(f"[Unibot] → {reply}")
        threading.Thread(target=speak_text, args=(reply,), daemon=True).start()
    except Exception as e:
        print(f"[Ollama Error] {e}")
def rms(data: bytes) -> int:
    if not data: return 0
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return int(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0

def clean_for_match(text: str) -> str:
    """Normaliza texto para comparación robusta (sin espacios, tildes ni puntuación)."""
    if not text: return ""
    # Quitar exclamaciones, interrogaciones, puntos, comas
    for c in "¡!¿?.,:;\"'":
        text = text.replace(c, "")
    # Quitar tildes comunes
    replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Quitar espacios y pasar a minúsculas
    return text.lower().replace(" ", "").strip()

def voice_thread(stt_model):
    while True: # Bucle externo para permitir reinicio de stream
        target_idx = ui_state.get("mic_index")
        
        try:
            stream = None
            pa = None
            with ignore_stderr():
                pa = pyaudio.PyAudio()
                stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                                 input=True, input_device_index=target_idx,
                                 frames_per_buffer=CHUNK)

            print(f"[Voz] Stream abierto con dispositivo ID {target_idx} ✓")
            ui_state["status"] = "Escuchando... Di 'Unibot'"
            ui_state["mic_active"] = True

            max_silent = int(SAMPLE_RATE / CHUNK * SILENCE_SECS)
            max_record = int(SAMPLE_RATE / CHUNK * 8)
            
            # Calibración rápida inicial
            calibration = [rms(stream.read(CHUNK, exception_on_overflow=False))
                           for _ in range(int(SAMPLE_RATE / CHUNK * 1))]
            avg_rms = np.mean(calibration) if calibration else 200
            threshold = int(avg_rms * AMBIENT_MULT)

            while ui_state.get("mic_index") == target_idx:
                pre_buffer: deque = deque(maxlen=int(SAMPLE_RATE / CHUNK * 0.3))
                frames = []
                recording = False
                silent_chunks = 0
                total = 0

                while ui_state.get("mic_index") == target_idx:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    level = rms(data)
                    pre_buffer.append(data)

                    if level > threshold:
                        if not recording:
                            frames = list(pre_buffer)
                            total = len(frames)
                        recording = True
                        silent_chunks = 0
                        frames.append(data)
                        total += 1
                        if total >= max_record: break
                    elif recording:
                        frames.append(data)
                        total += 1
                        silent_chunks += 1
                        if silent_chunks > max_silent: break
                
                if not frames: continue

                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(b"".join(frames))

                try:
                    segs, _ = stt_model.transcribe(tmp.name, beam_size=3, language="es")
                    text = " ".join(s.text for s in segs).strip().lower()
                    if text:
                        print(f"[STT] Escuché: '{text}'")
                        ui_state["transcription"] = text
                        
                        clean_text = clean_for_match(text)
                        match_found = any(clean_for_match(w) in clean_text for w in WAKE_WORDS)
                        
                        if match_found:
                            if not ui_state.get("calibrating", False):
                                print(f"[¡UNIBOT!] → '{text}'")
                                ui_state["response"] = ""
                                threading.Thread(target=ask_ollama, args=(text,), daemon=True).start()
                            else:
                                print(f"[Calibración] Identificada variación: {text}")
                except Exception: pass
                finally: os.unlink(tmp.name)

        except Exception as e:
            print(f"[Voz Error] No se pudo abrir mic {target_idx}: {e}")
            ui_state["status"] = "Error de Micrófono"
            time.sleep(2)
        finally:
            if stream:
                try: stream.stop_stream(); stream.close()
                except: pass
            if pa:
                try: pa.terminate()
                except: pass
            print(f"[Voz] Recursos liberados para mic {target_idx}")

# ============================================================
# VIDEO THREAD (QThread)
# ============================================================
class VideoThread(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self, cap, yolo_model):
        super().__init__()
        self.cap = cap
        self.yolo = yolo_model
        self._running = True
        self._prev_time = time.time()
        self._new_cap = None
        self._lock = threading.Lock() # Bloqueo para evitar segfaults en read / release

    def change_camera(self, working_index: int):
        """Abre el dispositivo usando el índice verificado que funcionó al inicio."""
        if working_index < 0: return False
        try:
            print(f"[Cámara] Intentando abrir working_index {working_index}...")
            temp_cap = cv2.VideoCapture(working_index)
            if not temp_cap.isOpened():
                print(f"[Cámara] Error: No se pudo reabrir el índice {working_index}")
                return False
            
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            with self._lock:
                self._new_cap = temp_cap
                ui_state["camera_error"] = False
            return True
        except Exception as e:
            print(f"[Cámara] Error en switch: {e}")
        return False

    def _no_camera_frame(self) -> QImage:
        """Frame de error con mensaje dinámico."""
        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
        msg = "Error de Hardware o Dispositivo Ocupado"
        cv2.putText(placeholder, msg, (200, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 255), 2, cv2.LINE_AA)
        cv2.putText(placeholder, "Intenta seleccionar otra fuente", (380, 410),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 150), 1, cv2.LINE_AA)
        h, w, ch = placeholder.shape
        return QImage(placeholder.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)

    def run(self):
        while self._running:
            # Transferencia segura de nuevo dispositivo
            with self._lock:
                if self._new_cap is not None:
                    if self.cap:
                        self.cap.release()
                    self.cap = self._new_cap
                    self._new_cap = None

            # Lectura segura
            with self._lock:
                if not self._running: break
                if self.cap is None or not self.cap.isOpened():
                    self.frame_ready.emit(self._no_camera_frame())
                    self.msleep(100)
                    continue
                ret, frame = self.cap.read()

            if not ret or frame is None:
                self.msleep(30)
                continue

            frame = cv2.flip(frame, 1)

            # YOLO inference optimizada
            results = self.yolo(frame, verbose=False, imgsz=320, conf=0.25)
            # Dibujar boxes y labels directamente en el frame
            frame = results[0].plot()

            now = time.time()
            fps = 1.0 / (now - self._prev_time + 1e-9)
            self._prev_time = now
            ui_state["fps"] = fps

            h, w, ch = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_ready.emit(qt_img.copy())

    def stop(self):
        self._running = False
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            if self._new_cap:
                self._new_cap.release()
                self._new_cap = None
        self.wait()


# ============================================================
# SIDEBAR HELPERS
# ============================================================
def make_divider() -> QWidget:
    line = QWidget()
    line.setObjectName("divider")
    line.setFixedHeight(1)
    line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return line


def make_stat_card(label_text: str, value_object_name: str = "stat_value") -> tuple:
    """Devuelve (card_widget, value_label) para poder actualizar el valor."""
    card = QFrame()
    card.setObjectName("stat_card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)

    lbl = QLabel(label_text)
    lbl.setObjectName("stat_label")

    val = QLabel("—")
    val.setObjectName(value_object_name)

    layout.addWidget(lbl)
    layout.addWidget(val)
    return card, val


def make_section(title: str, text_object_name: str = "text_box") -> tuple:
    """Devuelve (container_widget, text_label)."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    title_lbl = QLabel(title.upper())
    title_lbl.setObjectName("section_title")

    text_lbl = QLabel("—")
    text_lbl.setObjectName(text_object_name)
    text_lbl.setWordWrap(True)
    text_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    text_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    text_lbl.setMinimumHeight(80) # Un poco más de espacio relativo

    layout.addWidget(title_lbl)
    layout.addWidget(text_lbl)
    return container, text_lbl


# ============================================================
# MAIN WINDOW
# ============================================================
class UnibotWindow(QMainWindow):
    def __init__(self, cap, yolo_model, current_index=0, available_cams=None):
        super().__init__()
        self.setWindowTitle("Unibot IA — Estacion de Demostracion")
        self.setStyleSheet(QSS_STYLE)
        self.showMaximized() # Iniciar maximizado para que sea responsivo

        self.current_index = current_index
        self.available_cams = available_cams or [current_index]
        # ---- Central widget + splitter ----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        main_layout.addWidget(splitter)

        # ---- Video panel ----
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumWidth(640)
        self.video_label.setStyleSheet("background-color: #000010;")
        splitter.addWidget(self.video_label)

        # ---- Sidebar ----
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 20, 15, 20) # Más aire
        sidebar_layout.setSpacing(10)
        splitter.addWidget(sidebar)
        
        # Ajustar proporciones del splitter (Video: 75%, Sidebar: 25%)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # -- Logo --
        logo_container = QWidget()
        logo_layout = QVBoxLayout(logo_container)
        logo_layout.setAlignment(Qt.AlignHCenter)

        logo = QLabel("UNIBOT")
        logo.setObjectName("logo")
        logo.setAlignment(Qt.AlignCenter)

        logo_sub = QLabel("ESTACIÓN IA · DEMO")
        logo_sub.setObjectName("logo_sub")
        logo_sub.setAlignment(Qt.AlignCenter)

        logo_layout.addWidget(logo)
        logo_layout.addWidget(logo_sub)
        sidebar_layout.addWidget(logo_container)
        sidebar_layout.addWidget(make_divider())

        # -- Stat cards --
        mic_card, self.mic_val = make_stat_card("MICRÓFONO", "stat_value_green")
        fps_card,  self.fps_val  = make_stat_card("FPS",       "stat_value")
        status_card, self.status_val = make_stat_card("ESTADO", "stat_value_cyan")

        sidebar_layout.addWidget(mic_card)
        sidebar_layout.addWidget(fps_card)
        sidebar_layout.addWidget(status_card)
        sidebar_layout.addWidget(make_divider())

        # -- Camera Selector --
        cam_title = QLabel("CÁMARA")
        cam_title.setObjectName("section_title")
        sidebar_layout.addWidget(cam_title)

        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumHeight(40) # Tamaño relativo mejorado
        if isinstance(self.available_cams, dict):
            for dev_num in sorted(self.available_cams.keys()):
                info = self.available_cams[dev_num]
                label = f"Cámara {dev_num + 1}: {info['name']}"
                self.cam_combo.addItem(label, info["working_index"])
        else:
            self.cam_combo.addItem("Sin cámaras", -1)
        
        # Seleccionar la actual en el combo
        cam_idx = self.cam_combo.findData(self.current_index)
        if cam_idx >= 0: self.cam_combo.setCurrentIndex(cam_idx)

        self.cam_combo.currentIndexChanged.connect(self.on_camera_changed)
        sidebar_layout.addWidget(self.cam_combo)
        
        sidebar_layout.addWidget(make_divider())

        # -- Microphone Selector --
        mic_title = QLabel("MICRÓFONO")
        mic_title.setObjectName("section_title")
        sidebar_layout.addWidget(mic_title)

        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumHeight(40)
        mics = ui_state.get("available_mics", {})
        if mics:
            for m_id in sorted(mics.keys()):
                self.mic_combo.addItem(f"Entrada {m_id}: {mics[m_id]}", m_id)
        else:
            self.mic_combo.addItem("Sin micrófonos", None)
        
        # Seleccionar el actual (por defecto el que se abrió al inicio)
        cur_mic = ui_state.get("mic_index")
        f_mic_idx = self.mic_combo.findData(cur_mic)
        if f_mic_idx >= 0: self.mic_combo.setCurrentIndex(f_mic_idx)

        self.mic_combo.currentIndexChanged.connect(self.on_mic_changed)
        sidebar_layout.addWidget(self.mic_combo)

        sidebar_layout.addWidget(make_divider())

        # -- Calibration Button --
        self.btn_calibrate = QPushButton("Calibrar Voz (Unibot)")
        self.btn_calibrate.setObjectName("btn_calibrate")
        self.btn_calibrate.clicked.connect(self.start_calibration)
        sidebar_layout.addWidget(self.btn_calibrate)

        sidebar_layout.addWidget(make_divider())
        # -- Transcription section --
        trans_widget, self.trans_lbl = make_section("Transcripción", "text_box")
        sidebar_layout.addWidget(trans_widget)

        # -- Response section --
        resp_widget, self.resp_lbl = make_section("Respuesta", "text_box_response")
        sidebar_layout.addWidget(resp_widget)

        sidebar_layout.addStretch()

        # -- Bottom bar --
        bottom_bar = QWidget()
        bottom_bar.setObjectName("bottom_bar")
        bb_layout = QHBoxLayout(bottom_bar)
        bb_layout.setContentsMargins(14, 0, 14, 0)

        self.bottom_lbl = QLabel("UNIBOT v2.0  ·  Q para salir")
        self.bottom_lbl.setObjectName("bottom_text")
        bb_layout.addWidget(self.bottom_lbl)

        sidebar_layout.addWidget(bottom_bar)

        splitter.setSizes([900, 320])

        # ---- Video thread ----
        self.video_thread = VideoThread(cap, yolo_model)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.start()

        # ---- Refresh timer (sidebar) ----
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_sidebar)
        self.refresh_timer.start(80)  # ~12 Hz para la sidebar

    # ----------------------------------------------------------
    def on_camera_changed(self, index_in_combo):
        working_idx = self.cam_combo.itemData(index_in_combo)
        if working_idx is None: return
        ui_state["status"] = f"Cambiando cámara..."
        threading.Thread(target=self.video_thread.change_camera, args=(working_idx,), daemon=True).start()

    def update_frame(self, qt_img: QImage):
        if qt_img.isNull() or not self.isVisible():
            return
        try:
            pix = QPixmap.fromImage(qt_img)
            w = self.video_label.width()
            h = self.video_label.height()
            if w < 10 or h < 10: return
            
            scaled = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled)
        except:
            pass

    def refresh_sidebar(self):
        try:
            # Micrófono
            if ui_state.get("mic_active", False):
                self.mic_val.setText("● ACTIVO")
                self.mic_val.setObjectName("stat_value_green")
            else:
                self.mic_val.setText("○ PROCESANDO")
                self.mic_val.setObjectName("stat_value")
            self.mic_val.setStyleSheet("") 

            # FPS
            fps = ui_state.get("fps", 0)
            self.fps_val.setText(f"{fps:.0f} fps" if fps is not None else "0 fps")

            # Estado
            status = ui_state.get("status", "Iniciando...")
            self.status_val.setText(str(status))

            # Transcripción
            trans = ui_state.get("transcription", "")
            self.trans_lbl.setText(str(trans)[-200:] if trans else "—")

            # Respuesta
            resp = ui_state.get("response", "")
            self.resp_lbl.setText(str(resp)[-300:] if resp else "—")
        except Exception:
            pass

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()

    def on_mic_changed(self, index):
        mic_id = self.mic_combo.itemData(index)
        if mic_id is not None:
            print(f"[Voz] Cambiando micrófono a ID {mic_id}...")
            ui_state["mic_index"] = mic_id
            ui_state["status"] = "Cambiando micrófono..."

    def start_calibration(self):
        if ui_state["calibrating"]: return
        threading.Thread(target=self._calibration_process, daemon=True).start()

    def _calibration_process(self):
        ui_state["calibrating"] = True
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.setText("Escuchando...")
        
        samples = []
        for i in range(3):
            ui_state["status"] = f"Dí 'Unibot' (Muestra {i+1}/3)"
            time.sleep(1.0) # Pausa pequeña
            # La lógica de grabación ya reside en voice_thread, 
            # simplemente activaremos un flag para capturar la siguiente transcripción.
            ui_state["mic_active"] = True
            
            # Esperar a que voice_thread procese algo
            start_t = time.time()
            last_trans = ui_state["transcription"]
            detected = False
            while time.time() - start_t < 4.0:
                if ui_state["transcription"] != last_trans:
                    new_text = ui_state["transcription"].lower().strip()
                    if new_text:
                        # Guardar la versión normalizada para comparación futura
                        clean_v = clean_for_match(new_text)
                        samples.append(new_text)
                        detected = True
                        break
                time.sleep(0.1)
            
            if not detected:
                ui_state["status"] = "No se escuchó nada..."
                time.sleep(1)
        
        # Procesar muestras
        unique_samples = list(set(samples))
        for s in unique_samples:
            if s not in WAKE_WORDS:
                WAKE_WORDS.append(s)
        
        ui_state["status"] = "Calibración finalizada ✓"
        self.btn_calibrate.setText("Voz Calibrada ✓")
        time.sleep(2)
        self.btn_calibrate.setEnabled(True)
        self.btn_calibrate.setText("Calibrar Voz (Unibot)")
        ui_state["calibrating"] = False
        print(f"[Calibración] Nuevas variaciones: {unique_samples}")
        print(f"[Calibración] Lista total: {WAKE_WORDS}")

    def closeEvent(self, event):
        try:
            if hasattr(self, 'refresh_timer'):
                self.refresh_timer.stop()
            if hasattr(self, 'video_thread'):
                self.video_thread.stop()
        except:
            pass
        event.accept()


# ============================================================
# MAIN
# ============================================================
def main():
    # Silenciar logs de OpenCV/FFMPEG
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    os.environ["AV_LOG_FORCE_NOCOLOUR"] = "1"
    
    parser = argparse.ArgumentParser(description="Unibot Native Client")
    parser.add_argument("--model", type=str,
                        default="yolov8n.pt",
                        help="Ruta al modelo YOLO (base o entrenado)")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"[Aviso] Modelo '{model_path}' no encontrado. Usando yolov8n.pt base.")
        model_path = "yolov8n.pt"

    # Cargar Whisper
    print("[Unibot] Cargando Whisper (tiny)...")
    stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("[Unibot] Whisper listo ✓")

    # Cargar YOLO
    print(f"[Unibot] Cargando modelo YOLO: {os.path.basename(model_path)}")
    from ultralytics import YOLO
    yolo = YOLO(model_path)

    # Qt App — crear ANTES de cualquier otra verificación de hardware
    app = QApplication(sys.argv)
    app.setApplicationName("Unibot IA")

    # Iniciar hilo de voz
    threading.Thread(target=voice_thread, args=(stt_model,), daemon=True).start()

    # Escaneo exhaustivo usando sysfs y verificación de índices
    print("[Unibot] Escaneando hardware de visión...")
    # available_cams[id_real_v4l] = {"name": "Logitech...", "working_index": opencv_idx}
    available_cams = {} 
    
    sys_paths = sorted(glob.glob("/sys/class/video4linux/video*"))
    for d_path in sys_paths:
        try:
            dev_num = int(os.path.basename(d_path).replace("video", ""))
            with open(os.path.join(d_path, "name"), "r") as f:
                name = f.read().strip()
            
            # Probar el índice directo e índice + 1
            working_idx = -1
            with ignore_stderr():
                for probe_idx in [dev_num, dev_num + 1]:
                    test = cv2.VideoCapture(probe_idx)
                    if test.isOpened():
                        # Verificar que realmente da frames (opcional pero seguro)
                        ret, _ = test.read()
                        if ret:
                            working_idx = probe_idx
                            test.release()
                            break
                        test.release()
            
            if working_idx != -1:
                available_cams[dev_num] = {"name": name, "working_index": working_idx}
        except:
            continue
    
    # Cámara por defecto: Priorizar Logitech o índice mayor
    cap = None
    current_working_idx = -1
    default_dev_num = -1

    if available_cams:
        # Buscar "Logitech" o similar para prioridad
        priority_devs = [num for num, info in available_cams.items() if "Logitech" in info["name"]]
        if priority_devs:
            default_dev_num = priority_devs[0]
        else:
            default_dev_num = max(available_cams.keys())
        
        info = available_cams[default_dev_num]
        current_working_idx = info["working_index"]
        
        with ignore_stderr():
            test = cv2.VideoCapture(current_working_idx)
            if test.isOpened():
                test.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                test.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap = test
                print(f"[Visión] Iniciando con '{info['name']}' (ID {default_dev_num}, index {current_working_idx}) ✓")

    if cap is None:
        print("[Aviso] No se pudo inicializar ninguna cámara.")
        ui_state["status"] = "Sin cámara detectada"

    # Escaneo de Micrófonos ANTES de crear la ventana
    print("[Unibot] Escaneando hardware de audio...")
    available_mics = {}
    with ignore_stderr():
        pa_probe = pyaudio.PyAudio()
        for i in range(pa_probe.get_device_count()):
            dev_info = pa_probe.get_device_info_by_index(i)
            if dev_info.get("maxInputChannels") > 0:
                name = dev_info.get("name")
                available_mics[i] = name
                print(f"  [Audio] Mic detectado: ID {i} -> {name}")
        pa_probe.terminate()
    
    ui_state["available_mics"] = available_mics
    # Intentar pipewire o pulse como defecto si existen
    for mid, mname in available_mics.items():
        if any(x in mname.lower() for x in ["pipewire", "default", "pulse", "usb"]):
            ui_state["mic_index"] = mid
            print(f"  [Audio] Seleccionado por defecto: {mname}")
            break
    if ui_state["mic_index"] is None and available_mics:
        ui_state["mic_index"] = list(available_mics.keys())[0]

    # En el combo usaremos el dev_num + 1 como etiqueta, pero guardamos working_index
    window = UnibotWindow(cap, yolo, current_working_idx, available_cams)
    window.show()

    ret = app.exec_()
    # cap se libera dentro de VideoThread.stop() o al cerrar
    sys.exit(ret)


if __name__ == "__main__":
    main()
