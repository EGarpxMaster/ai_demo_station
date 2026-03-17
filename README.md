# Unibot — AI Demo Station

Estación multimodal de inteligencia artificial que integra visión por computadora, procesamiento de lenguaje natural y voz en tiempo real.

## 🚀 Inicio Rápido

### 1. Requisitos Previos
- Docker y Docker Compose
- Python 3.10+
- Hardware: Cámara (Webcam) y Micrófono.

### 2. Instalación Automática
Ejecute el script de instalación según su sistema operativo para configurar servicios y dependencias:

**Linux:**
```bash
bash setup.sh
```

**Windows:**
```powershell
.\setup.ps1
```

### 3. Levantar Servicios (Backend)
Inicia los motores de IA (Visión, Cerebro/Ollama, Voz):
```bash
docker-compose up -d
```

### 4. Lanzar Cliente Nativo
El cliente nativo proporciona una interfaz moderna para interactuar con Unibot:
```bash
python unibot_client.py
```

---

## 🛠️ Características del Cliente

- **Visión en Tiempo Real**: Detección de objetos con YOLOv8 integrado en el feed de video.
- **Selector de Hardware**: Cambia dinámicamente entre múltiples cámaras y micrófonos (soporte optimizado para Logitech C920).
- **Calibración de Voz**: Botón dedicado para aprender tu pronunciación exacta de "Unibot" y mejorar la activación.
- **Sensibilidad Inteligente**: Detección de Wake Word robusta que ignora ruidos, espacios y variaciones fonéticas.
- **Interfaz 2K**: Diseño responsivo escalado para monitores de alta resolución.

## 🏗️ Arquitectura

- **Vision Service**: YOLOv8 servido vía API interna.
- **Brain Service**: Ollama (Llama 3.2 1B) para procesamiento de texto.
- **Voice Service**: Faster-Whisper para STT y TTS local.
- **Native Client**: Interfaz PyQt5 que orquesta todos los servicios.

---

## 📄 Notas de Desarrollo
- Los modelos pesados (`.pt`) y el entorno virtual (`venv/`) están excluidos del repositorio vía `.gitignore`.
- Se utiliza un sistema de escaneo dinámico para identificar índices de hardware real, evitando errores de "dispositivo ocupado".
