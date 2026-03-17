FROM python:3.10-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Configurar el directorio de cache de PyTorch para usar el volumen montado
ENV TORCH_HOME=/app/models/torch

COPY . .

EXPOSE 5002

CMD ["python", "tts_app.py"]
