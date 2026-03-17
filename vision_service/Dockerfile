FROM python:3.10-slim

WORKDIR /app

# Instalar librerías de sistema requeridas para OpenCV y descargas
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el punto de entrada principal
COPY . .

# Puerto para Gradio
EXPOSE 7860

# Comando por defecto
CMD ["python", "app.py"]
