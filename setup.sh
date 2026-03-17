#!/bin/bash
# setup.sh - Ubuntu Setup Script for AI Demo Station

# Colores para salida
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== Iniciando instalación de AI Demo Station para Ubuntu ===${NC}"

# Verificar e instalar dependencias base
echo -e "\n${BLUE}1. Verificando dependencias base (curl, wget)...${NC}"
sudo apt-get update
sudo apt-get install -y curl wget software-properties-common

# 1. Instalar Docker si no existe
echo -e "\n${BLUE}2. Verificando Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker no está instalado. Instalando Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}Docker instalado. Es posible que requieras reiniciar tu sesión al finalizar para aplicar grupos.${NC}"
else
    echo -e "${GREEN}Docker ya está instalado.${NC}"
fi

# 2. Instalar Docker Compose (si es muy antiguo o no existe, usando plugin)
echo -e "\n${BLUE}3. Verificando Docker Compose...${NC}"
if ! docker compose version &> /dev/null; then
    echo -e "${RED}Docker Compose plugin no encontrado. Instalando...${NC}"
    sudo apt-get install -y docker-compose-plugin
else
    echo -e "${GREEN}Docker Compose plugin ya instalado.${NC}"
fi

# 3. NVIDIA Container Toolkit
echo -e "\n${BLUE}4. Verificando NVIDIA Container Toolkit...${NC}"
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo -e "${RED}NVIDIA Container Toolkit no detectado. Instalando...${NC}"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo -e "${GREEN}NVIDIA Container Toolkit instalado.${NC}"
else
    echo -e "${GREEN}NVIDIA Container Toolkit ya está instalado.${NC}"
fi

# 4. Descargar modelos base locales
echo -e "\n${BLUE}5. Preparando carpetas locales y descargando modelos base...${NC}"
mkdir -p models/vision models/voice models/llm
echo -e "${GREEN}Carpetas de modelos creadas.${NC}"

if [ ! -f "models/vision/yolo_custom.pt" ]; then
    echo -e "${BLUE}Descargando modelo YOLO base (yolov8n.pt) como fallback...${NC}"
    curl -L -o models/vision/yolo_custom.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
    echo -e "${GREEN}Modelo YOLO base descargado.${NC}"
else
    echo -e "${GREEN}Modelo YOLO ya existe localmente.${NC}"
fi

echo -e "\n${GREEN}=== Instalación base (Host) completada ===${NC}"
echo -e "Puedes iniciar el sistema con: ${BLUE}docker compose up -d${NC}"
echo -e "Nota: Para acceder a /dev/video0 o cámara, asegúrate de pertenecer al grupo 'video':"
echo -e "      ${BLUE}sudo usermod -aG video \$USER${NC}"
