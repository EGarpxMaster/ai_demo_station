# setup.ps1 - Windows Setup Script for AI Demo Station
# Ejecutar como Administrador

Write-Host "=== Iniciando validación y configuración de AI Demo Station para Windows ===" -ForegroundColor Cyan

# 1. Verificar WSL2
Write-Host "`n1. Verificando WSL2..." -ForegroundColor Cyan
$wsl_status = wsl --status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL2 no parece estar instalado o configurado correctamente. Instalando..." -ForegroundColor Yellow
    wsl --install
    Write-Host "Instalación de WSL solicitada. Es posible que requiera un reinicio." -ForegroundColor Magenta
} else {
    Write-Host "WSL2 parece estar presente." -ForegroundColor Green
}

# 2. Verificar Docker Desktop
Write-Host "`n2. Verificando Docker Desktop..." -ForegroundColor Cyan
try {
    docker --version | Out-Null
    Write-Host "Docker CLI encontrado." -ForegroundColor Green
    
    $docker_info = docker info 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "El motor de Docker no está corriendo. Por favor, inicia Docker Desktop." -ForegroundColor Red
        # Podríamos intentar iniciar Docker Desktop
        # Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    } else {
        Write-Host "Docker Engine corriendo correctamente." -ForegroundColor Green
    }
} catch {
    Write-Host "Docker no está instalado en el PATH. Instala Docker Desktop y actívalo en WSL2." -ForegroundColor Red
    Write-Host "URL: https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor Yellow
}

# 3. Verificar soporte de GPU (Nvidia)
Write-Host "`n3. Verificando soporte de GPU (NVIDIA)..." -ForegroundColor Cyan
try {
    nvidia-smi | Out-Null
    Write-Host "Tarjeta gráfica NVIDIA detectada. Docker en WSL2 utilizará soporte GPU nativo (RTX 4060)." -ForegroundColor Green
} catch {
    Write-Host "Comando 'nvidia-smi' no encontrado en el PATH de Windows." -ForegroundColor Red
    Write-Host "Asegúrate de tener los últimos drivers NVIDIA instalados." -ForegroundColor Yellow
}

# 4. Crear estructura de modelos locales y descargar base
Write-Host "`n4. Preparando carpetas de modelos (Volume Binds) y base..." -ForegroundColor Cyan
$models_dirs = @("models\vision", "models\voice", "models\llm")
foreach ($dir in $models_dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "Creado: $dir" -ForegroundColor Green
    }
}

$yolo_path = "models\vision\yolo_custom.pt"
if (-not (Test-Path $yolo_path)) {
    Write-Host "Descargando modelo YOLO base (yolov8n.pt) como fallback..." -ForegroundColor Cyan
    curl.exe -L -o $yolo_path https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
    Write-Host "Modelo YOLO base descargado." -ForegroundColor Green
} else {
    Write-Host "Modelo YOLO ya existe localmente." -ForegroundColor Green
}

Write-Host "`n=== Validación de Host (Windows) completada ===" -ForegroundColor Green
Write-Host "Ejecuta: 'docker compose up -d' o 'docker-compose up -d' para levantar el sistema." -ForegroundColor Cyan
Write-Host "Si hay problemas con la cámara web, asegúrate de que Docker for Windows / WSL tenga permisos de dispositivo." -ForegroundColor Yellow
