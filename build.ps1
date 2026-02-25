$ErrorActionPreference = "Stop"

Write-Host "[1/3] Instalando dependencias..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "[2/3] Gerando executavel..."
python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name YoloEpiDetector `
  --add-data "yolo8n_v8\weights\best.pt;yolo8n_v8\weights" `
  app.py

Write-Host "[3/3] Build finalizado."
Write-Host "Executavel: dist\YoloEpiDetector\YoloEpiDetector.exe"

