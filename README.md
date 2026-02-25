# YOLOv8 EPI Detector (Desktop)

App desktop em Python/Tkinter para rodar o modelo YOLOv8 (`best.pt`) com interface moderna, analise de conformidade de EPI e exportacao de relatorios.

LINK PARA DOWNLOAD DO INSTALADOR(1.7GB) - https://mega.nz/file/K84UHK6R#LIUA7efafdq4N7Wh3QbdhQMU5qOt7GsWdpOp1zmoOPo

## Requisitos

- Windows 10/11
- Python 3.10+ (recomendado 3.10 ou 3.11)
- Pip

## Rodar em desenvolvimento

```powershell
python -m pip install -r requirements.txt
python app.py
```

## Gerar executavel (.exe)

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

Saida esperada:

- `dist\YoloEpiDetector\YoloEpiDetector.exe`

## Gerar instalador (Setup .exe)

1. Instale o Inno Setup 6.
2. Rode:

```powershell
powershell -ExecutionPolicy Bypass -File .\build-installer.ps1
```

Saida esperada:

- `installer-output\YoloEpiDetector-Setup.exe`

## Recursos implementados

- Deteccao em imagem.
- Captura webcam otimizada:
  - clique em `Capturar webcam (3s)`
  - contagem regressiva de 3 segundos
  - captura de um unico frame para inferencia (sem stream continuo)
- Relatorio da ultima analise na interface:
  - status (`CONFORME`, `NAO CONFORME`, `SEM DETECCOES`)
  - score de conformidade
  - total de deteccoes e nao conformidades
  - lista de deteccoes com confianca
- Exportacao historica de relatorios em CSV.

## Modelo

- Caminho padrao do modelo: `yolo8n_v8\weights\best.pt`
- Para usar outro `.pt`, defina `YOLO_MODEL_PATH`.
