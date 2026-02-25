$ErrorActionPreference = "Stop"

function Resolve-IsccPath {
    $candidates = @(
        "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LocalAppData\Programs\Inno Setup 6\ISCC.exe"
    )

    foreach ($path in $candidates) {
        if (Test-Path $path) {
            return $path
        }
    }

    $fromPath = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }

    return $null
}

if (-not (Test-Path ".\dist\YoloEpiDetector\YoloEpiDetector.exe")) {
    Write-Host "Build do executavel nao encontrado. Rodando build.ps1..."
    powershell -ExecutionPolicy Bypass -File .\build.ps1
}

$iscc = Resolve-IsccPath
if (-not $iscc) {
    Write-Error "Inno Setup nao encontrado. Instale o Inno Setup 6 e rode novamente este script."
}

Write-Host "Compilando instalador com Inno Setup..."
& $iscc ".\installers\YoloEpiDetector.iss"

Write-Host "Instalador gerado em: installer-output\YoloEpiDetector-Setup.exe"
