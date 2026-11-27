<#
.SYNOPSIS
    Descarga e inicia el ejecutable portátil de ml_framework desde GitHub Releases
    sin requerir permisos de administrador ni instalación previa de Python.
#>

$ErrorActionPreference = "Stop"

# --- Configuración del repositorio en GitHub ---
$RepoOwner  = "TU_USUARIO"
$RepoName   = "TU_REPOSITORIO"
$AssetName  = "ml_framework_win64.zip"
$ReleaseUrl = "https://github.com/$RepoOwner/$RepoName/releases/latest/download/$AssetName"

# --- Rutas locales del usuario (no requieren elevación UAC / Administrador) ---
$InstallDir = Join-Path $env:LOCALAPPDATA "ml_framework"
$TempZip    = Join-Path $env:TEMP $AssetName

# Forzar TLS 1.2 para compatibilidad con GitHub en PowerShell 5.1+
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   INSTALADOR PORTATIL - MODULAR ML FRAMEWORK" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "[1/3] Descargando última versión desde GitHub Releases..."
Write-Host "      Origen: $ReleaseUrl" -ForegroundColor DarkGray

try {
    Invoke-WebRequest -Uri $ReleaseUrl -OutFile $TempZip -UseBasicParsing
}
catch {
    Write-Host "`n[ERROR] No se pudo descargar el paquete desde GitHub Releases." -ForegroundColor Red
    Write-Host "Verifica que el repositorio sea público y que exista un Release con '$AssetName'." -ForegroundColor Yellow
    Write-Host "Detalle: $($_.Exception.Message)" -ForegroundColor DarkGray
    exit 1
}

Write-Host "[2/3] Extrayendo archivos en $InstallDir..."
if (Test-Path $InstallDir) {
    Remove-Item -Path $InstallDir -Recurse -Force
}
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Expand-Archive -Path $TempZip -DestinationPath $InstallDir -Force
Remove-Item -Path $TempZip -Force -ErrorAction SilentlyContinue

# Localizar ml_framework.exe sin importar si el .zip contiene una subcarpeta raíz
$ExeCandidateDirect = Join-Path $InstallDir "ml_framework.exe"
$ExeCandidateNested = Join-Path $InstallDir "ml_framework\ml_framework.exe"

if (Test-Path $ExeCandidateDirect) {
    $ExePath = $ExeCandidateDirect
}
elseif (Test-Path $ExeCandidateNested) {
    $ExePath = $ExeCandidateNested
}
else {
    $Found = Get-ChildItem -Path $InstallDir -Filter "ml_framework.exe" -Recurse | Select-Object -First 1
    if ($null -eq $Found) {
        Write-Host "[ERROR] No se encontró 'ml_framework.exe' dentro del archivo descargado." -ForegroundColor Red
        exit 1
    }
    $ExePath = $Found.FullName
}

Write-Host "[3/3] Iniciando aplicación portátil...`n" -ForegroundColor Green
& $ExePath @args
