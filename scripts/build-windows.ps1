# Build MoRa Stats into an installer on a Windows machine.
#
#     powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1
#
# Needs Python 3.11 or newer, Node 20 or newer, Rust, and the Microsoft C++
# build tools. It leaves a setup .exe in
# app\src-tauri\target\release\bundle\nsis.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$triple = "x86_64-pc-windows-msvc"

Write-Host "1/5  Installing the engine and running its tests"
Push-Location "$root\engine"
python -m pip install --upgrade pip
pip install -e ".[dev]" pyinstaller
pytest -q
if ($LASTEXITCODE -ne 0) { throw "The engine tests failed. Nothing is packaged from a red build." }

Write-Host "2/5  Freezing the engine into mora-engine.exe"
pyinstaller packaging\mora-engine.spec --distpath dist --workpath build --noconfirm
Pop-Location

Write-Host "3/5  Placing the sidecar where Tauri looks for it"
New-Item -ItemType Directory -Force -Path "$root\app\src-tauri\binaries" | Out-Null
Copy-Item "$root\engine\dist\mora-engine.exe" `
          "$root\app\src-tauri\binaries\mora-engine-$triple.exe" -Force

Write-Host "4/5  Building the frontend"
Push-Location "$root\app"
npm ci
npm run build

Write-Host "5/5  Building the installer"
npx --yes @tauri-apps/cli@^2 build
Pop-Location

$out = "$root\app\src-tauri\target\release\bundle\nsis"
Write-Host ""
Write-Host "Done. The installer is in $out"
Get-ChildItem $out -Filter *.exe | ForEach-Object { Write-Host "  $($_.Name)  $([math]::Round($_.Length/1MB,1)) MB" }
