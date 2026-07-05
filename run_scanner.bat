@echo off
chcp 65001 >nul
title Business Card Scanner Services

echo [1/5] Starting Ollama Service...
start "Ollama" cmd /c "ollama serve"

echo [2/5] Starting Open WebUI...
start "Open WebUI" cmd /c "open-webui serve"

echo [3/5] Starting DMR CORS Proxy (Port 12435)...
start "DMR Proxy" cmd /c "python dmr_proxy.py"

echo [4/5] Starting OpenCV Backend (Port 5050)...
start "OpenCV Backend" cmd /c "python opencv_backend.py"

echo [5/6] Starting VCF Checker Backend (Port 5051)...
start "VCF Checker" cmd /c "python vcf_checker.py"

echo [6/6] Starting HTTP Server (Port 8000)...
start "HTTP Server" cmd /c "python -m http.server 8000"

echo.
echo All services started!
echo.
echo Please open the following URLs in your browser:
echo - Scanner:     http://localhost:8000/scanner.html
echo - VCF Checker: http://localhost:8000/vcf-checker.html
echo - Open WebUI:  http://localhost:8080 (Default Port)
echo.
echo To stop the services, close the newly opened command windows.
pause