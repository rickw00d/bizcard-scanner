@echo off
chcp 65001 >nul
title Business Card Scanner Launcher

echo Starting all services in one Windows Terminal window (6 tabs)...

wt -w scanner ^
 new-tab --title "Ollama"       -d "%~dp0." cmd /k ollama serve ; ^
 new-tab --title "Open WebUI"   -d "%~dp0." cmd /k open-webui serve ; ^
 new-tab --title "DMR Proxy"    -d "%~dp0." cmd /k python dmr_proxy.py ; ^
 new-tab --title "OpenCV 5050"  -d "%~dp0." cmd /k python opencv_backend.py ; ^
 new-tab --title "VCF 5051"     -d "%~dp0." cmd /k python vcf_checker.py ; ^
 new-tab --title "HTTP 8000"    -d "%~dp0." cmd /k python -m http.server 8000

echo.
echo All services started!
echo.
echo Please open the following URLs in your browser:
echo - Scanner:     http://localhost:8000/scanner.html
echo - VCF Checker: http://localhost:8000/vcf-checker.html
echo - Open WebUI:  http://localhost:8080 (Default Port)
echo.
echo To stop the services, close the Windows Terminal window (all tabs).
pause
