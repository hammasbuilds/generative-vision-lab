@echo off
REM Start the Generative Vision Lab viewer. Takes --port (default 8843) and --no-browser.
cd /d "%~dp0"
uv run --no-project --python 3.12 --with pillow --with numpy python app\server.py %*
