@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating a project-local Python environment...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Python 3.12 was not found. Install Python 3.12 and try again.
        pause
        exit /b 1
    )
    echo Installing project dependencies...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
)

if not exist "artifacts\quality_model.joblib" (
    echo Training the demo model and generating session data...
    .venv\Scripts\python.exe -m src.train
    if errorlevel 1 (
        echo Model training failed.
        pause
        exit /b 1
    )
)

echo Starting StreamIQ. Keep this window open while using the demo.
.venv\Scripts\python.exe -m streamlit run app.py
endlocal
