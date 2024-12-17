@echo off

:: Change to the directory of this batch file
cd /d "%~dp0"

:: Activate the virtual environment
call venv\Scripts\activate

:: Run the Python script
python run_gradio.py --model-half

:: Pause to see the output
pause