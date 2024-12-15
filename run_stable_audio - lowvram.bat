@echo off

:: Change to the directory of this batch file
cd /d "%~dp0"

:: Activate the virtual environment
call venv\Scripts\activate

:: Run the Python script
python run_gradio.py --ckpt-path ".\models\stable_audio\model.safetensors" --model-config ".\models\stable_audio\model_config.json" --model-half

:: Pause to see the output
pause