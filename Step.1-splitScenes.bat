@ECHO OFF
for %%i in (InputVideo\*.mp4) do (  
    "venv\python.exe" "step 1.split scenes(TransNet)\inference\transnetv2.py" "%%i"
    echo "%%i.scenes.txt"
    move "%%i.scenes.txt" "Scenes\" 
    del /f /q "%%i.predictions.txt"
)
ECHO FINISH!
pause