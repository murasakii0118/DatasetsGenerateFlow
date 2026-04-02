@echo off
chcp 65001
set /p userinput=lora的名称:
if "%userinput%"=="" (
    set userinput=My_lora
    echo 使用默认名称:My_lora
    )
@REM for /D %%i in (precisionFrame\*) do echo %userinput% "%%i"
for /D %%i in (precisionFrame\*) do (

    "step 5.generatePrompt\inferprompt.exe" %userinput% ".\%%i"
    )
pause