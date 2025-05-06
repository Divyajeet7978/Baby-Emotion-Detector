@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
  set "DEL=%%a"
)
set "RED=!DEL!
set "GREEN=!DEL!
set "YELLOW=!DEL!
set "BLUE=!DEL!
set "PURPLE=!DEL!
set "CYAN=!DEL!
set "NC=!DEL!

cls
echo %PURPLE%
echo ___________             .__       .__    ___________               __  .__                ________          __                 __  .__               
echo \_   _____/____    ____ ^|__^|____  ^|  ^|   \_   _____/ _____   _____/  ^|_^|__^| ____   ____   \______ \   _____/  ^|_^| ____   _____/  ^|_^|__^| ____   ____  
echo  ^|    __^) \__  \ _/ ___\^|  \__  \ ^|  ^|    ^|    __^)_ /     \ /  _ \   __\  ^|/  _ \ /    \   ^|    ^|  \_/ __ \   __\/ __ \_/ ___\   __\  ^|/  _ \ /    \ 
echo  ^|     \   / __ \\  \___^|  ^|/ __ \^|  ^|__  ^|        \  Y Y  (  ^<^>^> )  ^| ^|  ^|(  ^<^>^> )   ^|  \  ^|    `   \  ___/^|  ^| \  ___/\  \___^|  ^| ^|  ^|(  ^<^>^> )   ^|  \
echo  \___  /  (____  /\___  ^>__^|(____  /____/ /_______  /__^|_^|  /\____/^|__^| ^|__^|\____/^|___^|  / /_______  /\___  ^>__^|  \___  ^>\___  ^>__^| ^|__^|\____/^|___^|  /
echo      \/        \/     \/        \/               \/      \/                           \/          \/     \/          \/     \/                    \/ 
echo %NC%
echo ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
echo ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
echo This system is designed to detect and classify a baby’s emotional state based on 
echo facial expressions. By utilizing Convolutional Neural Networks (CNNs), 
echo the model processes grayscale images to predict emotions, including 
echo Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
echo ----------------------------------------------------------------------------------


echo %PURPLE%
echo.🌈 Launching Emotion Detection Engine...%NC%
echo 👇 Click the link below to see the live demo, hosted on this Development/Local Server😊%NC%
echo %CYAN%✨ Tip: Face the camera directly for best results 😊%NC%
echo.
echo --------------------------------------------------------------------------------------------------------



python detection_app.py
