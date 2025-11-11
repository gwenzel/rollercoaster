@echo off
REM Launcher script for the drawing app
REM Uses the 'rc' conda environment which has streamlit installed

echo Starting Drawing App...
echo.
echo The app will open at: http://localhost:8502
echo Press Ctrl+C to stop the server
echo.

C:\Users\Lenovo\anaconda3\Scripts\activate.bat rc
streamlit run app_drawing.py --server.port 8502

pause
