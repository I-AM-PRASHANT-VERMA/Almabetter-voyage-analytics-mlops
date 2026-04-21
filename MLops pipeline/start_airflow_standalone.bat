@echo off
REM Airflow standalone is blocked on native Windows because the CLI expects POSIX-only modules.
echo Native Apache Airflow is not supported on normal Windows PowerShell.
REM Point the user to the supported path instead of failing silently.
echo Use WSL2 or a Linux container for the Airflow demo in this project.
echo.
REM Keep the next action right in the terminal output.
echo Recommended next step:
echo 1. Open Ubuntu or another WSL2 terminal
echo 2. Go to this project folder under /mnt/e/
echo 3. Run the Airflow commands from README.md
REM Non-zero exit code makes it obvious that this shortcut is an instruction stub, not a launcher.
exit /b 1
