@echo off
REM Build script for creating Windows executable with Nuitka
REM Run this on Windows (or in VirtualBox Windows VM)

echo ========================================
echo Building dbf_to_sqlserver.exe with Nuitka
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

echo Step 1: Checking dependencies...
python -c "import nuitka" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Nuitka is not installed
    echo Please run: pip install -r requirements.txt -r requirements-build.txt
    exit /b 1
)

echo Step 2: Cleaning previous build artifacts...
if exist "dbf_to_sqlserver.dist" rmdir /s /q dbf_to_sqlserver.dist
if exist "dbf_to_sqlserver.build" rmdir /s /q dbf_to_sqlserver.build
if exist "dbf_to_sqlserver.onefile-build" rmdir /s /q dbf_to_sqlserver.onefile-build
if exist "dbf_to_sqlserver.exe" del /f /q dbf_to_sqlserver.exe

echo Step 3: Running Nuitka compilation...
echo This may take several minutes...
echo.

python -m nuitka ^
    --standalone ^
    --onefile ^
    --no-deployment-flag=self-execution ^
    --windows-console-mode=force ^
    --enable-plugin=anti-bloat ^
    --assume-yes-for-downloads ^
    --follow-imports ^
    --include-package=sqlalchemy.dialects.mssql ^
    --include-package=dbfread ^
    --include-module=pymssql ^
    --include-module=pyodbc ^
    --nofollow-import-to=pytest ^
    --nofollow-import-to=setuptools ^
    --nofollow-import-to=distutils ^
    --nofollow-import-to=sqlalchemy.testing ^
    --output-filename=dbf_to_sqlserver.exe ^
    dbf_to_sqlserver.py

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executable location: dbf_to_sqlserver.exe
echo File size:
dir dbf_to_sqlserver.exe | findstr "dbf_to_sqlserver.exe"
echo.
echo Test the executable with:
echo   dbf_to_sqlserver.exe --help
echo.

pause
