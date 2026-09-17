@echo off
rem ===========================================================================
rem  Linux-GUI.cmd  --  WSLg launcher (Windows entry point)
rem
rem  Double-click                 -> open the GUI launcher window
rem  Linux-GUI.cmd desktop [WxH]  -> start the full XFCE desktop directly
rem  Linux-GUI.cmd stop           -> close the full desktop
rem  Linux-GUI.cmd check          -> environment diagnostics
rem  Linux-GUI.cmd fix            -> restart WSL, then start the desktop
rem  Linux-GUI.cmd thunar         -> start a single program
rem
rem  NOTE: keep this file ASCII-only. cmd.exe parses .cmd as ANSI and would
rem  corrupt UTF-8 Chinese here (that is what broke the earlier version).
rem  All the real logic lives in LinuxGUI.ps1.
rem ===========================================================================

setlocal EnableExtensions
chcp 65001 >nul
set "PS1=%~dp0LinuxGUI.ps1"

if not exist "%PS1%" (
    echo.
    echo   [ERROR] LinuxGUI.ps1 not found:
    echo           %PS1%
    echo.
    pause
    exit /b 1
)

if "%~1"=="" (
    start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%PS1%"
    exit /b 0
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "RC=%ERRORLEVEL%"
echo.
echo   exit code: %RC%
echo.
pause
exit /b %RC%
