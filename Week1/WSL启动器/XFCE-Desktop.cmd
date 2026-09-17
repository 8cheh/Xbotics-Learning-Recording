@echo off
rem ===========================================================================
rem  XFCE-Desktop.cmd  --  one click: full XFCE desktop in a single window
rem
rem  Usage:  XFCE-Desktop.cmd              default 1600x900
rem          XFCE-Desktop.cmd 1920x1080    custom resolution
rem
rem  Why a nested X server:
rem    WSLg is RAIL mode - every top-level X window becomes its own Windows
rem    window. Running startxfce4 directly therefore gives loose floating
rem    windows, never one whole desktop. The launcher starts Xephyr inside
rem    Linux and puts the whole XFCE desktop inside it, so WSLg maps it to
rem    ONE window.
rem
rem  ASCII-only on purpose (cmd.exe parses .cmd as ANSI).
rem ===========================================================================

setlocal EnableExtensions
chcp 65001 >nul

set "GEOM=%~1"
if "%GEOM%"=="" set "GEOM=1600x900"

call "%~dp0Linux-GUI.cmd" desktop %GEOM%
exit /b %ERRORLEVEL%
