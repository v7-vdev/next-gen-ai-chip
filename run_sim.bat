@echo off
set "PATH=C:\iverilog\bin;%PATH%"
echo [1/2] Compiling ai_core.v and tb_ai_core.v...
iverilog -o ai_core_sim.out -g2012 ai_core.v tb_ai_core.v
if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    exit /b %ERRORLEVEL%
)

echo [2/2] Running vvp simulation...
vvp ai_core_sim.out
