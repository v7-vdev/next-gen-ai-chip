# Helper to compile and run Icarus Verilog simulation
$env:Path = "C:\iverilog\bin;" + $env:Path

Write-Host "[1/2] Compiling ai_core.v and tb_ai_core.v with Icarus Verilog..." -ForegroundColor Cyan
iverilog -o ai_core_sim.out -g2012 ai_core.v tb_ai_core.v

if ($LASTEXITCODE -eq 0) {
    Write-Host "[2/2] Running vvp simulation engine..." -ForegroundColor Cyan
    vvp ai_core_sim.out
} else {
    Write-Host "Compilation failed with error code $LASTEXITCODE" -ForegroundColor Red
}
