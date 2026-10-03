# SupplyMind — Start all services
# Run: .\start.ps1

$Root = $PSScriptRoot

Write-Host "Starting SupplyMind platform..." -ForegroundColor Cyan

# 1. Start FastAPI backend
$backendJob = Start-Job -ScriptBlock {
    param($path, $root)
    Set-Location $path
    & "$root\..\.venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
} -ArgumentList "$Root\backend", $Root

# 2. Start Next.js frontend
$frontendJob = Start-Job -ScriptBlock {
    param($path)
    Set-Location $path
    npm run dev
} -ArgumentList "$Root\frontend"

Write-Host ""
Write-Host "  Backend  → http://localhost:8000" -ForegroundColor Green
Write-Host "  Frontend → http://localhost:3000" -ForegroundColor Green
Write-Host "  API docs → http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop all services." -ForegroundColor Yellow

# Stream output from both jobs
try {
    while ($true) {
        $backendOutput  = Receive-Job $backendJob  -ErrorAction SilentlyContinue
        $frontendOutput = Receive-Job $frontendJob -ErrorAction SilentlyContinue
        if ($backendOutput)  { Write-Host "[backend]  $backendOutput"  -ForegroundColor Blue }
        if ($frontendOutput) { Write-Host "[frontend] $frontendOutput" -ForegroundColor Magenta }
        Start-Sleep -Milliseconds 500
    }
} finally {
    Stop-Job $backendJob, $frontendJob
    Remove-Job $backendJob, $frontendJob
}
