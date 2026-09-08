# PowerShell helper to start FastAPI backend with virtual environment
$ErrorActionPreference = "Stop"
Write-Host "Starting InsightRAG Backend on http://localhost:8000 ..." -ForegroundColor Cyan
& ".\.venv\Scripts\uvicorn.exe" backend.app.main:app --reload --host 0.0.0.0 --port 8000

