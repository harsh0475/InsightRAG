# PowerShell helper to run tests with virtual environment
$ErrorActionPreference = "Stop"
Write-Host "Running test suite..." -ForegroundColor Cyan
& ".\.venv\Scripts\pytest.exe" -v

