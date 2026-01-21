# Runs static type checks

$ErrorActionPreference = "Stop"

Write-Host "Running mypy..."
if (Test-Path "src") {
    mypy src
}
else {
    Write-Host "No src directory found. Skipping mypy."
}
