# Runs linting and tests

$ErrorActionPreference = "Stop"

Write-Host "Running flake8..."
if (Test-Path "src") {
    flake8 src
}
else {
    Write-Host "No src directory found. Skipping flake8."
}

Write-Host "Running pytest..."
if (Test-Path "tests") {
    pytest tests
}
else {
    Write-Host "No tests directory found. Skipping pytest."
}
