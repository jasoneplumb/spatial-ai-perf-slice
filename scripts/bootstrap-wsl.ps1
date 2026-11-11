# Run from an elevated Windows PowerShell.
# Installs WSL with Ubuntu. Reboot if Windows requests it.
$ErrorActionPreference = "Stop"

Write-Host "Installing/updating WSL and Ubuntu..."
wsl --install -d Ubuntu
wsl --update

Write-Host ""
Write-Host "After Ubuntu opens, create your Linux username/password."
Write-Host "Then clone spatial-ai-perf-slice inside your Linux home directory and run:"
Write-Host "  bash ./scripts/setup-wsl.sh"
