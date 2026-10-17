# Waits until an HTTP endpoint answers 200, then exits 0. Exits 1 on timeout.
# launcher.bat calls this so Edge opens the moment the API is actually ready
# instead of spinning a fixed batch loop. ASCII only (PowerShell 5.1 safe).
param(
    [string]$Url = "http://localhost:8000/api/health",
    [int]$TimeoutSec = 180
)
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri $Url
        if ($r.StatusCode -eq 200) { exit 0 }
    } catch { }
    Start-Sleep -Milliseconds 750
}
exit 1
