# Creates the "qwen3-tts Studio" desktop shortcut (sleek icon, minimized launcher).
$appDir  = "$env:USERPROFILE\qwen3-tts_studio"
$desktop = [Environment]::GetFolderPath('Desktop')
New-Item -ItemType Directory -Force -Path $appDir | Out-Null
Copy-Item -Force "$PSScriptRoot\launcher.bat" "$appDir\launcher.bat"
Copy-Item -Force "$PSScriptRoot\..\icon.ico"  "$appDir\icon.ico"
Remove-Item -Force "$desktop\Qwen3 TTSI.lnk" -ErrorAction SilentlyContinue
$W = New-Object -ComObject WScript.Shell
$S = $W.CreateShortcut("$desktop\qwen3-tts Studio.lnk")
$S.TargetPath      = "$appDir\launcher.bat"
$S.WorkingDirectory = $appDir
$S.IconLocation     = "$appDir\icon.ico"
$S.WindowStyle      = 1
$S.Description      = "qwen3-tts Studio - American voice studio on localhost:8000"
$S.Save()
Write-Output "Shortcut created: $desktop\qwen3-tts Studio.lnk"
