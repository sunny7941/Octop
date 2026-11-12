# Registers the Octop air-gap watchdog scheduled task.
# Must run elevated (one UAC prompt). Safe to re-run — it overwrites the task.
# The watchdog honors airgap_switch.py (OFF by default): while the switch is off
# it only removes leftover rules and never blocks the network.
$ErrorActionPreference = 'Stop'

$octopHome = if ($env:OCTOP_HOME) { $env:OCTOP_HOME } else { Join-Path $env:USERPROFILE '.octop' }
$watchdog = Join-Path $octopHome 'airgap\airgap_watchdog.py'

$pythonw = @(
    (Join-Path $octopHome 'portable\runtime\pythonw.exe'),
    (Join-Path $octopHome 'venv\Scripts\pythonw.exe')
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $pythonw) { throw "pythonw not found under $octopHome (portable\runtime or venv\Scripts)" }
if (-not (Test-Path $watchdog)) { throw "watchdog not found: $watchdog" }

$action = New-ScheduledTaskAction -Execute $pythonw -Argument "`"$watchdog`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(30) `
    -RepetitionInterval (New-TimeSpan -Minutes 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -StartWhenAvailable

Register-ScheduledTask -TaskName "OctopAirGap" `
    -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
    -Description "Enforce the Octop air-gap switch: while it is ON, block all outbound traffic whenever Octop runs and restore egress when it exits or the switch turns OFF. The switch is OFF by default." `
    -Force | Out-Null

Write-Output "registered task OctopAirGap"
(Get-ScheduledTask -TaskName "OctopAirGap" | Select-Object TaskName, State | Format-Table -AutoSize | Out-String).Trim()
