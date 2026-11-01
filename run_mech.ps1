# =====================================================================
#  CEC2020 MECHANICAL ENGINEERING STAGE, 2026-10-31.
#
#  Pre-registered in PROTOCOL_CEC2020_MECH.md (written before any run):
#  RC15-RC33 except RC24, 18 problems, verified against the organisers'
#  MATLAB code in Octave (crosscheck_cec2020rw.log, 18/18 match), the same 8
#  algorithms as the FES-parity stage, official MaxFEs, 51 runs.
#  Do NOT edit any .py file while this runs (rule 9b).
# =====================================================================
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
  '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000001")   # CONTINUOUS | SYSTEM_REQUIRED
Write-Output "working directory: $PWD"
$t0 = Get-Date
function Stamp($msg) {
  $el = [math]::Round(((Get-Date) - $t0).TotalHours, 2)
  Write-Output ""
  Write-Output "=================================================================="
  Write-Output " [$el h] $msg"
  Write-Output "=================================================================="
}

Stamp "CEC2020 MECH STAGE   8 algs x 18 problems x 51 runs, official MaxFEs"
py -3.14 MDAC_EXPERIMENT.py --stage mech --cec cec2020mech --workers 20 --out FINAL_mech

Stamp "REANALYSIS (all 8, then the published six)"
cmd /c "py -3.14 -W ignore::RuntimeWarning reanalyse.py --dir FINAL_mech --tag mech 2>&1"
cmd /c "py -3.14 -W ignore::RuntimeWarning reanalyse.py --dir FINAL_mech --tag mech --suffix _pub --algs MDAC jSO L-SHADE-SPACMA L-SHADE EBOwithCMAR CMA-ES 2>&1"

Stamp "FINISHED"
