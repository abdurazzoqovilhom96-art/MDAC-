# =====================================================================
#  CEC2011 REAL-WORLD STAGE, 2026-10-31.
#
#  Eight bound-constrained CEC2011 problems (benchmarks/cec2011_official.py,
#  verified against the organisers' MATLAB code in Octave:
#  crosscheck_cec2011.log, 8/8 problems match), the same 8 algorithms as the
#  FES-parity stage, the competition's 150 000 evaluations, 51 runs.
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

Stamp "CEC2011 RW STAGE   8 algs x 8 problems x 51 runs, 150 000 FEs"
py -3.14 MDAC_EXPERIMENT.py --stage rw --cec cec2011 --workers 20 --out FINAL_rw

Stamp "REANALYSIS (all 8, then the published six)"
cmd /c "py -3.14 -W ignore::RuntimeWarning reanalyse.py --dir FINAL_rw --tag rw 2>&1"
cmd /c "py -3.14 -W ignore::RuntimeWarning reanalyse.py --dir FINAL_rw --tag rw --suffix _pub --algs MDAC jSO L-SHADE-SPACMA L-SHADE EBOwithCMAR CMA-ES 2>&1"

Stamp "FINISHED"
