# =====================================================================
#  C-EXACT RE-RUN, 2026-10-31.
#
#  A compiled cross-check (crosscheck_cec2017_c.py) showed that F2, F6, F7,
#  F8, F9, F13, F14 and F20 differed from the organisers' C code. They are
#  now C-exact (30/30 match). This script re-runs ONLY their cells, in every
#  stage that used them, with the same algorithms, dims, runs and seeds:
#
#    1  FES parity, 8 algorithms, D = 10/30/50/100, 51 runs
#    2  equal-time budget for these cells, measured from stage 1 on THIS
#       machine and merged into equal_time_budget.json
#    3  equal wall-clock, same 8 algorithms and dims -- keep the machine quiet
#    4  full ablation, 25 variants, D = 10/30/50
#    5  architectural-parity control, D = 10/30
#
#  Afterwards: merge_cfix.py swaps the rows into FINAL_*, reanalyse.py redoes
#  the statistics. Clustering does not use CEC functions and is untouched.
#  Do NOT edit any .py file while this runs (rule 9b).
# =====================================================================
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
# Keep the machine from idle-sleeping while this script runs. This is a
# per-process request, released automatically when the script exits; it
# changes no power setting. A closed lid or a manual Sleep still sleeps.
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
$ONLY = @("CEC2017_F2_", "CEC2017_F6_", "CEC2017_F7_", "CEC2017_F8_",
          "CEC2017_F9_", "CEC2017_F13_", "CEC2017_F14_", "CEC2017_F20_")
$W = 20

Stamp "PREFLIGHT"
py -3.14 verify_cec2017_data.py

Stamp "STAGE 1/5  FES PARITY   8 algs x 8 funcs x D=10/30/50/100 x 51 runs"
py -3.14 MDAC_EXPERIMENT.py --stage full --cec official --workers $W --only @ONLY --out FINAL_fes_cfix

Stamp "STAGE 2/5  equal-time budget for the re-run cells"
if (-not (Test-Path "_backup_2026-10-16\equal_time_budget.json")) {
  Copy-Item equal_time_budget.json "_backup_2026-10-16\equal_time_budget.json"
}
py -3.14 make_time_budget.py FINAL_fes_cfix --tag full --out equal_time_budget_cfix.json
py -3.14 merge_time_budget.py equal_time_budget.json equal_time_budget_cfix.json --machine "i7-13700F, $W workers"

Stamp "STAGE 3/5  EQUAL WALL-CLOCK   same 8 algs, same dims, 51 runs"
py -3.14 MDAC_EXPERIMENT.py --stage full --cec official --workers $W --time_budget --only @ONLY --out FINAL_time_cfix

Stamp "STAGE 4/5  FULL ABLATION at 51 runs, D=10/30/50"
py -3.14 MDAC_EXPERIMENT.py --stage abl51 --cec official --workers $W --only @ONLY --out FINAL_abl_cfix

Stamp "STAGE 5/5  ARCHITECTURAL-PARITY CONTROL, D=10/30, 51 runs"
py -3.14 MDAC_EXPERIMENT.py --stage parity --cec official --runs 51 --workers $W --only @ONLY --out FINAL_parity_cfix

Stamp "ALL FIVE STAGES FINISHED"
