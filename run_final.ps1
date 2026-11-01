# =====================================================================
#  THE DEFINITIVE EXPERIMENT -- run once, start to finish.
#
#  Everything the paper needs, at the competition's own 51 runs, on the
#  VERIFIED official CEC2017 suite (_TABLES_VERIFIED = True, eight defects
#  found and fixed), with the shipped configuration (C1 removed).
#
#  Stages are ordered by importance, so an interruption still leaves the
#  most valuable results on disk.
#
#    1  FES parity, 8 algorithms, D = 10/30/50/100
#       -> the headline table AND the two open mechanism questions
#          (A14: metric vs diversity, C2: does the metric generator pay)
#    2  equal-time budget rebuilt from stage 1, covering all four dims
#    3  equal wall-clock, same 8 algorithms, same dims
#       -> the second budget convention; nothing else may run during it
#    4  full ablation at the SAME power as the headline, D = 10/30/50
#    5  architectural-parity control (coded long ago, never run)
#    6  clustering, the AI application, re-tested with the faster algorithm
#
#  D=20 is excluded: only 19 of its 30 rotation files are installed.
# =====================================================================
$ErrorActionPreference = "Continue"
# Pin the working directory to the script's own folder. `-File` does NOT set
# it, so an inherited CWD would scatter FINAL_* and the checkpoints somewhere
# unfindable -- which happened once and cost a relaunch.
Set-Location -Path $PSScriptRoot
Write-Output "working directory: $PWD"
$t0 = Get-Date
function Stamp($msg) {
  $el = [math]::Round(((Get-Date) - $t0).TotalHours, 2)
  Write-Output ""
  Write-Output "=================================================================="
  Write-Output " [$el h] $msg"
  Write-Output "=================================================================="
}

Stamp "PREFLIGHT"
python verify_cec2017_data.py
python benchmarks/clustering.py
python MDAC_EXPERIMENT.py --stage smoke --workers 4

Stamp "STAGE 1/6  FES PARITY   8 algs x 42 funcs x D=10/30/50/100 x 51 runs"
python MDAC_EXPERIMENT.py --stage full --cec official --workers 15 --out FINAL_fes

Stamp "STAGE 2/6  rebuild the equal-time budget from stage 1"
python make_time_budget.py FINAL_fes --tag full

Stamp "STAGE 3/6  EQUAL WALL-CLOCK   same 8 algs, same dims, 51 runs"
python MDAC_EXPERIMENT.py --stage full --cec official --workers 15 --time_budget --out FINAL_time

Stamp "STAGE 4/6  FULL ABLATION at 51 runs, D=10/30/50"
python MDAC_EXPERIMENT.py --stage abl51 --cec official --workers 15 --out FINAL_abl

Stamp "STAGE 5/6  ARCHITECTURAL-PARITY CONTROL, D=10/30, 51 runs"
python MDAC_EXPERIMENT.py --stage parity --cec official --runs 51 --workers 15 --out FINAL_parity

Stamp "STAGE 6/6  CLUSTERING (real UCI data), 51 runs"
python MDAC_EXPERIMENT.py --stage top --cec clustering --dims 40 44 60 65 78 90 --runs 51 --workers 15 --algs MDAC MDAC-ctr jSO L-SHADE-SPACMA L-SHADE EBOwithCMAR CMA-ES k-means++ --out FINAL_cluster

Stamp "ALL SIX STAGES FINISHED"
