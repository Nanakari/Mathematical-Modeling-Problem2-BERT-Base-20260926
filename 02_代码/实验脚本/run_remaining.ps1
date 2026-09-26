$ErrorActionPreference = 'Stop'
$root = 'D:\codex_projects\projects\Mathematical Modeling\problem2'
$python = Join-Path $root '.venv-cuda\Scripts\python.exe'
$lane = Join-Path $root 'outputs\bert_base_paper_20260926'
$features = Join-Path $root 'outputs\teacher_student_v5_20260924\data\train_valid_only_v5.pkl'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location $root

$runs = @(
  @{ Name = 'no_missing_aug_seed1729'; Script = (Join-Path $lane 'ablation_run.py'); Train = $true }
)

foreach ($run in $runs) {
  $name = $run.Name
  $config = Join-Path $lane "$name.json"
  $out = Join-Path $lane $name
  $status = Join-Path $lane 'run_status.jsonl'
  if ($run.Train) {
    & $python $run.Script prepare --config $config --feature-path $features *> (Join-Path $lane "$name.prepare.log")
    if ($LASTEXITCODE -ne 0) { throw "$name prepare failed" }
    & $python $run.Script train --config $config --feature-path $features *> (Join-Path $lane "$name.train.log")
    if ($LASTEXITCODE -ne 0) { throw "$name train failed" }
    @{ run = $name; stage = 'train'; completed_utc = (Get-Date).ToUniversalTime().ToString('o') } |
      ConvertTo-Json -Compress | Add-Content -Encoding utf8 $status
  }
  $checkpoint = Join-Path $out 'best_model.safetensors'
  $evalOut = Join-Path $out 'validation_quick.json'
  & $python $run.Script evaluate --config $config --checkpoint $checkpoint --feature-path $features --scenario-set quick_all_random --output $evalOut *> (Join-Path $lane "$name.evaluate.log")
  if ($LASTEXITCODE -ne 0) { throw "$name evaluate failed" }
  @{ run = $name; stage = 'evaluate'; completed_utc = (Get-Date).ToUniversalTime().ToString('o') } |
    ConvertTo-Json -Compress | Add-Content -Encoding utf8 $status
}
