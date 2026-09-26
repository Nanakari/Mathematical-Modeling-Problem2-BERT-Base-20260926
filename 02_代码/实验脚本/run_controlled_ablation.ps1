$ErrorActionPreference = 'Stop'
$root = 'D:\codex_projects\projects\Mathematical Modeling\problem2'
$python = Join-Path $root '.venv-cuda\Scripts\python.exe'
$lane = Join-Path $root 'outputs\bert_base_paper_20260926'
$features = Join-Path $root 'outputs\teacher_student_v5_20260924\data\train_valid_only_v5.pkl'
$script = Join-Path $lane 'ablation_run.py'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location $root

$name = 'no_missing_aug_seed1729'
& $python $script train --config (Join-Path $lane "$name.json") --feature-path $features --resume *> (Join-Path $lane "$name.resume.log")
if ($LASTEXITCODE -ne 0) { throw "$name resume failed" }
@{ run = $name; stage = 'train_completed'; completed_utc = (Get-Date).ToUniversalTime().ToString('o') } |
  ConvertTo-Json -Compress | Add-Content -Encoding utf8 (Join-Path $lane 'controlled_status.jsonl')

foreach ($name in @('text_only_seed1729', 'text_audio_seed1729', 'text_vision_seed1729', 'no_av_direct_seed1729')) {
  $config = Join-Path $lane "$name.json"
  & $python $script prepare --config $config --feature-path $features *> (Join-Path $lane "$name.prepare.log")
  if ($LASTEXITCODE -ne 0) { throw "$name prepare failed" }
  & $python $script train --config $config --feature-path $features *> (Join-Path $lane "$name.train.log")
  if ($LASTEXITCODE -ne 0) { throw "$name train failed" }
  @{ run = $name; stage = 'train_completed'; completed_utc = (Get-Date).ToUniversalTime().ToString('o') } |
    ConvertTo-Json -Compress | Add-Content -Encoding utf8 (Join-Path $lane 'controlled_status.jsonl')
}
