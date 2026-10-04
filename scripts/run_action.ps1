param(
    [Parameter(Mandatory=$true)][string]$RestoreRoot,
    [Parameter(Mandatory=$true)][string]$Python,
    [ValidateSet('celebration','taunt')][string]$Action = 'celebration',
    [ValidateSet('eval','train')][string]$Mode = 'eval'
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $RestoreRoot).Path
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$repo = Join-Path $root 'source\engineai_rl_lab'
$isaac = Join-Path $root 'source\IsaacLab\source'
$selected = Get-Content -LiteralPath (Join-Path $root 'selected_models.json') -Raw | ConvertFrom-Json
$record = $selected.$Action
if ($null -eq $record) { throw "No selected model for $Action" }
$taskId = if ($record.PSObject.Properties.Name -contains 'task_id') {
    [string]$record.task_id
} else {
    'Tracking-Flat-T800-Wo-State-Estimation-v0'
}
if ($taskId -notin @('Tracking-Flat-T800-v0', 'Tracking-Flat-T800-Wo-State-Estimation-v0')) {
    throw "Unsupported tracking task for $Action`: $taskId"
}
$checkpoint = Join-Path $root $record.checkpoint
$motion = Join-Path $root $record.motion
if (-not (Test-Path -LiteralPath $checkpoint) -or -not (Test-Path -LiteralPath $motion)) { throw 'Model or reference motion is missing.' }
$trainingOverrides = @()
if ($record.PSObject.Properties.Name -contains 'training_overrides') {
    $trainingOverrides = @($record.training_overrides | ForEach-Object { [string]$_ })
}
$env:PYTHONUTF8 = '1'
# The caller must accept NVIDIA's EULA as required by the official runtime.
$env:PYTHONPATH = @(
    (Join-Path $isaac 'isaaclab'),
    (Join-Path $isaac 'isaaclab_assets'),
    (Join-Path $isaac 'isaaclab_tasks'),
    (Join-Path $isaac 'isaaclab_rl'),
    (Join-Path $repo 'source\engineai_rl_lab')
) -join [IO.Path]::PathSeparator
Push-Location -LiteralPath $repo
try {
    if ($Mode -eq 'eval') {
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $output = Join-Path $root "new-evaluation\${Action}_${stamp}.npz"
        & $pythonPath (Join-Path $root 'tools\export_t800_tracking_eval.py') --task $taskId --headless --checkpoint $checkpoint --motion-file $motion --output $output --seed 42 @trainingOverrides
    } else {
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        & $pythonPath (Join-Path $repo 'scripts\tracking\train.py') --task $taskId --headless --num_envs 1024 --max_iterations 20000 --seed 42 --logger tensorboard --run_name "restored_${Action}_${stamp}" --motion_file $motion @trainingOverrides
    }
    if ($LASTEXITCODE -ne 0) { throw "Action $Mode failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
