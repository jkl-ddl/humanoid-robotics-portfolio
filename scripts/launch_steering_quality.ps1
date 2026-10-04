param(
    [ValidateSet('Smoke', 'Train')][string]$Mode = 'Smoke',
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [Parameter(Mandatory=$true)][string]$SourceRoot,
    [Parameter(Mandatory=$true)][string]$PriorRoot,
    [Parameter(Mandatory=$true)][string]$BaseCheckpoint,
    [string]$CodeRoot = "$PSScriptRoot\..\code",
    [string]$Image = 't800-smp-runtime:restore',
    [ValidateRange(1, 20000)][int]$Iterations = 500,
    [ValidateRange(0.0, 0.015)][double]$ArmPostureWeight = 0.003,
    [ValidateRange(0.5, 4.0)][double]$TaskRewardWeight = 2.0,
    [ValidateRange(0.25, 4.0)][double]$VelocityErrorScale = 2.0,
    [ValidateRange(1.0, 8.0)][double]$SmpErrorScale = 6.0,
    [ValidateSet('configured', 'walk12')][string]$PriorEeContract = 'walk12',
    [ValidateRange(0.1, 0.95)][double]$VelocityTaskFraction = 0.75,
    [ValidateRange(-0.5, 0.0)][double]$FootSlipWeight = -0.10,
    [ValidateRange(0.08, 0.25)][double]$FootSiteTarget = 0.16,
    [switch]$NoArmPosture
)
$ErrorActionPreference = 'Stop'
# Public defaults describe the selected 36991 configuration. Historical
# branches require their own recorded parameters; the archived script is intact.
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$CodeRoot = (Resolve-Path -LiteralPath $CodeRoot).Path
if ($NoArmPosture) { $ArmPostureWeight = 0.0 }
$weightText = $ArmPostureWeight.ToString([Globalization.CultureInfo]::InvariantCulture)
$taskWeightText = $TaskRewardWeight.ToString([Globalization.CultureInfo]::InvariantCulture)
$velocityScaleText = $VelocityErrorScale.ToString([Globalization.CultureInfo]::InvariantCulture)
$smpScaleText = $SmpErrorScale.ToString([Globalization.CultureInfo]::InvariantCulture)
$velocityFractionText = $VelocityTaskFraction.ToString([Globalization.CultureInfo]::InvariantCulture)
$slipWeightText = $FootSlipWeight.ToString([Globalization.CultureInfo]::InvariantCulture)
$siteTargetText = $FootSiteTarget.ToString([Globalization.CultureInfo]::InvariantCulture)
$checkpointFile = Get-Item -LiteralPath $BaseCheckpoint
$parentLabel = $checkpointFile.BaseName.Replace('model_', '')
$containerName = 'smp-steering-quality-' + $Mode.ToLowerInvariant() + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$stageRoot = Join-Path $OutputRoot $Mode.ToLowerInvariant()
if (Test-Path -LiteralPath $stageRoot) { throw "Refuse to reuse output directory: $stageRoot" }
foreach ($path in @($SourceRoot, $PriorRoot, $BaseCheckpoint, $CodeRoot)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing input: $path" }
}
$existingNames = docker ps -a --format '{{.Names}}'
if ($LASTEXITCODE -ne 0) { throw 'Docker is unavailable.' }
if ($existingNames -contains $containerName) { throw "Container already exists: $containerName" }
$parentDir = Join-Path $stageRoot "smp_steering_quality_t800\parent$parentLabel"
New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
Copy-Item -LiteralPath $BaseCheckpoint -Destination (Join-Path $parentDir $checkpointFile.Name)

# Only config selection is local. Environment, PPO and runner use upstream code.
$trainingCode = @'
import json, sys
from dataclasses import replace
import steering_quality_task as quality
from mjlab.scripts.train import TrainConfig, launch_training
mode, parent, checkpoint = sys.argv[1:4]
iterations, arm_weight = int(sys.argv[4]), float(sys.argv[5])
task_weight = float(sys.argv[6])
velocity_scale = float(sys.argv[7])
smp_scale = float(sys.argv[8])
prior_contract = sys.argv[9]
velocity_fraction = float(sys.argv[10])
slip_weight, site_target = float(sys.argv[11]), float(sys.argv[12])
cfg = TrainConfig.from_task(quality.TASK_ID)
if prior_contract == 'walk12':
    from prior_contract import apply_walk12_prior_contract
    apply_walk12_prior_contract(cfg.env)
cfg.env.rewards['upper_arm_posture'].weight = arm_weight
cfg.env.rewards['foot_slip'].weight = slip_weight
cfg.env.rewards['foot_clearance'].params['target_height'] = site_target
cfg.env.rewards['task_smp_product'].weight = task_weight
cfg.env.rewards['task_smp_product'].params['ws'] = smp_scale
velocity_term = cfg.env.rewards['task_smp_product'].params['task_terms'][0]
if velocity_term[0].__name__ != 'steering_target_velocity':
    raise ValueError('Unexpected task contract: first reward must track velocity')
velocity_term[2]['vel_err_scale'] = velocity_scale
task_terms = cfg.env.rewards['task_smp_product'].params['task_terms']
if len(task_terms) != 2 or task_terms[1][0].__name__ != 'steering_face_direction':
    raise ValueError('Unexpected task contract: second reward must track facing')
cfg.env.rewards['task_smp_product'].params['task_terms'] = (
    (velocity_term[0], velocity_fraction, velocity_term[2]),
    (task_terms[1][0], 1.0 - velocity_fraction, task_terms[1][2]),
)
cfg.env.scene.num_envs = 32 if mode == 'Smoke' else 512
cfg.agent.seed = 42
cfg.agent.logger = 'tensorboard'
cfg.agent.max_iterations = 2 if mode == 'Smoke' else iterations
cfg.agent.save_interval = 1 if mode == 'Smoke' else 100
cfg.agent.resume = True
cfg.agent.load_run = 'parent' + parent
cfg.agent.load_checkpoint = checkpoint
cfg.agent.run_name = f'resume{parent}_quality_{mode.lower()}_seed42_arm{arm_weight:g}_task{task_weight:g}_vel{velocity_scale:g}_ws{smp_scale:g}_ee{prior_contract}_vmix{velocity_fraction:g}_slip{abs(slip_weight):g}_site{site_target:g}'
cfg = replace(cfg, log_root='/logs', enable_nan_guard=True)
cfg.env.sim.nan_guard.output_dir = '/logs/nan_guard'
if mode == 'Smoke':
    cfg.env.events['init_smp_state'].params.update(
        gsi_buffer_size=64, gsi_batch_size=64, compile_model=False)
    cfg.env.events['gsi_refresh'].params['num_samples'] = 64
print(json.dumps({'task': quality.TASK_ID, 'mode': mode,
  'num_envs': cfg.env.scene.num_envs,
  'iterations': cfg.agent.max_iterations,
  'flat_orientation_weight': cfg.env.rewards['flat_orientation'].weight,
  'foot_site_target_m': cfg.env.rewards['foot_clearance'].params['target_height'],
  'upper_arm_posture_weight': cfg.env.rewards['upper_arm_posture'].weight}), flush=True)
print(json.dumps({'task_smp_product_weight': cfg.env.rewards['task_smp_product'].weight}), flush=True)
print(json.dumps({'velocity_error_scale': velocity_term[2]['vel_err_scale']}), flush=True)
print(json.dumps({'smp_error_scale': cfg.env.rewards['task_smp_product'].params['ws']}), flush=True)
print(json.dumps({'velocity_task_fraction': velocity_fraction,
                  'facing_task_fraction': 1.0 - velocity_fraction}), flush=True)
print(json.dumps({'foot_slip_weight': slip_weight,
                  'foot_site_target_m': site_target}), flush=True)
print(json.dumps({'prior_ee_contract': prior_contract,
                  'prior_ee_body_names': cfg.env.events['init_smp_state'].params['ee_body_names']}), flush=True)
launch_training(quality.TASK_ID, cfg)
'@
$dockerArgs = @(
    'run', '--rm', '--gpus', 'all', '--name', $containerName,
    '--shm-size', '1g', '--entrypoint', 'python',
    '-e', 'PYTHONUNBUFFERED=1', '-e', 'MUJOCO_GL=disable',
    '-e', 'ENGINEAI_NATIVE_SDK_ROOT=/native', '-e', 'SMP_T800_PRIOR=/prior/pretrained.pt',
    '-e', 'SMP_RUNTIME_CONTRACT_JSON=/logs/smp_runtime_contract.json',
    '-e', 'PYTHONPATH=/mjlab/src:/smp/src:/work',
    '--mount', "type=bind,source=$CodeRoot,target=/work,readonly",
    '--mount', "type=bind,source=$SourceRoot\smp,target=/smp,readonly",
    '--mount', "type=bind,source=$SourceRoot\mjlab,target=/mjlab,readonly",
    '--mount', "type=bind,source=$SourceRoot\engineai_robotics_native_sdk,target=/native,readonly",
    '--mount', "type=bind,source=$PriorRoot,target=/prior,readonly",
    '--mount', "type=bind,source=$stageRoot,target=/logs",
    '--mount', 'type=volume,source=t800-smp-warp-cache-20261004,target=/root/.cache/warp',
    '--mount', 'type=volume,source=t800-smp-torch-cache-20261004,target=/root/.cache/torch',
    $Image, '-c', $trainingCode, $Mode, $parentLabel, $checkpointFile.Name,
    [string]$Iterations, $weightText, $taskWeightText, $velocityScaleText, $smpScaleText, $PriorEeContract, $velocityFractionText, $slipWeightText, $siteTargetText
)
& docker @dockerArgs
if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE; output retained in $stageRoot" }
