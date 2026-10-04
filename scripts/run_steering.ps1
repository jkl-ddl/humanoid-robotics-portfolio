param(
    [Parameter(Mandatory=$true)][string]$RestoreRoot,
    [string]$Image = 't800-smp-runtime:restore',
    [string]$OutputDir
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $RestoreRoot).Path
$selected = Get-Content -LiteralPath (Join-Path $root 'selected_models.json') -Raw | ConvertFrom-Json
if (-not $selected.steering) { throw 'No selected steering model.' }
if (-not $OutputDir) {
    $OutputDir = Join-Path $root ('new-evaluation\steering_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
$tools = Join-Path $root 'tools\closeout'
$prior = Split-Path -Parent (Join-Path $root $selected.steering.prior)
$replayArgs = @{
    Checkpoint = (Join-Path $root $selected.steering.checkpoint)
    OutputDir = $OutputDir
    SourceRoot = (Join-Path $root 'source')
    PriorRoot = $prior
    CodeRoot = (Join-Path $tools 'code')
    Image = $Image
    Seed = $selected.steering.seed
}
# Historical releases have no explicit contract selector and retain their
# original replay behavior; new versions carry the selector with the model.
if ($selected.steering.PSObject.Properties.Name -contains 'prior_ee_contract') {
    $replayArgs.PriorEeContract = $selected.steering.prior_ee_contract
}
& (Join-Path $tools 'evaluate_steering.ps1') @replayArgs
