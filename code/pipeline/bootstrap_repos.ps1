param(
  [string]$Root = "$PSScriptRoot\..\vendor",
  [string]$GvhMrCommit = "6ec3ca39336c50492c0fae65fba2fb831fc7d866",
  [string]$GmrCommit = "bb1bbe40774794fceb2a7c579a3464a28e68c844",
  [string]$BeyondMimicCommit = "2184fa93e950960733d7024930a2c79616d295a0"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Root | Out-Null
function Get-PinnedRepo([string]$Url, [string]$Name, [string]$Commit) {
  $dest = Join-Path $Root $Name
  if (Test-Path $dest) { throw "Refusing to reuse existing checkout: $dest" }
  git clone $Url $dest
  git -C $dest checkout --detach $Commit
  git -C $dest status --short --branch
}
Get-PinnedRepo "https://github.com/zju3dv/GVHMR.git" "GVHMR" $GvhMrCommit
Get-PinnedRepo "https://github.com/YanjieZe/GMR.git" "GMR" $GmrCommit
Get-PinnedRepo "https://github.com/HAOTianGa03/GAOTIANHAO-G1-BeyondMimic.git" "BeyondMimic" $BeyondMimicCommit
Write-Host "Pinned source checkouts created under $Root"
