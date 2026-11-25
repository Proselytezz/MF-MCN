param(
    [Parameter(Mandatory = $true)][string]$SshHost,
    [string]$User = "root",
    [int]$Port = 22,
    [string]$RemotePath = "/root/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    [string]$RemoteSplits = "/root/data/splits",
    [string]$LocalSrc = "E:\MF-MCM-main\data\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    [string]$LocalSplits = "E:\MF-MCM-main\data\BraTS2023_GLI_01scale",
    [int]$BatchSize = 100,
    [int]$Retry = 3
)

# 只传预训练所需的最小数据集：12.5GB 原始 nii.gz + 4 个划分列表，
# 远端的 h5 由 brats_preprocessing.py 现场生成（已验证与本地逐体素一致）。

$ErrorActionPreference = "Continue"
$doneLog = "E:\MF-MCM-main\transfer_raw_done.log"
$failLog = "E:\MF-MCM-main\transfer_raw_failed.log"

if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
    throw "未找到 tar.exe（需要 Windows 10 及以上自带的 bsdtar）"
}

$cases = Get-ChildItem $LocalSrc -Directory |
    Where-Object { $_.Name -like 'BraTS-GLI-*' } |
    Sort-Object Name |
    Select-Object -ExpandProperty Name
if ($cases.Count -eq 0) { throw "在 $LocalSrc 下未找到 BraTS-GLI-* 目录" }

$totalBytes = (Get-ChildItem $LocalSrc -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("待传病例: {0} 例, 原始数据 {1:N1} GB, 目标 {2}@{3}:{4}" -f `
    $cases.Count, ($totalBytes / 1GB), $User, $SshHost, $RemotePath)

# 断点续传：按批次记录已完成
$done = @{}
if (Test-Path $doneLog) { Get-Content $doneLog | ForEach-Object { $done[$_.Trim()] = $true } }

$sshCommon = @("-p", "$Port", "-o", "StrictHostKeyChecking=accept-new")
$scpCommon = @("-P", "$Port", "-o", "StrictHostKeyChecking=accept-new")

& ssh @sshCommon "$User@$SshHost" "mkdir -p '$RemotePath' '$RemoteSplits' /root/transfer" | Out-Null

# 划分列表（远端无法从 900 例子集反推 1251 例的划分，必须一并上传）
Get-ChildItem $LocalSplits -Filter '*_path_list.txt' | ForEach-Object {
    & scp @scpCommon $_.FullName "${User}@${SshHost}:$RemoteSplits/" | Out-Null
    Write-Host ("已上传划分列表 " + $_.Name)
}

$start = Get-Date
$sentBytes = 0
$failed = @()

for ($i = 0; $i -lt $cases.Count; $i += $BatchSize) {
    $batchId = [Math]::Floor($i / $BatchSize)
    $tag = "batch_{0:d4}" -f $batchId
    if ($done.ContainsKey($tag)) { continue }

    $batch = $cases[$i..([Math]::Min($i + $BatchSize - 1, $cases.Count - 1))]
    $batchBytes = ($batch | ForEach-Object {
        (Get-ChildItem (Join-Path $LocalSrc $_) -Recurse -File | Measure-Object Length -Sum).Sum
    } | Measure-Object -Sum).Sum

    $pkg = Join-Path $env:TEMP "$tag.tar"
    if (Test-Path $pkg) { Remove-Item $pkg -Force }
    # --format=gnutar 保证 Linux 端 GNU tar 可直接解包
    & tar -cf $pkg --format=gnutar -C $LocalSrc @batch
    if ($LASTEXITCODE -ne 0) { throw "打包 $tag 失败" }
    $pkgMB = (Get-Item $pkg).Length / 1MB
    Write-Host ("{0}: {1} 例, 包 {2:N0} MB, 开始上传..." -f $tag, $batch.Count, $pkgMB)

    $ok = $false
    for ($try = 1; $try -le $Retry; $try++) {
        & scp @scpCommon $pkg "${User}@${SshHost}:/root/transfer/$tag.tar"
        if ($LASTEXITCODE -ne 0) { Write-Host ("  第 {0} 次上传失败, 重试" -f $try); continue }
        & ssh @sshCommon "$User@$SshHost" "tar -xf /root/transfer/$tag.tar -C '$RemotePath' && rm -f /root/transfer/$tag.tar"
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        Write-Host ("  第 {0} 次远端解包失败, 重试" -f $try)
    }
    Remove-Item $pkg -Force

    if ($ok) {
        Add-Content $doneLog $tag
        $done[$tag] = $true
        $sentBytes += $batchBytes
    }
    else {
        Add-Content $failLog $tag
        $failed += $tag
    }

    $elapsed = ((Get-Date) - $start).TotalSeconds
    $rate = if ($elapsed -gt 0) { $sentBytes / $elapsed / 1MB } else { 0 }
    $remain = $totalBytes - $sentBytes
    $etaHour = if ($rate -gt 0) { $remain / ($rate * 1MB) / 3600 } else { 0 }
    Write-Host ("进度 {0}/{1} 批, 已传 {2:N1} GB, {3:N1} MB/s, 已用 {4:N0} 分钟, 预计剩余 {5:N1} 小时, 失败 {6}" -f `
        $batchId + 1, [Math]::Ceiling($cases.Count / $BatchSize), ($sentBytes / 1GB), $rate, ($elapsed / 60), $etaHour, $failed.Count)
}

Write-Host ("完成: 失败批次 {0}; 失败清单见 {1}" -f $failed.Count, $failLog)
if ($failed.Count -gt 0) { Write-Host ("失败批次: " + ($failed -join ', ')) }
