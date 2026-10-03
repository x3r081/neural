param(
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Pinned from the official ggml-org/llama.cpp GitHub release API on 2026-10-02.
$tag = 'b11349'
$sourceCommit = 'fb4b2737a808a3fb7c2117a498f43815dc9be53e'
$apiUrl = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$tag"
$sourceUrl = 'https://github.com/ggml-org/llama.cpp.git'
$assets = @(
    @{
        Name = 'llama-b11349-bin-win-cuda-12.4-x64.zip'
        Sha256 = '1084a0a4c6567511c7f67d8c4db71979f837170ed22b2fcb72c0d8637a4eaf14'
    },
    @{
        Name = 'cudart-llama-bin-win-cuda-12.4-x64.zip'
        Sha256 = '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'
    }
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$toolsRoot = Join-Path $projectRoot '.tools'
$sourceDir = Join-Path $toolsRoot 'llama.cpp'
$downloadsDir = Join-Path $toolsRoot 'downloads'
$binaryDir = Join-Path $toolsRoot 'llama-b11349-win-cuda-12.4'
$manifestPath = Join-Path $toolsRoot 'llama-qwen-setup.json'
New-Item -ItemType Directory -Force -Path $toolsRoot, $downloadsDir, $binaryDir | Out-Null

if ($VerifyOnly) {
    if (-not (Test-Path -LiteralPath $sourceDir)) { throw "Missing llama.cpp source: $sourceDir" }
    $actualCommit = (& git -C $sourceDir rev-parse HEAD).Trim()
    if ($actualCommit -ne $sourceCommit) { throw "llama.cpp source mismatch: expected $sourceCommit, got $actualCommit" }
    $cli = Join-Path $binaryDir 'llama-cli.exe'
    if (-not (Test-Path -LiteralPath $cli)) { throw "Missing binary: $cli" }
    $version = (& $cli --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $version -notmatch '11349' -or $version -notmatch 'fb4b273') {
        throw "llama-cli version mismatch: $version"
    }
    Write-Output "SOURCE_OK commit=$actualCommit"
    Write-Output "BINARY_OK $version"
    Get-Content -LiteralPath $manifestPath
    exit 0
}

$release = Invoke-RestMethod -Uri $apiUrl -Headers @{
    'User-Agent' = 'NeuralQwen36-llama-setup'
    'Accept' = 'application/vnd.github+json'
}
if ($release.tag_name -ne $tag) { throw "GitHub API tag mismatch: $($release.tag_name)" }
if ($release.target_commitish -ne $sourceCommit) {
    throw "GitHub API commit mismatch: expected $sourceCommit, got $($release.target_commitish)"
}

if (-not (Test-Path -LiteralPath $sourceDir)) {
    & git clone --branch $tag --depth 1 --single-branch $sourceUrl $sourceDir
    if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }
}
$actualCommit = (& git -C $sourceDir rev-parse HEAD).Trim()
if ($actualCommit -ne $sourceCommit) { throw "llama.cpp source mismatch: expected $sourceCommit, got $actualCommit" }

foreach ($asset in $assets) {
    $apiAsset = $release.assets | Where-Object { $_.name -eq $asset.Name } | Select-Object -First 1
    if (-not $apiAsset) { throw "Pinned asset is absent from official release $tag`: $($asset.Name)" }
    $apiHash = ([string]$apiAsset.digest -replace '^sha256:', '').ToLowerInvariant()
    if ($apiHash -ne $asset.Sha256) { throw "GitHub API digest changed for $($asset.Name): $apiHash" }

    $archive = Join-Path $downloadsDir $asset.Name
    $actualHash = if (Test-Path -LiteralPath $archive) {
        (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { '' }
    if ($actualHash -ne $asset.Sha256) {
        $partial = "$archive.part"
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
        & curl.exe --fail --location --retry 3 --retry-delay 2 --output $partial $apiAsset.browser_download_url
        if ($LASTEXITCODE -ne 0) { throw "Asset download failed for $($asset.Name)" }
        $actualHash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $asset.Sha256) { throw "SHA256 mismatch for downloaded $($asset.Name): $actualHash" }
        Move-Item -LiteralPath $partial -Destination $archive -Force
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $binaryDir -Force
    Write-Output "ASSET_OK $($asset.Name) sha256=$actualHash bytes=$((Get-Item -LiteralPath $archive).Length)"
}

$cli = Join-Path $binaryDir 'llama-cli.exe'
if (-not (Test-Path -LiteralPath $cli)) { throw "Archive did not provide expected llama-cli.exe at $cli" }
$version = (& $cli --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $version -notmatch '11349' -or $version -notmatch 'fb4b273') {
    throw "llama-cli version mismatch: $version"
}

$manifest = [ordered]@{
    repository = 'https://github.com/ggml-org/llama.cpp'
    release_api = $apiUrl
    tag = $tag
    source_commit = $sourceCommit
    release_published_at = ([datetime]$release.published_at).ToUniversalTime().ToString('o')
    source_path = $sourceDir
    binary_path = $binaryDir
    binary_flavor = 'Windows x64 CUDA 12.4; companion CUDA runtime archive applied'
    version_output = $version
    assets = @($assets | ForEach-Object {
        $assetPath = Join-Path $downloadsDir $_.Name
        [ordered]@{ name = $_.Name; path = $assetPath; sha256 = $_.Sha256; bytes = (Get-Item -LiteralPath $assetPath).Length }
    })
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Output "SETUP_OK $version"
Write-Output "MANIFEST $manifestPath"
