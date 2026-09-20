[CmdletBinding()]
param(
    [switch]$Smoke,
    [switch]$Final
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$PaperRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $MyInvocation.MyCommand.Path))
$BuildDir = Join-Path $PaperRoot 'build'
$TmpDir = Join-Path $BuildDir 'tmp'
$TectonicCacheDir = Join-Path $BuildDir 'cache'
$TemplateDir = Join-Path $PaperRoot 'template'
$MainTex = Join-Path $PaperRoot 'main.tex'
$BuildBase = Join-Path $BuildDir 'main'
$FinalPdf = Join-Path $PaperRoot 'ViPragSent_NAACL.pdf'
$LogCapture = Join-Path $BuildDir 'build_commands.log'
$ManifestPath = Join-Path $BuildDir 'build_manifest.json'

if (-not (Test-Path -LiteralPath $MainTex)) { throw "Missing production entry point: $MainTex" }
foreach ($dir in @($BuildDir, $TmpDir, $TectonicCacheDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

if (-not $Smoke -and -not $Final) {
    $Smoke = $true
}

$SectionDir = Join-Path $PaperRoot 'sections'
$SectionFiles = @()
if (Test-Path -LiteralPath $SectionDir) {
    $SectionFiles = @(Get-ChildItem -LiteralPath $SectionDir -Filter '*.tex' -File)
}
$BibFile = Join-Path $PaperRoot 'references.bib'
$RequiredSections = @(
    'abstract',
    'introduction',
    'related_work',
    'method',
    'experiments',
    'results',
    'discussion',
    'limitations',
    'ethics',
    'conclusion'
)

if ($Final) {
    $missingSections = @($RequiredSections | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $SectionDir ($_.ToString() + '.tex')))
    })
    if ($missingSections.Count -gt 0) {
        throw ('Final build blocked: missing required author sections: ' + ($missingSections -join ', '))
    }
    if (-not (Test-Path -LiteralPath $BibFile) -or (Get-Item -LiteralPath $BibFile).Length -eq 0) {
        throw 'Final build blocked: author-owned references.bib is missing or empty.'
    }

    $sourceFiles = @((Get-Item -LiteralPath $MainTex)) + $SectionFiles + @((Get-Item -LiteralPath $BibFile))
    $sourceText = ($sourceFiles | ForEach-Object { Get-Content -Raw -LiteralPath $_.FullName }) -join "`n"
    $placeholderPattern = '(?i)(AUTHOR[- ]?(TITLE|NAME|INFO)|AUTHOR[- ]?SUPPLIED|PENDING|TODO|TBD|PLACEHOLDER|LOREM IPSUM|FILL[- ]?IN)'
    if ($sourceText -match $placeholderPattern) {
        throw 'Final build blocked: placeholder or author-pending marker remains in the manuscript inputs.'
    }
    if ($sourceText -match '(?i)\\title\s*\{\s*ViPragSent\s*\}') {
        throw 'Final build blocked: the title must be descriptive and identify the XLM-R-large primary system.'
    }

    $bibKeys = @{}
    foreach ($m in [regex]::Matches((Get-Content -Raw -LiteralPath $BibFile), '@\w+\s*\{\s*([^,\s]+)')) {
        $bibKeys[$m.Groups[1].Value] = $true
    }
    $citationKeys = New-Object System.Collections.Generic.HashSet[string]
    foreach ($f in @($SectionFiles)) {
        $sectionText = Get-Content -Raw -LiteralPath $f.FullName
        foreach ($m in [regex]::Matches($sectionText, '\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*)?\{([^}]+)\}')) {
            foreach ($key in $m.Groups[1].Value.Split(',')) {
                $trimmed = $key.Trim()
                if ($trimmed) { [void]$citationKeys.Add($trimmed) }
            }
        }
    }
    $missingCitations = @($citationKeys | Where-Object { -not $bibKeys.ContainsKey($_) })
    if ($missingCitations.Count -gt 0) {
        throw ('Final build blocked: citation keys missing from references.bib: ' + ($missingCitations -join ', '))
    }
}

$tectonicCommand = Get-Command tectonic -ErrorAction SilentlyContinue
if ($tectonicCommand) {
    $tectonicPath = $tectonicCommand.Source
} else {
    $bundledTectonic = Get-ChildItem -Path 'C:\Users\ADM\.codex\plugins\cache' -Filter 'tectonic.exe' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    $tectonicPath = if ($bundledTectonic) { $bundledTectonic.FullName } else { $null }
}
if (-not $tectonicPath) { throw 'Bundled or PATH tectonic is required but was not found.' }

# All mutable compiler state, package/config caches, and temporary files are
# redirected into paper/build. No global package installation is requested.
$env:TEMP = $TmpDir
$env:TMP = $TmpDir
$env:XDG_CACHE_HOME = $TectonicCacheDir
$env:TECTONIC_CACHE_DIR = Join-Path $TectonicCacheDir 'tectonic'

Set-Content -LiteralPath $LogCapture -Value "Build started $(Get-Date -Format o)" -Encoding UTF8

function Invoke-LoggedTool {
    param(
        [Parameter(Mandatory=$true)][string]$FilePath,
        [Parameter(Mandatory=$true)][string[]]$Arguments,
        [Parameter(Mandatory=$true)][string]$Label
    )
    Add-Content -LiteralPath $LogCapture -Value ("`n[$Label] " + $FilePath + ' ' + ($Arguments -join ' '))
    Push-Location $PaperRoot
    try {
        $safeLabel = ($Label -replace '[^A-Za-z0-9_.-]', '_')
        $stdoutPath = Join-Path $TmpDir "$safeLabel.stdout.log"
        $stderrPath = Join-Path $TmpDir "$safeLabel.stderr.log"
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & $FilePath @Arguments 1> $stdoutPath 2> $stderrPath
            $toolExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath | Add-Content -LiteralPath $LogCapture }
        if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath | Add-Content -LiteralPath $LogCapture }
        if ($toolExitCode -ne 0) {
            $tail = (Get-Content -LiteralPath $LogCapture -Tail 30) -join "`n"
            throw "$Label failed with exit code $toolExitCode`n$tail"
        }
    }
    finally {
        Pop-Location
    }
}

$tectonicArgs = @(
    '-X',
    'compile',
    '--keep-logs',
    '--keep-intermediates',
    "-Z",
    "search-path=$TemplateDir",
    "--outdir=$BuildDir",
    $MainTex
)

Invoke-LoggedTool -FilePath $tectonicPath -Arguments $tectonicArgs -Label 'tectonic compile'

$BuiltPdf = "$BuildBase.pdf"
if (-not (Test-Path -LiteralPath $BuiltPdf)) { throw "Compiler completed without producing $BuiltPdf" }
$logText = Get-Content -Raw -LiteralPath $LogCapture
$CompilerLog = "$BuildBase.log"
$compilerLogText = if (Test-Path -LiteralPath $CompilerLog) { Get-Content -Raw -LiteralPath $CompilerLog } else { '' }
if ($Final) {
    if (-not (Test-Path -LiteralPath "$BuildBase.bbl") -or (Get-Item -LiteralPath "$BuildBase.bbl").Length -eq 0) {
        throw 'Final build blocked: bibliography output main.bbl is missing or empty.'
    }
    $combinedLogText = $logText + "`n" + $compilerLogText
    if ($combinedLogText -match '(?im)undefined') {
        throw 'Final build blocked: compiler log contains an undefined citation, reference, or other unresolved symbol.'
    }
}

if ($Final) {
    Copy-Item -LiteralPath $BuiltPdf -Destination $FinalPdf -Force
    Write-Output "FINAL PDF: $FinalPdf"
} else {
    Write-Output "SMOKE PDF: $BuiltPdf"
}

$manifest = [ordered]@{
    generated_at = (Get-Date).ToUniversalTime().ToString('o')
    mode = if ($Final) { 'final' } else { 'smoke' }
    paper_root = $PaperRoot
    engine = $tectonicPath
    bibliography_engine = 'tectonic-managed-bibliography'
    output_pdf = if ($Final) { $FinalPdf } else { $BuiltPdf }
    section_files = @($SectionFiles | ForEach-Object { $_.FullName })
    bibliography_present = Test-Path -LiteralPath $BibFile
    required_sections = $RequiredSections
    mutable_state_root = $BuildDir
    global_install_requested = $false
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
