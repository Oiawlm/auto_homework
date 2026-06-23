param(
    [string]$WorkspaceRoot
)

$ErrorActionPreference = "Stop"

$skillDir = Split-Path -Parent $PSScriptRoot
$skillPath = Join-Path $skillDir "SKILL.md"
$referencePath = Join-Path $skillDir "references\answers.md"

if ($WorkspaceRoot) {
    $answerPath = Join-Path (Resolve-Path $WorkspaceRoot) "answers.md"
} else {
    $repoAnswerPath = Join-Path (Resolve-Path (Join-Path $skillDir "..\..")) "answers.md"
    $answerPath = if (Test-Path $repoAnswerPath) { $repoAnswerPath } else { $referencePath }
}

$skill = Get-Content -Raw -Encoding UTF8 $skillPath
$answer = Get-Content -Raw -Encoding UTF8 $answerPath
$errors = [System.Collections.Generic.List[string]]::new()

$forbiddenPatterns = [ordered]@{
    "(?m)^\u66F4\u65B0\u65F6\u95F4\uFF1A" = "dated update metadata"
    "(?m)^.*\u8865\u5145\u8BB0\u5F55\uFF08" = "dated or account-specific supplement"
    "(?m)^## \u672C\u8F6E" = "session-history section"
    "(?m)^\u672C\u6B21\u901A\u8FC7\u5165\u53E3\uFF1A" = "session-specific route label"
    "(?m)^.*\u8D26\u53F7\s+\d+" = "account identifier"
    "\u6B64\u524D\u540C\u6837\u903B\u8F91" = "chronological retry narrative"
    "\u8865\u9F50\u540E\u4F5C\u4E1A\u4ECE" = "before-and-after progress narrative"
}

foreach ($entry in $forbiddenPatterns.GetEnumerator()) {
    if ($answer -match $entry.Key) {
        $errors.Add("Answer bank contains $($entry.Value).")
    }
}

$canonicalRoute = "(?s)\| Homework list \|.*\u67E5\u770B\u4F5C\u54C1.*\| Work/detail or overall-evaluation page \|.*\u8FDB\u5165\u5B9E\u8BAD.*\| Task page \|"
if ($skill -notmatch $canonicalRoute) {
    $errors.Add("SKILL.md does not contain the canonical list -> detail -> task route.")
}

if ((Resolve-Path $answerPath).Path -ne (Resolve-Path $referencePath).Path) {
    $answerHash = (Get-FileHash -Algorithm SHA256 $answerPath).Hash
    $referenceHash = (Get-FileHash -Algorithm SHA256 $referencePath).Hash
    if ($answerHash -ne $referenceHash) {
        $errors.Add("Workspace and bundled answer banks are not synchronized.")
    }
}

if ($errors.Count -gt 0) {
    $errors | ForEach-Object { Write-Error $_ -ErrorAction Continue }
    exit 1
}

Write-Output "Documentation validation passed."
