# Director Brain CI 检查脚本（监工用）
# 任何窗口交付前必须跑，全部 PASS 才算交付
$ErrorActionPreference = "Continue"
$pass = 0
$fail = 0

function Check($name, $cond, $detail) {
    if ($cond) {
        Write-Host "[PASS] $name" -ForegroundColor Green
        if ($detail) { Write-Host "       $detail" -ForegroundColor DarkGray }
        $script:pass++
    } else {
        Write-Host "[FAIL] $name" -ForegroundColor Red
        if ($detail) { Write-Host "       $detail" -ForegroundColor DarkGray }
        $script:fail++
    }
}

Write-Host "==== Director Brain CI Check ====" -ForegroundColor Cyan
Write-Host ""

# 1. 单元测试
Write-Host "--- 单元测试 ---" -ForegroundColor Cyan
$ut = python -m pytest tests/unit/ -q 2>&1
$utExit = $LASTEXITCODE
Check "单元测试全过" ($utExit -eq 0) $ut[-1]

# 2. 合同测试
Write-Host "--- 合同测试 ---" -ForegroundColor Cyan
$ct = python -m pytest tests/contract/ -q 2>&1
$ctExit = $LASTEXITCODE
Check "合同测试全过" ($ctExit -eq 0) $ct[-1]

# 3. 集成测试（如果有）
Write-Host "--- 集成测试 ---" -ForegroundColor Cyan
$it = python -m pytest tests/integration/ -q 2>&1
$itExit = $LASTEXITCODE
Check "集成测试全过" ($itExit -eq 0) $it[-1]

# 4. 改动范围检查（git）
Write-Host "--- 改动范围 ---" -ForegroundColor Cyan
$gitStatus = git status --porcelain 2>&1
if ($gitStatus) {
    Write-Host "  改动文件：" -ForegroundColor Yellow
    $gitStatus | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Check "改动范围已评审" $true "共 $($gitStatus.Count) 个文件改动"
} else {
    Check "无未提交改动" $true "工作区干净"
}

# 总结
Write-Host ""
Write-Host "==== 总结 ====" -ForegroundColor Cyan
Write-Host "PASS: $pass  FAIL: $fail" -ForegroundColor $(if ($fail -eq 0) { "Green" } else { "Red" })
if ($fail -gt 0) {
    Write-Host "有 FAIL 项，交付打回。" -ForegroundColor Red
    exit 1
} else {
    Write-Host "全部通过，可以交付。" -ForegroundColor Green
}
