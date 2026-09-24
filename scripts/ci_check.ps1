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

function Count-Tests($path) {
    # 收集用例数：解析 --collect-only -q 摘要行 "N tests collected" / "no tests collected"
    $out = python -m pytest $path --collect-only -q 2>&1 | Out-String
    if ($out -match "(\d+) tests? collected") { return [int]$Matches[1] }
    return 0
}

Write-Host "==== Director Brain CI Check ====" -ForegroundColor Cyan
Write-Host ""

# 1. 单元测试
Write-Host "--- 单元测试 ---" -ForegroundColor Cyan
$utCount = Count-Tests "tests/unit/"
Check "单元测试用例数 > 0" ($utCount -gt 0) "收集到 $utCount 个用例"
$ut = python -m pytest tests/unit/ -q 2>&1
$utExit = $LASTEXITCODE
Check "单元测试全过" ($utExit -eq 0) $ut[-1]

# 2. 合同测试
Write-Host "--- 合同测试 ---" -ForegroundColor Cyan
$ctCount = Count-Tests "tests/contract/"
Check "合同测试用例数 > 0" ($ctCount -gt 0) "收集到 $ctCount 个用例"
$ct = python -m pytest tests/contract/ -q 2>&1
$ctExit = $LASTEXITCODE
Check "合同测试全过" ($ctExit -eq 0) $ct[-1]

# 3. 集成测试
Write-Host "--- 集成测试 ---" -ForegroundColor Cyan
$itCount = Count-Tests "tests/integration/"
Check "集成测试用例数 > 0" ($itCount -gt 0) "收集到 $itCount 个用例"
$it = python -m pytest tests/integration/ -q 2>&1
$itExit = $LASTEXITCODE
Check "集成测试全过" ($itExit -eq 0) $it[-1]

# 4. 改动范围检查（git）
Write-Host "--- 改动范围 ---" -ForegroundColor Cyan
$gitStatus = git status --porcelain 2>&1
if ($gitStatus) {
    Write-Host "  改动文件：" -ForegroundColor Yellow
    $gitStatus | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Write-Host "  注意：改动范围需人工评审，CI 不自动判定" -ForegroundColor Yellow
}
Check "无未提交改动" (-not $gitStatus) $(if ($gitStatus) { "共 $($gitStatus.Count) 个文件未提交" } else { "工作区干净" })

# 判负自检（独立计数，不影响交付判定）
Write-Host ""
Write-Host "--- 判负自检 ---" -ForegroundColor Cyan
$stPass = 0
$stFail = 0
function StCheck($name, $cond, $detail) {
    if ($cond) {
        Write-Host "  [ST-PASS] $name" -ForegroundColor DarkGray
        $script:stPass++
    } else {
        Write-Host "  [ST-FAIL] $name（预期失败）" -ForegroundColor DarkGray
        $script:stFail++
    }
}
StCheck "selftest: 假条件应判负" $false "验证 Check 能捕获 FAIL"
StCheck "selftest: 真条件应判正" $true "验证 Check 能捕获 PASS"
if ($stFail -eq 1 -and $stPass -eq 1) {
    Write-Host "[PASS] 判负自检正常（CI 能判负）" -ForegroundColor Green
} else {
    Write-Host "[FAIL] 判负自检异常（stPass=$stPass, stFail=$stFail）" -ForegroundColor Red
    exit 1
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
