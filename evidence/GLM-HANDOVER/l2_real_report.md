# L2 素材矩阵 · 鲁棒性画像

任务 24：出片 8｜拒绝 16｜错误 0｜出片中带硬伤 3

| 任务 | 素材 | 状态 | 时长/目标 | 偏差% | 镜头 | ASL | 黑帧 | 硬伤 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| MVI_4301-base3 | MVI_4301 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4301-fast3 | MVI_4301 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4380-base3 | MVI_4380 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4380-fast3 | MVI_4380 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4307-base3 | MVI_4307 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4307-fast3 | MVI_4307 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4315-base3 | MVI_4315 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4315-fast3 | MVI_4315 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4211-base3 | MVI_4211 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4211-fast3 | MVI_4211 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4417-base3 | MVI_4417 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4417-fast3 | MVI_4417 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4340-base3 | MVI_4340 | 出片⚠硬伤 | 2.7s/3s | 10.0% | 1 | 2.7s | 0 | YES | - |
| MVI_4340-fast3 | MVI_4340 | **拒绝** | - | - | - | - | - | - | 全部 1 个候选镜头均违反 must_avoid 约束（['模糊']）——素材无法满足用户约束，拒绝导演（fail-closed）；请更换素材或调整意图 |
| MVI_4246-base3 | MVI_4246 | 出片⚠硬伤 | 2.7s/3s | 10.0% | 1 | 2.7s | 0 | YES | - |
| MVI_4246-fast3 | MVI_4246 | 出片⚠硬伤 | 2.7s/3s | 10.0% | 1 | 2.7s | 0 | YES | 快剪风格，避免模糊镜头 |
| MVI_4384-base4 | MVI_4384 | 出片✅ | 3.6s/4s | 10.0% | 1 | 3.6s | 0 | no | - |
| MVI_4384-fast4 | MVI_4384 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4320-base4 | MVI_4320 | 出片✅ | 3.6s/4s | 10.0% | 1 | 3.6s | 0 | no | - |
| MVI_4320-fast4 | MVI_4320 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4213-base4 | MVI_4213 | 出片✅ | 3.6s/4s | 10.0% | 1 | 3.6s | 0 | no | - |
| MVI_4213-fast4 | MVI_4213 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| MVI_4467-base6 | MVI_4467 | 出片✅ | 5.4s/6s | 10.0% | 3 | 1.8s | 0 | no | - |
| MVI_4467-fast6 | MVI_4467 | 出片✅ | 5.4s/6s | 10.0% | 3 | 1.8s | 0 | no | 快剪风格，避免模糊镜头 |

> 拒绝 = fail-closed 生效（素材/约束无法满足目标，系统宁可拒绝也不硬凑）。拒绝原因是 L2 的重要产出：它是素材鲁棒性画像的一部分。