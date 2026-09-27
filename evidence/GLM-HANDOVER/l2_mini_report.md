# L2 素材矩阵 · 鲁棒性画像

任务 6：出片 5｜拒绝 1｜错误 0｜出片中带硬伤 5

| 任务 | 素材 | 状态 | 时长/目标 | 偏差% | 镜头 | ASL | 黑帧 | 硬伤 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| ref-base15 | sintel_ref_480p | 出片⚠硬伤 | 13.5s/15s | 10.0% | 4 | 3.375s | 3 | YES | - |
| ref-fastcut15 | sintel_ref_480p | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| ref-slow10 | sintel_ref_480p | 出片⚠硬伤 | 10.0s/10s | 0.0% | 4 | 2.5s | 2 | YES | 慢节奏，长镜头 |
| trailer-base15 | sintel_trailer | 出片⚠硬伤 | 15.083s/15s | 0.6% | 6 | 2.514s | 3 | YES | - |
| trailer-fastcut15 | sintel_trailer | 出片⚠硬伤 | 15.583s/15s | 3.9% | 7 | 2.226s | 2 | YES | 快剪风格 |
| trailer-slow10 | sintel_trailer | 出片⚠硬伤 | 11.0s/10s | 10.0% | 5 | 2.2s | 2 | YES | 慢节奏，长镜头 |

> 拒绝 = fail-closed 生效（素材/约束无法满足目标，系统宁可拒绝也不硬凑）。拒绝原因是 L2 的重要产出：它是素材鲁棒性画像的一部分。