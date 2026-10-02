# L2 素材矩阵 · 鲁棒性画像

任务 120：出片 96｜拒绝 24｜错误 0｜出片中带硬伤 0

| 任务 | 素材 | 状态 | 时长/目标 | 偏差% | 镜头 | ASL | 黑帧 | 硬伤 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| dev-MVI_4311-base1 | MVI_4311 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| dev-MVI_4311-fast1 | MVI_4311 | **拒绝** | - | - | - | - | - | - | 全部 1 个候选镜头均违反 must_avoid 约束（['模糊']）——素材无法满足用户约束，拒绝导演（fail-closed）；请更换素材或调整意图 |
| dev-MVI_4311-slow1 | MVI_4311 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| dev-MVI_4392-base2 | MVI_4392 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | - |
| dev-MVI_4392-fast2 | MVI_4392 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4392-slow2 | MVI_4392 | 出片✅ | 2.16s/2s | 8.0% | 1 | 2.16s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4395-base1 | MVI_4395 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| dev-MVI_4395-fast1 | MVI_4395 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4395-slow1 | MVI_4395 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| dev-MVI_4256-base3 | MVI_4256 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | - |
| dev-MVI_4256-fast3 | MVI_4256 | **拒绝** | - | - | - | - | - | - | 全部 1 个候选镜头均违反 must_avoid 约束（['模糊']）——素材无法满足用户约束，拒绝导演（fail-closed）；请更换素材或调整意图 |
| dev-MVI_4256-slow3 | MVI_4256 | 出片✅ | 3.24s/3s | 8.0% | 1 | 3.24s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4290-base1 | MVI_4290 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| dev-MVI_4290-fast1 | MVI_4290 | **拒绝** | - | - | - | - | - | - | 全部 1 个候选镜头均违反 must_avoid 约束（['模糊']）——素材无法满足用户约束，拒绝导演（fail-closed）；请更换素材或调整意图 |
| dev-MVI_4290-slow1 | MVI_4290 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| dev-MVI_4223-base1 | MVI_4223 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| dev-MVI_4223-fast1 | MVI_4223 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4223-slow1 | MVI_4223 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| dev-MVI_4240-base2 | MVI_4240 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | - |
| dev-MVI_4240-fast2 | MVI_4240 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4240-slow2 | MVI_4240 | 出片✅ | 2.16s/2s | 8.0% | 1 | 2.16s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4262-base3 | MVI_4262 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | - |
| dev-MVI_4262-fast3 | MVI_4262 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4262-slow3 | MVI_4262 | 出片✅ | 3.24s/3s | 8.0% | 1 | 3.24s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4320-base4 | MVI_4320 | 出片✅ | 3.68s/4s | 8.0% | 1 | 3.68s | 0 | no | - |
| dev-MVI_4320-fast4 | MVI_4320 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| dev-MVI_4320-slow4 | MVI_4320 | 出片✅ | 4.32s/4s | 8.0% | 1 | 4.32s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4456-base1 | MVI_4456 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| dev-MVI_4456-fast1 | MVI_4456 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4456-slow1 | MVI_4456 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| dev-MVI_4301-base1 | MVI_4301 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| dev-MVI_4301-fast1 | MVI_4301 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4301-slow1 | MVI_4301 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| dev-MVI_4404-base1 | MVI_4404 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| dev-MVI_4404-fast1 | MVI_4404 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4404-slow1 | MVI_4404 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| dev-MVI_4212-base3 | MVI_4212 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | - |
| dev-MVI_4212-fast3 | MVI_4212 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4212-slow3 | MVI_4212 | 出片✅ | 3.24s/3s | 8.0% | 1 | 3.24s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4438-base3 | MVI_4438 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | - |
| dev-MVI_4438-fast3 | MVI_4438 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4438-slow3 | MVI_4438 | 出片✅ | 3.24s/3s | 8.0% | 1 | 3.24s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4446-base6 | MVI_4446 | 出片✅ | 5.52s/6s | 8.0% | 3 | 1.84s | 0 | no | - |
| dev-MVI_4446-fast6 | MVI_4446 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| dev-MVI_4446-slow6 | MVI_4446 | 出片✅ | 6.48s/6s | 8.0% | 3 | 2.16s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4437-base4 | MVI_4437 | 出片✅ | 3.68s/4s | 8.0% | 1 | 3.68s | 0 | no | - |
| dev-MVI_4437-fast4 | MVI_4437 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| dev-MVI_4437-slow4 | MVI_4437 | 出片✅ | 4.32s/4s | 8.0% | 1 | 4.32s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4400-base5 | MVI_4400 | 出片✅ | 4.6s/5s | 8.0% | 3 | 1.533s | 0 | no | - |
| dev-MVI_4400-fast5 | MVI_4400 | 出片✅ | 4.6s/5s | 8.0% | 3 | 1.533s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4400-slow5 | MVI_4400 | 出片✅ | 5.4s/5s | 8.0% | 3 | 1.8s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4451-base6 | MVI_4451 | 出片✅ | 5.52s/6s | 8.0% | 3 | 1.84s | 0 | no | - |
| dev-MVI_4451-fast6 | MVI_4451 | 出片✅ | 5.52s/6s | 8.0% | 3 | 1.84s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4451-slow6 | MVI_4451 | 出片✅ | 6.48s/6s | 8.0% | 3 | 2.16s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4406-base2 | MVI_4406 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | - |
| dev-MVI_4406-fast2 | MVI_4406 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | 快剪风格，避免模糊镜头 |
| dev-MVI_4406-slow2 | MVI_4406 | 出片✅ | 2.16s/2s | 8.0% | 1 | 2.16s | 0 | no | 慢节奏，长镜头 |
| dev-MVI_4241-base4 | MVI_4241 | 出片✅ | 3.68s/4s | 8.0% | 1 | 3.68s | 0 | no | - |
| dev-MVI_4241-fast4 | MVI_4241 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| dev-MVI_4241-slow4 | MVI_4241 | 出片✅ | 4.32s/4s | 8.0% | 1 | 4.32s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4452-base1 | MVI_4452 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| holdout-MVI_4452-fast1 | MVI_4452 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4452-slow1 | MVI_4452 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| holdout-MVI_4327-base2 | MVI_4327 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | - |
| holdout-MVI_4327-fast2 | MVI_4327 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4327-slow2 | MVI_4327 | 出片✅ | 2.16s/2s | 8.0% | 1 | 2.16s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4209-base5 | MVI_4209 | 出片✅ | 4.6s/5s | 8.0% | 2 | 2.3s | 0 | no | - |
| holdout-MVI_4209-fast5 | MVI_4209 | 出片✅ | 4.6s/5s | 8.0% | 2 | 2.3s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4209-slow5 | MVI_4209 | 出片✅ | 5.4s/5s | 8.0% | 2 | 2.7s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4341-base6 | MVI_4341 | 出片✅ | 5.52s/6s | 8.0% | 3 | 1.84s | 0 | no | - |
| holdout-MVI_4341-fast6 | MVI_4341 | **拒绝** | - | - | - | - | - | - | 全部 3 个候选镜头均违反 must_avoid 约束（['模糊']）——素材无法满足用户约束，拒绝导演（fail-closed）；请更换素材或调整意图 |
| holdout-MVI_4341-slow6 | MVI_4341 | 出片✅ | 6.48s/6s | 8.0% | 3 | 2.16s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4467-base6 | MVI_4467 | 出片✅ | 5.52s/6s | 8.0% | 3 | 1.84s | 0 | no | - |
| holdout-MVI_4467-fast6 | MVI_4467 | 出片✅ | 5.52s/6s | 8.0% | 3 | 1.84s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4467-slow6 | MVI_4467 | 出片✅ | 6.48s/6s | 8.0% | 3 | 2.16s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4414-base3 | MVI_4414 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | - |
| holdout-MVI_4414-fast3 | MVI_4414 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4414-slow3 | MVI_4414 | 出片✅ | 3.24s/3s | 8.0% | 1 | 3.24s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4447-base9 | MVI_4447 | 出片✅ | 9.0s/9s | 0.0% | 4 | 2.25s | 0 | no | - |
| holdout-MVI_4447-fast9 | MVI_4447 | 出片✅ | 8.28s/9s | 8.0% | 3 | 2.76s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4447-slow9 | MVI_4447 | 出片✅ | 9.72s/9s | 8.0% | 4 | 2.43s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4394-base2 | MVI_4394 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | - |
| holdout-MVI_4394-fast2 | MVI_4394 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4394-slow2 | MVI_4394 | 出片✅ | 2.16s/2s | 8.0% | 1 | 2.16s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4265-base3 | MVI_4265 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | - |
| holdout-MVI_4265-fast3 | MVI_4265 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4265-slow3 | MVI_4265 | 出片✅ | 3.24s/3s | 8.0% | 1 | 3.24s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4339-base2 | MVI_4339 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | - |
| holdout-MVI_4339-fast2 | MVI_4339 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4339-slow2 | MVI_4339 | 出片✅ | 2.16s/2s | 8.0% | 1 | 2.16s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4255-base1 | MVI_4255 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| holdout-MVI_4255-fast1 | MVI_4255 | **拒绝** | - | - | - | - | - | - | 全部 1 个候选镜头均违反 must_avoid 约束（['模糊']）——素材无法满足用户约束，拒绝导演（fail-closed）；请更换素材或调整意图 |
| holdout-MVI_4255-slow1 | MVI_4255 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| holdout-MVI_4230-base1 | MVI_4230 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| holdout-MVI_4230-fast1 | MVI_4230 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4230-slow1 | MVI_4230 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| holdout-MVI_4397-base1 | MVI_4397 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| holdout-MVI_4397-fast1 | MVI_4397 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4397-slow1 | MVI_4397 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| holdout-MVI_4389-base5 | MVI_4389 | 出片✅ | 4.6s/5s | 8.0% | 3 | 1.533s | 0 | no | - |
| holdout-MVI_4389-fast5 | MVI_4389 | 出片✅ | 4.6s/5s | 8.0% | 3 | 1.533s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4389-slow5 | MVI_4389 | 出片✅ | 5.4s/5s | 8.0% | 3 | 1.8s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4261-base1 | MVI_4261 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| holdout-MVI_4261-fast1 | MVI_4261 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4261-slow1 | MVI_4261 | **拒绝** | - | - | - | - | - | - | shot_too_short_unfixable |
| holdout-MVI_4224-base2 | MVI_4224 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | - |
| holdout-MVI_4224-fast2 | MVI_4224 | 出片✅ | 1.84s/2s | 8.0% | 1 | 1.84s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4224-slow2 | MVI_4224 | 出片✅ | 2.16s/2s | 8.0% | 1 | 2.16s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4372-base1 | MVI_4372 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| holdout-MVI_4372-fast1 | MVI_4372 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4372-slow1 | MVI_4372 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| holdout-MVI_4328-base1 | MVI_4328 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | - |
| holdout-MVI_4328-fast1 | MVI_4328 | 出片✅ | 0.92s/1s | 8.0% | 1 | 0.92s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4328-slow1 | MVI_4328 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| holdout-MVI_4246-base3 | MVI_4246 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | - |
| holdout-MVI_4246-fast3 | MVI_4246 | 出片✅ | 2.76s/3s | 8.0% | 1 | 2.76s | 0 | no | 快剪风格，避免模糊镜头 |
| holdout-MVI_4246-slow3 | MVI_4246 | 出片✅ | 3.24s/3s | 8.0% | 1 | 3.24s | 0 | no | 慢节奏，长镜头 |
| holdout-MVI_4458-base7 | MVI_4458 | 出片✅ | 6.44s/7s | 8.0% | 3 | 2.147s | 0 | no | - |
| holdout-MVI_4458-fast7 | MVI_4458 | **拒绝** | - | - | - | - | - | - | target_duration_unreachable |
| holdout-MVI_4458-slow7 | MVI_4458 | 出片✅ | 7.56s/7s | 8.0% | 3 | 2.52s | 0 | no | 慢节奏，长镜头 |

> 拒绝 = fail-closed 生效（素材/约束无法满足目标，系统宁可拒绝也不硬凑）。拒绝原因是 L2 的重要产出：它是素材鲁棒性画像的一部分。