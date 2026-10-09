"""M1.1 确定性技术分析：基于 OpenCV 的镜头中点帧指标计算。

对每个镜头读取中点帧（与前 200ms 一帧对比计算抖动），产出 7 项数值指标并封装为
``FilmObservation``（``claim_kind=MEASURED``）。读帧失败或分析异常时返回
``claim_kind=NOT_DETERMINED`` 的观测，不抛异常。
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

from director_brain.models.film_observation import (
    ClaimKind,
    FilmObservation,
    TimebaseUnit,
)

DETERMINISTIC_ANALYSIS_VERSION = "v1.1"
DETERMINISTIC_SAMPLE_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
DETERMINISTIC_SHAKE_LOOKBACK_US = 200_000
OPENCV_RUNTIME_VERSION = str(cv2.__version__)


def _load_frame_at(video_path: str, time_us: int):
    """读取指定时刻（微秒）的一帧；失败返回 None。"""
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_us / 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


def _build_obs(
    video_path: str,
    shot: dict,
    claim: str,
    claim_kind: ClaimKind,
    confidence: float,
) -> FilmObservation:
    """按规范组装 FilmObservation（含公共字段）。"""
    shot_id = shot["shot_id"]
    return FilmObservation(
        observation_id=f"det_{shot_id}",
        media_asset_id=shot_id,
        media_hash=shot["source_media_hash"],
        start_frame=int(shot["source_in_us"]),
        end_frame=int(shot["source_out_us"]),
        timebase=1_000_000,
        timebase_unit=TimebaseUnit.MICROSECONDS,
        observation_type="deterministic_technical",
        claim=claim,
        provider="deterministic_opencv",
        model_version="opencv_5.0",
        prompt_version="n/a",
        confidence=confidence,
        review_state="auto_verified",
        claim_kind=claim_kind,
        schema_version="1.0",
        project_id="unknown",
        created_at=int(time.time()),
        producer="deterministic_opencv",
        source_ref=video_path,
    )


def _fail_obs(video_path: str, shot: dict) -> FilmObservation:
    """读帧/分析失败时的兜底观测。"""
    return _build_obs(
        video_path, shot,
        claim=json.dumps({"error": "no_frame"}),
        claim_kind=ClaimKind.NOT_DETERMINED,
        confidence=0.0,
    )


def analyze_shot(video_path: str, shot: dict) -> FilmObservation:
    """对单个镜头做三点采样确定性技术指标。

    采样点：镜头时长的 25% / 50% / 75% 三处（P0 修复：单点中点采样对
    镜头内黑场/淡入淡出完全盲——3 秒整段黑场的中点如果恰好不在黑区，
    brightness_mean 看起来正常但成片里有 3 秒黑洞）。

    指标：blur_score / brightness_mean / exposure_ok / shake_score /
    dark_sample_ratio（亮度 < 15/255 的采样占比）/ dominant_hue /
    saturation_mean / center_weight。
    """
    in_us = int(shot["source_in_us"])
    out_us = int(shot["source_out_us"])
    dur = out_us - in_us
    # route-10（部分落地）：五点采样——P0 三点（25/50/75%）的实测教训是
    # fade 尾巴逃过测量（语义 pilot 黑帧 3 段）；五点把首尾 fade 纳入
    # dark_ratio/brightness 的度量范围（排除阈值 ≥0.5 不变，策略待拍板）。
    sample_times = [
        in_us + int(dur * f) for f in DETERMINISTIC_SAMPLE_FRACTIONS
    ]

    try:
        frames = []
        for t in sample_times:
            frame = _load_frame_at(video_path, t)
            if frame is not None:
                frames.append(frame)
        if not frames:
            return _fail_obs(video_path, shot)

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)

        # 多点亮度：均值 + 暗帧占比（亮度 < 15/255 视为近黑）
        brightnesses = [float(g.mean()) for g in grays]
        brightness = sum(brightnesses) / len(brightnesses)
        dark_count = sum(1 for b in brightnesses if b < 15.0)
        dark_ratio = dark_count / len(brightnesses)

        # 模糊度：多点 Laplacian 方差取中位数（比单点更稳）
        blurs = [float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in grays]
        blur = sorted(blurs)[len(blurs) // 2]

        # 曝光：多点均值
        exposure_ok = bool(40 < brightness < 215)

        # 抖动：首采样点与前 200ms 帧的 absdiff 均值
        shake = 0.0
        earlier = _load_frame_at(
            video_path,
            max(in_us, sample_times[0] - DETERMINISTIC_SHAKE_LOOKBACK_US),
        )
        if earlier is not None:
            g2 = cv2.cvtColor(earlier, cv2.COLOR_BGR2GRAY)
            if g2.shape == grays[0].shape:
                shake = float(np.mean(cv2.absdiff(grays[0], g2)))

        # 色彩：HSV 均值（中点帧）
        hue = float(hsv[:, :, 0].mean())
        sat = float(hsv[:, :, 1].mean())

        # 中心权重：中点帧
        h, w = grays[0].shape
        cy0, cy1 = h // 3, 2 * h // 3
        cx0, cx1 = w // 3, 2 * w // 3
        center = grays[0][cy0:cy1, cx0:cx1].astype(float)
        whole = grays[0].astype(float)
        center_weight = float(center.mean()) / max(float(whole.mean()), 1.0)

        metrics = {
            "blur_score": round(blur, 2),
            "brightness_mean": round(brightness, 1),
            "exposure_ok": exposure_ok,
            "dark_sample_ratio": round(dark_ratio, 2),
            "shake_score": round(shake, 2),
            "dominant_hue": round(hue, 1),
            "saturation_mean": round(sat, 1),
            "center_weight": round(center_weight, 3),
        }
        return _build_obs(
            video_path, shot,
            claim=json.dumps(metrics),
            claim_kind=ClaimKind.MEASURED,
            confidence=1.0,
        )
    except Exception:
        return _fail_obs(video_path, shot)
