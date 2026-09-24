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

from director_brain.models.film_observation import ClaimKind, FilmObservation


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
    """对单个镜头计算确定性技术指标。

    指标：blur_score / brightness_mean / exposure_ok / shake_score /
    dominant_hue / saturation_mean / center_weight。
    """
    in_us = int(shot["source_in_us"])
    out_us = int(shot["source_out_us"])
    mid = (in_us + out_us) // 2

    try:
        frame = _load_frame_at(video_path, mid)
        if frame is None:
            return _fail_obs(video_path, shot)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 模糊度：Laplacian 方差（越低越模糊）
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 曝光：灰度均值
        brightness = float(gray.mean())

        # 抖动：中点帧与前 200ms 帧的 absdiff 均值
        earlier = _load_frame_at(video_path, max(in_us, mid - 200_000))
        shake = 0.0
        if earlier is not None:
            g2 = cv2.cvtColor(earlier, cv2.COLOR_BGR2GRAY)
            if g2.shape == gray.shape:
                shake = float(np.mean(cv2.absdiff(gray, g2)))

        # 色彩：HSV 均值
        hue = float(hsv[:, :, 0].mean())
        sat = float(hsv[:, :, 1].mean())

        # 中心权重：中心三分之一区域能量占比
        h, w = gray.shape
        cy0, cy1 = h // 3, 2 * h // 3
        cx0, cx1 = w // 3, 2 * w // 3
        center = gray[cy0:cy1, cx0:cx1].astype(float)
        whole = gray.astype(float)
        center_weight = float(center.mean()) / max(float(whole.mean()), 1.0)

        metrics = {
            "blur_score": round(blur, 2),
            "brightness_mean": round(brightness, 1),
            "exposure_ok": bool(40 < brightness < 215),
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
