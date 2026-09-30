"""roughcut --semantic 生产入口测试（候选①）。

覆盖：--semantic 在通路非 ACTIVE 时 fail-closed 拒绝（禁止半消费）；
ACTIVE 时语义观测进入生产链（以 mock VLM 批量分析验证接线，不发真实
HTTP）。默认（无 --semantic）行为与旧路径完全一致。
"""
from __future__ import annotations

import json
import time

import pytest

from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.pathway_protocol import PathwayStatus, set_pathway_status
from scripts.roughcut import run_roughcut


def _tech_obs(idx: int, start_us: int, end_us: int, blur: float = 150.0) -> FilmObservation:
    return FilmObservation(
        observation_id=f"det_{idx}", media_asset_id=f"shot_{idx:08d}",
        media_hash=f"hash_{idx}", start_frame=start_us, end_frame=end_us,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim=json.dumps({"blur_score": blur, "brightness_mean": 120.0,
                          "exposure_ok": True, "shake_score": 5.0}),
        provider="deterministic_opencv", model_version="opencv",
        prompt_version="n/a", confidence=1.0, review_state="auto_verified",
        claim_kind=ClaimKind.MEASURED, schema_version="1.0",
        project_id="t", created_at=int(time.time()),
        producer="deterministic_opencv", source_ref="t.mp4")


@pytest.fixture()
def real_video(tmp_path):
    """ffmpeg testsrc 真实小视频（analyze_media 可跑）。"""
    import subprocess
    path = str(tmp_path / "sem_test.mp4")
    subprocess.run(
        ["ffmpeg", "-f", "lavfi",
         "-i", "testsrc=duration=6:size=320x240:rate=25",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path, "-y"],
        capture_output=True, text=True, check=True)
    return path


def test_semantic_flag_refused_when_pathway_not_active(real_video, tmp_path, capsys):
    """通路 SHADOW/EXPERIMENTAL + --semantic → exit 1 + 响亮说明。"""
    set_pathway_status("vlm_semantic", PathwayStatus.SHADOW)
    try:
        rc = run_roughcut(
            input_path=real_video,
            output_path=str(tmp_path / "out.mp4"),
            target_duration=4,
            dry_run=True,
            semantic=True,
        )
        assert rc == 1
        out = capsys.readouterr().out
        assert "fail-closed" in out
        assert "vlm_semantic" in out
    finally:
        set_pathway_status("vlm_semantic", PathwayStatus.EXPERIMENTAL)


def test_semantic_flag_wires_vlm_batch_when_active(real_video, tmp_path, capsys, monkeypatch):
    """通路 ACTIVE + --semantic → 生产链带语义观测出片（mock 批量分析）。"""
    import observation_service.vlm_observation as mod

    def fake_batch(video_path, shots, adapter=None, cache=None):
        return [
            FilmObservation(
                observation_id=f"vlm_{s['shot_id']}",
                media_asset_id=s["shot_id"],
                media_hash=s["source_media_hash"],
                start_frame=s["source_in_us"], end_frame=s["source_out_us"],
                timebase=1_000_000, observation_type="vlm_semantic",
                claim=json.dumps({
                    "shot_function": "ACTION", "proposed_role_v2": "support",
                    "motion_amount": "subtle", "frame_description": "t",
                    "importance": 5, "narrative_role": "climax",
                    "emotional_tone": "tense", "action_type": "action",
                    "scene_description": "测试语义"}),
                provider="mock", model_version="mock", prompt_version="v3",
                confidence=0.7, review_state="auto_generated",
                claim_kind=ClaimKind.MODEL_OBSERVATION, schema_version="1.0",
                project_id="t", created_at=int(time.time()),
                producer="mock", source_ref=video_path)
            for s in shots
        ]

    monkeypatch.setattr(mod, "batch_vlm_observations", fake_batch)
    # 无 ARK key：叙事层响亮跳过（逐镜头语义仍生效）
    monkeypatch.setenv("ARK_API_KEY", "")
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # 无 .env 可读

    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    try:
        rc = run_roughcut(
            input_path=real_video,
            output_path=str(tmp_path / "out.mp4"),
            target_duration=4,
            dry_run=True,  # 不渲染，只验证语义接线进内核
            semantic=True,
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "语义观测" in out
        assert "EDL 摘要" in out
    finally:
        set_pathway_status("vlm_semantic", PathwayStatus.EXPERIMENTAL)
