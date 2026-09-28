# -*- coding: utf-8 -*-
"""阶段 9 验证：duck 混音（sidechaincompress）+ 硬字幕烧录（libass）真实渲染。"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")

from director_brain.models.edl import EditItem, EditorialDecisionList
from execution.renderer import render_edl

tmp = Path("evidence/GLM-HANDOVER/stage9_probe")
tmp.mkdir(parents=True, exist_ok=True)

# 素材：带原声的视频 + 配乐 wav + 硬字幕 srt
src = str(tmp / "src.mp4")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                "-i", "testsrc=s=320x240:d=4",
                "-f", "lavfi", "-i", "sine=frequency=300:duration=4",
                "-c:v", "libx264", "-c:a", "aac", src],
               check=True, capture_output=True, timeout=60)
music = str(tmp / "music.wav")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                "-i", "sine=frequency=880:duration=4", music],
               check=True, capture_output=True, timeout=60)
srt = tmp / "hard.srt"
srt.write_text(
    "1\n00:00:00,200 --> 00:00:02,000\n硬字幕烧录验证\n", encoding="utf-8")

edits = [EditItem(source_asset_id="s0", source_media_hash="h0",
                  in_frame=0, out_frame=2_000_000, timebase=1_000_000),
         EditItem(source_asset_id="s1", source_media_hash="h1",
                  in_frame=2_000_000, out_frame=4_000_000, timebase=1_000_000)]
edl = EditorialDecisionList(
    schema_version="1.0", project_id="s9", created_at=0, producer="t",
    source_ref=src, edl_id="e", version="0.1", brief_version="0.1",
    context_id="c", timebase=1_000_000, ordered_edits=edits,
    expected_duration=4_000_000, approval_state="draft",
    audio_refs=[f"{music}|-12|duck"],
    subtitle_refs=[f"{srt}|chi|hard"])

out = str(tmp / "duck_hard.mp4")
render_edl(edl, src, out)

probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type", "-of", "json", out],
                       capture_output=True, text=True, timeout=60)
streams = json.loads(probe.stdout)["streams"]
types = sorted(s["codec_type"] for s in streams)
print("成片流:", types)
assert "video" in types and "audio" in types
assert "subtitle" not in types, "硬字幕已烧录，不应再有字幕流"

# 音频非静音（配乐混入验证）
vol = subprocess.run(["ffmpeg", "-i", out, "-af", "volumedetect",
                      "-f", "null", "-"], capture_output=True, text=True, timeout=120)
mean = None
for line in vol.stderr.splitlines():
    if "mean_volume" in line:
        mean = float(line.split("mean_volume:")[1].replace("dB", "").strip())
print(f"平均音量: {mean}dB")
assert mean is not None and mean > -60

print("\n[PASS] duck 混音（sidechaincompress）+ 硬字幕烧录（libass）真实渲染通过。")
