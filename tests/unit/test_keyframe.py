from __future__ import annotations

import subprocess
from pathlib import Path


def test_local_preview_extraction_restricts_ffmpeg_to_file_protocol(
    monkeypatch,
):
    from observation_service import keyframe

    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[-2]).write_bytes(b"synthetic-frame")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(keyframe.subprocess, "run", fake_run)
    with keyframe.extract_keyframes(
        "authorized-source.mp4", 0, 1_000_000, local_only=True,
    ) as frames:
        assert len(frames) == 3

    assert len(commands) == 3
    for command in commands:
        assert command[0] == "ffmpeg"
        whitelist_index = command.index("-protocol_whitelist")
        input_index = command.index("-i")
        assert command[whitelist_index + 1] == "file"
        assert whitelist_index < input_index
        assert not any(
            token in command for token in ("http", "https", "tcp", "udp")
        )


def test_default_keyframe_extraction_keeps_existing_protocol_behavior(
    monkeypatch,
):
    from observation_service import keyframe

    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[-2]).write_bytes(b"synthetic-frame")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(keyframe.subprocess, "run", fake_run)
    with keyframe.extract_keyframes("source.mp4", 0, 1_000_000) as frames:
        assert len(frames) == 3

    assert commands
    assert all("-protocol_whitelist" not in command for command in commands)
