"""scripts/roughcut.py 端到端 CLI 单元测试。

通过 monkeypatch 替换 pipeline 函数验证 CLI 流程，不实际运行 analyze_media / ffmpeg。
模块通过 importlib 动态加载，因此用 monkeypatch.setattr 直接替换模块属性。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# 确保项目根在 path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from director_brain.models.director_brief import DirectorBrief
from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList, EditItem
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.models.story_graph import StoryGraph
from director_brain.plan_repair import (
    REASON_DURATION_UNREACHABLE,
    RepairOutcome,
)


def _load_roughcut():
    """动态导入 roughcut 模块（避免 __main__ 执行）。"""
    spec = importlib.util.spec_from_file_location(
        "roughcut_under_test",
        Path(_PROJECT_ROOT) / "scripts" / "roughcut.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_edl(n_edits: int = 3) -> EditorialDecisionList:
    edits = [
        EditItem(
            source_asset_id=f"shot_{i:03d}",
            source_media_hash=f"hash_{i}",
            in_frame=i * 2_000_000,
            out_frame=(i + 1) * 2_000_000,
            timebase=1_000_000,
            rationale=f"act=hook, heuristic:blur=5.0",
        )
        for i in range(n_edits)
    ]
    return EditorialDecisionList(
        schema_version="1.0", project_id="test", created_at=0,
        producer="test", source_ref="test.mp4",
        edl_id="edl_test", version="0.1", brief_version="0.1",
        context_id="ctx_test", timebase=1_000_000,
        ordered_edits=edits, approval_state="draft",
        expected_duration=n_edits * 2_000_000,
    )


def _make_plan() -> DirectorDecisionPlan:
    return DirectorDecisionPlan(
        schema_version="1.0", project_id="test", created_at=0,
        producer="test", source_ref="test.mp4",
        plan_id="plan_test", version="0.1", brief_version="0.1",
        film_state_version="0.1", validation_status="valid",
        approval_state="draft",
    )


def _make_brief(target_duration_us: int = 15_000_000) -> DirectorBrief:
    return DirectorBrief(
        schema_version="1.0", project_id="test", created_at=0,
        producer="test", source_ref="test.mp4",
        brief_id="brief_test", version="0.1",
        source_text="test", language="not_determined",
        intent="test", audience="not_determined",
        target_duration=target_duration_us,
        source_duration_us=52_000_000,
        delivery_profile="not_determined",
        themes=[], relationships=[],
        emotional_arc="not_determined",
        visual_language="consistent_exposure",
        editing_language="not_determined",
        sound_language="no_speech",
        must_include=[], must_avoid=[], privacy_constraints=[],
        approval_state="draft",
    )


def _make_graph() -> StoryGraph:
    return StoryGraph(
        schema_version="1.0", project_id="test", created_at=0,
        producer="test", source_ref="test.mp4",
        graph_id="graph_test", version="0.1",
        nodes=[], edges=[],
    )


def _make_obs(n: int = 3) -> list[FilmObservation]:
    return [
        FilmObservation(
            observation_id=f"obs_{i}",
            media_asset_id=f"shot_{i:03d}",
            media_hash=f"hash_{i}",
            start_frame=i * 2_000_000,
            end_frame=(i + 1) * 2_000_000,
            timebase=1_000_000,
            observation_type="deterministic_technical",
            claim='{"blur_score": 5.0, "exposure_ok": true}',
            provider="test", model_version="test", prompt_version="test",
            confidence=0.9, review_state="auto_generated",
            claim_kind=ClaimKind.MEASURED,
            schema_version="1.0", project_id="test", created_at=0,
            producer="test", source_ref="test.mp4",
        )
        for i in range(n)
    ]


def _patch_pipeline(monkeypatch, mod, *, validate_result=(True, []), repair_result=None, render_side_effect=None):
    """用 monkeypatch 替换 roughcut 模块中的所有 pipeline 函数。"""
    mock_analyze = MagicMock(return_value=_make_obs())
    mock_transcribe = MagicMock(return_value=[])
    mock_brief = MagicMock(return_value=_make_brief())
    mock_graph = MagicMock(return_value=_make_graph())
    mock_infer = MagicMock(return_value=[])
    mock_reasoner = MagicMock()
    mock_reasoner.generate_plan.return_value = (_make_edl(), _make_plan())
    mock_reasoner_fn = MagicMock(return_value=mock_reasoner)
    mock_validate = MagicMock(return_value=validate_result)
    mock_render = MagicMock()
    if render_side_effect is not None:
        mock_render.side_effect = render_side_effect
    else:
        # 默认：创建输出文件并返回其路径（模拟真实渲染）
        def _fake_render(edl, source_video, output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fake rendered video")
            return output_path
        mock_render.side_effect = _fake_render

    monkeypatch.setattr(mod, "analyze_media", mock_analyze)
    monkeypatch.setattr(mod, "transcribe", mock_transcribe)
    monkeypatch.setattr(mod, "compile_brief", mock_brief)
    monkeypatch.setattr(mod, "build_story_graph", mock_graph)
    monkeypatch.setattr(mod, "infer_relations", mock_infer)
    monkeypatch.setattr(mod, "get_director_reasoner", mock_reasoner_fn)
    monkeypatch.setattr(mod, "validate_plan", mock_validate)
    monkeypatch.setattr(mod, "render_edl", mock_render)

    mocks = {
        "analyze": mock_analyze,
        "transcribe": mock_transcribe,
        "brief": mock_brief,
        "graph": mock_graph,
        "infer": mock_infer,
        "reasoner_fn": mock_reasoner_fn,
        "reasoner": mock_reasoner,
        "validate": mock_validate,
        "render": mock_render,
    }

    if repair_result is not None:
        mock_repair = MagicMock(return_value=repair_result)
        monkeypatch.setattr(mod, "repair_plan", mock_repair)
        mocks["repair"] = mock_repair

    return mocks


class TestRoughcutCLI:
    """roughcut.py CLI 行为测试。"""

    def test_dry_run_does_not_render(self, monkeypatch, tmp_path):
        """--dry-run 不调用 render_edl，不创建输出文件。"""
        mod = _load_roughcut()
        mocks = _patch_pipeline(monkeypatch, mod)

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"fake")
        output_file = tmp_path / "output.mp4"

        rc = mod.run_roughcut(
            input_path=str(input_file),
            output_path=str(output_file),
            target_duration=15,
            dry_run=True,
        )

        assert rc == 0
        mocks["render"].assert_not_called()
        assert not output_file.exists()

    def test_intent_text_forwarded_to_brief_compiler(self, monkeypatch, tmp_path):
        """P0-4 入口：intent_text 透传给 compile_brief（None 表示未提供）。"""
        mod = _load_roughcut()
        mocks = _patch_pipeline(monkeypatch, mod)

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"fake")
        output_file = tmp_path / "output.mp4"

        rc = mod.run_roughcut(
            input_path=str(input_file),
            output_path=str(output_file),
            dry_run=True,
            intent_text="做一个快节奏的短视频，不要模糊镜头",
        )

        assert rc == 0
        _, kwargs = mocks["brief"].call_args
        assert kwargs.get("intent_text") == "做一个快节奏的短视频，不要模糊镜头"

        # 未提供时透传 None（而非缺参）
        mod2 = _load_roughcut()
        mocks2 = _patch_pipeline(monkeypatch, mod2)
        mod2.run_roughcut(
            input_path=str(input_file), output_path=str(output_file), dry_run=True,
        )
        _, kwargs2 = mocks2["brief"].call_args
        assert kwargs2.get("intent_text") is None

    def test_normal_run_calls_render(self, monkeypatch, tmp_path):
        """非 dry-run 调用 render_edl 并返回 0。"""
        mod = _load_roughcut()
        mocks = _patch_pipeline(monkeypatch, mod)

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"fake")
        output_file = tmp_path / "output.mp4"

        rc = mod.run_roughcut(
            input_path=str(input_file),
            output_path=str(output_file),
            target_duration=15,
            dry_run=False,
        )

        assert rc == 0
        mocks["render"].assert_called_once()

    def test_input_not_found_returns_error(self, tmp_path):
        """输入视频不存在时返回非 0，不抛异常。"""
        mod = _load_roughcut()
        rc = mod.run_roughcut(
            input_path=str(tmp_path / "nonexistent.mp4"),
            output_path=str(tmp_path / "out.mp4"),
        )
        assert rc == 1

    def test_validation_failure_triggers_repair(self, monkeypatch, tmp_path):
        """验证失败时调用 repair_plan，物理修复成功后验证通过。"""
        mod = _load_roughcut()
        edl = _make_edl()
        plan = _make_plan()

        # 第一次验证失败，第二次（修复后）成功
        mock_validate = MagicMock(side_effect=[
            (False, ["duration out of range"]),
            (True, []),
        ])
        mock_repair = MagicMock(return_value=RepairOutcome(
            status="ok", edl=edl, plan=plan,
            adjustments=["extend:shot_000:out 2000000->2150000"],
        ))

        mocks = _patch_pipeline(monkeypatch, mod, validate_result=None)
        # 覆盖 validate 和 repair
        monkeypatch.setattr(mod, "validate_plan", mock_validate)
        monkeypatch.setattr(mod, "repair_plan", mock_repair)

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"fake")
        output_file = tmp_path / "out.mp4"

        rc = mod.run_roughcut(
            input_path=str(input_file),
            output_path=str(output_file),
            dry_run=False,
        )

        assert rc == 0
        mock_repair.assert_called_once()
        assert mock_validate.call_count == 2

    def test_repair_abstain_fails_closed_without_render(self, monkeypatch, tmp_path):
        """repair ABSTAIN（repair_requires_director）→ 不渲染，返回非 0。"""
        mod = _load_roughcut()
        edl = _make_edl()
        plan = _make_plan()

        mock_validate = MagicMock(return_value=(False, ["duration out of range"]))
        mock_repair = MagicMock(return_value=RepairOutcome(
            status="abstain",
            reason_code=REASON_DURATION_UNREACHABLE,
            reason="已选镜头物理延长到极限后仍距目标下界差 7.5M us",
        ))

        mocks = _patch_pipeline(monkeypatch, mod, validate_result=None)
        monkeypatch.setattr(mod, "validate_plan", mock_validate)
        monkeypatch.setattr(mod, "repair_plan", mock_repair)

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"fake")
        output_file = tmp_path / "out.mp4"

        rc = mod.run_roughcut(
            input_path=str(input_file),
            output_path=str(output_file),
            dry_run=False,
        )

        assert rc == 1
        mock_repair.assert_called_once()
        mocks["render"].assert_not_called()
        assert not output_file.exists()

    def test_evidence_too_poor_fails_closed(self, monkeypatch, tmp_path):
        """generate_plan 抛 EvidenceTooPoorError → 明确报错、返回非 0、不渲染。"""
        from director_brain.director_reasoner import EvidenceTooPoorError

        mod = _load_roughcut()
        mocks = _patch_pipeline(monkeypatch, mod)
        mocks["reasoner"].generate_plan.side_effect = EvidenceTooPoorError(
            "候选镜头 3 个中仅 0 个曝光合格"
        )

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"fake")
        output_file = tmp_path / "out.mp4"

        rc = mod.run_roughcut(
            input_path=str(input_file),
            output_path=str(output_file),
            dry_run=False,
        )

        assert rc == 1
        mocks["render"].assert_not_called()
        assert not output_file.exists()

    def test_render_failure_returns_error(self, monkeypatch, tmp_path):
        """render_edl 抛 RuntimeError 时返回非 0。"""
        mod = _load_roughcut()

        def _raise(edl, source_video, output_path):
            raise RuntimeError("ffmpeg render failed: mock error")

        mocks = _patch_pipeline(monkeypatch, mod, render_side_effect=_raise)

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"fake")
        output_file = tmp_path / "out.mp4"

        rc = mod.run_roughcut(
            input_path=str(input_file),
            output_path=str(output_file),
            dry_run=False,
        )
        assert rc == 1

    def test_argparse_defaults(self):
        """argparse 默认参数正确。"""
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--input", "-i", required=True)
        p.add_argument("--output", "-o", required=True)
        p.add_argument("--target-duration", type=int, default=15)
        p.add_argument("--dry-run", action="store_true")
        args = p.parse_args(["-i", "in.mp4", "-o", "out.mp4"])
        assert args.target_duration == 15
        assert args.dry_run is False
        args2 = p.parse_args(["-i", "in.mp4", "-o", "out.mp4", "--dry-run", "--target-duration", "30"])
        assert args2.dry_run is True
        assert args2.target_duration == 30
