"""M2.1 Brief Compiler：从观测自动编译 DirectorBrief。

从 ``analyze_media`` 产出的 deterministic_technical 观测与可选的
speech_transcript（ASR）观测中提取总时长、技术质量分布与语音文本，填充
:class:`~director_brain.models.director_brief.DirectorBrief` 的全部字段。

无法从观测可靠推断的字段统一填 ``"not_determined"`` 或空列表，保持
fail-soft 语义——观测缺失/claim 解析失败都不抛异常。
"""
from __future__ import annotations

import json
import re
import time

from director_brain._utils import short_hash
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.film_observation import FilmObservation

PRODUCER = "brief_compiler_v0.1"
DEFAULT_TARGET_DURATION_US = 15_000_000  # 15s

#: shake_score 均值超过该阈值时打上 high_motion 主题。
_HIGH_MOTION_SHAKE_THRESHOLD = 0.3
#: exposure_ok 通过率低于该值时判定为 mixed_exposure。
_MIXED_EXPOSURE_THRESHOLD = 0.8


def _safe_load_claim(claim: str) -> dict:
    """解析观测 claim JSON；失败或非 dict 时返回空 dict。"""
    try:
        data = json.loads(claim)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_intent(intent_text: str) -> dict:
    """从用户自然语言意图中规则提取语义字段。

    纯关键词+正则匹配，不调用 LLM。无法匹配的字段保持 ``"not_determined"``
    或空列表（fail-soft）。
    """
    text = intent_text or ""

    # ---- language：按中文字符占比 ----
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    has_latin = any(ch.isascii() and ch.isalpha() for ch in text)
    if cjk and cjk / max(len(text.strip()), 1) > 0.3:
        language = "zh"
    elif has_latin:
        language = "en"
    else:
        language = "not_determined"

    def _first(mapping: list[tuple[str, str]]) -> str:
        for keys, value in mapping:
            for k in keys:
                if k in text:
                    return value
        return "not_determined"

    audience = _first([
        (("朋友", "朋友圈"), "friends"),
        (("家人", "家庭", "亲子"), "family"),
        (("客户", "甲方", "商务", "B端"), "client"),
        (("公众", "公开", "发布", "全网"), "public"),
        (("孩子", "儿童", "少儿"), "children"),
    ])
    delivery_profile = _first([
        (("短视频", "抖音", "快手", "小红书", "竖屏"), "short_form"),
        (("vlog", "日志", "记录"), "vlog"),
        (("纪录片", "纪录"), "documentary"),
        (("广告", "宣传片", "品牌", "产品"), "commercial"),
        (("电影", "微电影", "院线"), "cinematic"),
    ])
    emotional_arc = _first([
        (("快节奏", "高能", "燃", "热血", "激昂", "亢奋"), "upbeat"),
        (("感人", "温情", "温暖", "治愈", "暖心"), "heartwarming"),
        (("悬疑", "紧张", "惊悚", "烧脑"), "suspenseful"),
        (("平静", "舒缓", "安静", "宁静"), "calm"),
        (("悲伤", "伤感", "难过", "忧伤"), "melancholic"),
    ])
    editing_language = _first([
        (("快剪", "快切", "快速剪辑", "闪切"), "fast_cut"),
        (("慢节奏", "慢剪", "长镜头", "慢镜头"), "slow_paced"),
        (("蒙太奇",), "montage"),
        (("跳切",), "jump_cut"),
    ])

    def _extract(patterns: list[str]) -> list[str]:
        out: list[str] = []
        for pat in patterns:
            for m in re.finditer(pat, text):
                item = m.group(1).strip(" ，。！？、,.;；")
                if item:
                    out.append(item)
        return out

    must_include = _extract([
        r"必须包含(.+?)(?:[，。！？\n]|$)",
        r"一定要有(.+?)(?:[，。！？\n]|$)",
        r"要有(.+?)(?:[，。！？\n]|$)",
        r"必须出现(.+?)(?:[，。！？\n]|$)",
    ])
    must_avoid = _extract([
        r"避免出现(.+?)(?:[，。！？\n]|$)",
        r"避免(.+?)(?:[，。！？\n]|$)",
        r"不要(.+?)(?:[，。！？\n]|$)",
        r"不能有(.+?)(?:[，。！？\n]|$)",
        r"禁止(.+?)(?:[，。！？\n]|$)",
    ])

    return {
        "language": language,
        "audience": audience,
        "delivery_profile": delivery_profile,
        "emotional_arc": emotional_arc,
        "editing_language": editing_language,
        "must_include": must_include,
        "must_avoid": must_avoid,
    }


def compile_brief(
    project_id: str,
    video_path: str,
    observations: list[FilmObservation],
    target_duration_us: int | None = None,
    intent_text: str | None = None,
) -> DirectorBrief:
    """从观测自动编译一份导演简报（draft 状态）。

    Args:
        project_id: 项目 ID，写入 Brief 顶层记录字段。
        video_path: 源视频路径，写入 ``source_ref``。
        observations: ``analyze_media`` 的输出列表，可附加 ASR 产出的
            ``speech_transcript`` 观测。
        target_duration_us: 目标成片时长（微秒）。传入时用传入值；未传入时
            用 :data:`DEFAULT_TARGET_DURATION_US`。**禁止**用源素材时长覆盖。
        intent_text: 用户自然语言意图描述。传入时用规则提取（非 LLM）解析
            language/audience/delivery_profile/emotional_arc/editing_language/
            must_include/must_avoid；未传入（或空串）时这些字段保持
            ``"not_determined"``/空列表，行为与未接入意图前一致。

    Returns:
        填充完毕的 :class:`DirectorBrief`，``approval_state="draft"``。
    """
    tech_obs = sorted(
        [o for o in observations if o.observation_type == "deterministic_technical"],
        key=lambda o: o.start_frame,
    )
    speech_obs = sorted(
        [o for o in observations if o.observation_type == "speech_transcript"],
        key=lambda o: o.start_frame,
    )

    # ---- 源素材时长 vs 目标成片时长（二者语义不同，不可混用）----
    source_duration_us = tech_obs[-1].end_frame if tech_obs else 0
    target = target_duration_us if target_duration_us is not None else DEFAULT_TARGET_DURATION_US

    # ---- 技术质量分布 ----
    blur_scores: list[float] = []
    shake_scores: list[float] = []
    exposure_ok_count = 0
    exposure_total = 0
    for o in tech_obs:
        data = _safe_load_claim(o.claim)
        if "blur_score" in data and isinstance(data["blur_score"], (int, float)):
            blur_scores.append(float(data["blur_score"]))
        if "shake_score" in data and isinstance(data["shake_score"], (int, float)):
            shake_scores.append(float(data["shake_score"]))
        if "exposure_ok" in data:
            exposure_total += 1
            if data["exposure_ok"]:
                exposure_ok_count += 1

    # ---- themes / visual_language ----
    themes: list[str] = []
    if shake_scores:
        mean_shake = sum(shake_scores) / len(shake_scores)
        if mean_shake >= _HIGH_MOTION_SHAKE_THRESHOLD:
            themes.append("high_motion")

    if exposure_total > 0:
        exposure_pass_rate = exposure_ok_count / exposure_total
        if exposure_pass_rate < _MIXED_EXPOSURE_THRESHOLD:
            visual_language = "mixed_exposure"
        else:
            visual_language = "consistent_exposure"
    else:
        visual_language = "not_determined"

    # ---- 语音 ----
    if speech_obs:
        source_text = " ".join((o.claim or "").strip() for o in speech_obs if o.claim)
        sound_language = "speech_present"
    else:
        source_text = "no_speech_detected"
        sound_language = "no_speech"

    brief_id = f"brief_{project_id}_{short_hash(video_path)}"

    # ---- 用户意图解析（规则提取，fail-soft）----
    intent_fields = {
        "language": "not_determined", "audience": "not_determined",
        "delivery_profile": "not_determined", "emotional_arc": "not_determined",
        "editing_language": "not_determined",
        "must_include": [], "must_avoid": [],
    }
    if intent_text:
        intent_fields.update(_parse_intent(intent_text))

    return DirectorBrief(
        schema_version="1.0",
        project_id=project_id,
        created_at=int(time.time()),
        producer=PRODUCER,
        source_ref=video_path,
        brief_id=brief_id,
        version="0.1",
        source_text=source_text,
        language=intent_fields["language"],
        intent="user_provided" if intent_text else "auto_compiled_from_observations",
        audience=intent_fields["audience"],
        target_duration=target,
        source_duration_us=source_duration_us,
        delivery_profile=intent_fields["delivery_profile"],
        themes=themes,
        relationships=[],
        emotional_arc=intent_fields["emotional_arc"],
        visual_language=visual_language,
        editing_language=intent_fields["editing_language"],
        sound_language=sound_language,
        must_include=intent_fields["must_include"],
        must_avoid=intent_fields["must_avoid"],
        privacy_constraints=[],
        approval_state="draft",
        approved_by=None,
        approved_at=None,
    )
