"""Opt-in local shadow retrieval over caller-selected cross-asset mentions.

This ranks every eligible candidate in the selected anchor scope. The score is
an uncalibrated text-embedding cosine similarity, not identity confidence or a
link decision. Candidate descriptions/evidence stay on the configured loopback
Ollama service; no candidate is pruned and no relationship is created.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Sequence

from director_brain.models.project_story_link_candidates import (
    ProjectStoryLinkCandidate,
)


RANKING_PROFILE_ID = "bge_m3_shadow_v1"
RANKING_MODEL = "bge-m3:latest"
RANKING_METRIC = "cosine_similarity"
RANKING_STATE = "shadow_ranked_unadmitted"
EMBED_BATCH_SIZE = 64


class ProjectStoryLinkRankingError(RuntimeError):
    """A fail-closed local ranking provider or output error."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProjectStoryLinkRankingError("provider_redirect_rejected")


_LOCAL_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    _RejectRedirectHandler(),
)


@dataclass(frozen=True)
class ProjectStoryLinkRankingResult:
    candidates: list[ProjectStoryLinkCandidate]
    model_digest: str
    provider_invocation_count: int
    embedding_invocation_count: int


def _loopback_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ProjectStoryLinkRankingError("non_loopback_provider_rejected")
    return f"http://{parsed.netloc}"


def _read_json(url: str, *, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with _LOCAL_OPENER.open(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ProjectStoryLinkRankingError("provider_request_rejected") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProjectStoryLinkRankingError("provider_unavailable") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectStoryLinkRankingError("provider_invalid_response") from exc
    if not isinstance(payload, dict):
        raise ProjectStoryLinkRankingError("provider_invalid_response")
    return payload


def _post_json(url: str, body: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _LOCAL_OPENER.open(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ProjectStoryLinkRankingError("provider_request_rejected") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProjectStoryLinkRankingError("provider_unavailable") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectStoryLinkRankingError("provider_invalid_response") from exc
    if not isinstance(payload, dict):
        raise ProjectStoryLinkRankingError("provider_invalid_response")
    return payload


@dataclass(frozen=True)
class _ModelBinding:
    digest: str
    dimensions: int


def _model_binding(base_url: str) -> _ModelBinding:
    payload = _read_json(f"{base_url}/api/tags", timeout=10.0)
    models = payload.get("models")
    if not isinstance(models, list):
        raise ProjectStoryLinkRankingError("provider_invalid_model_inventory")
    matches = [
        item for item in models
        if isinstance(item, dict) and item.get("name") == RANKING_MODEL
    ]
    if len(matches) != 1:
        raise ProjectStoryLinkRankingError("ranking_model_unavailable")
    item = matches[0]
    digest = item.get("digest")
    details = item.get("details")
    dimensions = details.get("embedding_length") if isinstance(details, dict) else None
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in digest)
        or isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or dimensions < 1
        or "embedding" not in item.get("capabilities", [])
    ):
        raise ProjectStoryLinkRankingError("provider_invalid_model_inventory")
    return _ModelBinding(digest=digest.lower(), dimensions=dimensions)


def resolve_project_story_link_ranking_binding(
    base_url: str,
) -> _ModelBinding:
    """Read the installed local model identity before cache lookup or ranking."""
    return _model_binding(_loopback_base_url(base_url))


def _event_text(candidate: ProjectStoryLinkCandidate, side: str) -> str:
    evidence = (
        candidate.left_event_evidence
        if side == "left"
        else candidate.right_event_evidence
    )
    if evidence is None:
        raise ProjectStoryLinkRankingError("candidate_evidence_missing")
    return json.dumps(
        evidence.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _candidate_text_pair(
    candidate: ProjectStoryLinkCandidate,
) -> tuple[str, str]:
    if candidate.relation_kind == "person_identity":
        left = candidate.left_person_description
        right = candidate.right_person_description
    else:
        left = _event_text(candidate, "left")
        right = _event_text(candidate, "right")
    if (
        not isinstance(left, str)
        or not left.strip()
        or not isinstance(right, str)
        or not right.strip()
    ):
        raise ProjectStoryLinkRankingError("candidate_evidence_missing")
    return left, right


def _embeddings_for_texts(
    texts: Sequence[str],
    *,
    base_url: str,
    binding: _ModelBinding,
) -> tuple[dict[str, tuple[float, ...]], int]:
    unique_texts = list(dict.fromkeys(texts))
    embeddings: dict[str, tuple[float, ...]] = {}
    invocation_count = 0
    for start in range(0, len(unique_texts), EMBED_BATCH_SIZE):
        batch = unique_texts[start:start + EMBED_BATCH_SIZE]
        payload = _post_json(
            f"{base_url}/api/embed",
            {
                "model": RANKING_MODEL,
                "input": batch,
                "truncate": False,
                "keep_alive": "5m",
            },
            timeout=120.0,
        )
        invocation_count += 1
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise ProjectStoryLinkRankingError("provider_invalid_embeddings")
        for text, vector in zip(batch, vectors, strict=True):
            if (
                not isinstance(vector, list)
                or len(vector) != binding.dimensions
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in vector
                )
            ):
                raise ProjectStoryLinkRankingError("provider_invalid_embeddings")
            norm = math.sqrt(math.fsum(float(value) ** 2 for value in vector))
            if norm <= 0:
                raise ProjectStoryLinkRankingError("provider_invalid_embeddings")
            embeddings[text] = tuple(float(value) for value in vector)
    return embeddings, invocation_count


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0 or len(left) != len(right):
        raise ProjectStoryLinkRankingError("provider_invalid_embeddings")
    score = math.fsum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    if not math.isfinite(score):
        raise ProjectStoryLinkRankingError("provider_invalid_embeddings")
    return max(-1.0, min(1.0, score))


def ranked_candidate_set_id(unranked_candidate_set_id: str, model_digest: str) -> str:
    """Bind pagination to the exhaustive scope, profile, and installed weights."""
    payload = "\n".join((
        unranked_candidate_set_id,
        RANKING_PROFILE_ID,
        RANKING_MODEL,
        model_digest.lower(),
        RANKING_METRIC,
    ))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"ranked_link_candidate_set_{digest}"


def project_story_link_ranking_snapshot_entries(
    candidates: Sequence[ProjectStoryLinkCandidate],
) -> list[tuple[str, float]]:
    """Extract a validated immutable ordering from a completed rank result."""
    ordered = sorted(
        candidates,
        key=lambda item: item.ranking_position
        if item.ranking_position is not None else 0,
    )
    if any(
        item.ranking_state != RANKING_STATE
        or item.ranking_position != position
        or item.ranking_score is None
        or not math.isfinite(item.ranking_score)
        or not -1.0 <= item.ranking_score <= 1.0
        for position, item in enumerate(ordered, start=1)
    ):
        raise ProjectStoryLinkRankingError("ranking_result_invalid")
    candidate_ids = [item.candidate_id for item in ordered]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ProjectStoryLinkRankingError("ranking_result_invalid")
    return [(item.candidate_id, float(item.ranking_score)) for item in ordered]


def apply_project_story_link_ranking_snapshot(
    candidates: Sequence[ProjectStoryLinkCandidate],
    entries: Sequence[tuple[str, float]],
) -> list[ProjectStoryLinkCandidate]:
    """Apply a stored score order only to its exact current candidate scope."""
    candidate_by_id = {item.candidate_id: item for item in candidates}
    if len(candidate_by_id) != len(candidates) or len(entries) != len(candidates):
        raise ProjectStoryLinkRankingError("ranking_cache_scope_mismatch")
    ranked: list[ProjectStoryLinkCandidate] = []
    seen: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        if (
            not isinstance(entry, tuple)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or entry[0] in seen
            or entry[0] not in candidate_by_id
            or isinstance(entry[1], bool)
            or not isinstance(entry[1], (int, float))
            or not math.isfinite(entry[1])
            or not -1.0 <= entry[1] <= 1.0
        ):
            raise ProjectStoryLinkRankingError("ranking_cache_invalid")
        candidate_id, score = entry
        seen.add(candidate_id)
        ranked.append(candidate_by_id[candidate_id].model_copy(update={
            "ranking_state": RANKING_STATE,
            "ranking_position": position,
            "ranking_score": float(score),
        }))
    if seen != set(candidate_by_id):
        raise ProjectStoryLinkRankingError("ranking_cache_scope_mismatch")
    return ranked


def rank_project_story_link_candidates(
    candidates: Sequence[ProjectStoryLinkCandidate],
    *,
    base_url: str,
    binding: _ModelBinding | None = None,
) -> ProjectStoryLinkRankingResult:
    """Rank every supplied pair from its stored text evidence, without pruning."""
    url = _loopback_base_url(base_url)
    binding = binding or _model_binding(url)
    if candidates:
        text_pairs = [_candidate_text_pair(item) for item in candidates]
        all_texts = [text for pair in text_pairs for text in pair]
        embeddings, invocation_count = _embeddings_for_texts(
            all_texts, base_url=url, binding=binding)
        scored = [
            (
                _cosine(embeddings[left], embeddings[right]),
                candidate,
            )
            for candidate, (left, right) in zip(candidates, text_pairs, strict=True)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].candidate_id))
        ranked = [
            candidate.model_copy(update={
                "ranking_state": RANKING_STATE,
                "ranking_position": position,
                "ranking_score": score,
            })
            for position, (score, candidate) in enumerate(scored, start=1)
        ]
    else:
        ranked = []
        invocation_count = 0
    return ProjectStoryLinkRankingResult(
        candidates=ranked,
        model_digest=binding.digest,
        provider_invocation_count=1 + invocation_count,
        embedding_invocation_count=invocation_count,
    )
