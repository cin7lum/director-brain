"""POC evaluation: deterministic field comparison against human reference."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schema import DirectorDecision

POC_DIR = Path(__file__).parent

def load_jsonl(path):
    items = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

benchmark = load_jsonl(POC_DIR / "benchmark.jsonl")
references = {r["id"]: r for r in load_jsonl(POC_DIR / "human_reference.jsonl")}
outputs_raw = load_jsonl(POC_DIR / "model_outputs.jsonl")

# ── Schema validation ──
schema_valid = 0
schema_invalid = []
validated_outputs = {}
for out in outputs_raw:
    try:
        d = DirectorDecision(**out)
        validated_outputs[d.decision_id] = d
        schema_valid += 1
    except Exception as e:
        schema_invalid.append((out.get("decision_id", "?"), str(e)))

schema_valid_rate = schema_valid / len(outputs_raw) if outputs_raw else 0

# ── Tool leakage check ──
import re
FORBIDDEN_TERMS = ["j_cut", "jcut", "l_cut", "lcut", "hold", "reorder",
                   "mcp", "davinci", "otio", "recordframe", "mediatype",
                   "trackindex", "resolve", "ffmpeg"]
# Controlled film vocabulary that contains these substrings but is NOT tool leakage
ALLOWED_CONTEXTS = ["reorder_story_beat", "emotional", "emotional_arc"]
tool_leak_count = 0
tool_leak_details = []
for oid, d in validated_outputs.items():
    core_text = json.dumps({k: v for k, v in d.model_dump().items()
                            if k != "user_terminology"}, ensure_ascii=False).lower()
    leaks = []
    for t in FORBIDDEN_TERMS:
        # word-boundary match, but exclude known film-vocabulary compounds
        pattern = r'(?<![a-z_])' + re.escape(t) + r'(?![a-z_])'
        for m in re.finditer(pattern, core_text):
            start = max(0, m.start() - 20)
            end = min(len(core_text), m.end() + 20)
            context = core_text[start:end]
            if not any(allowed in context for allowed in ALLOWED_CONTEXTS):
                leaks.append(t)
                break
    if leaks:
        tool_leak_count += 1
        tool_leak_details.append((oid, leaks))

# ── Exact parameter hallucination check ──
hallucination_count = 0
hallucination_details = []
for oid, d in validated_outputs.items():
    ref = references.get(oid)
    if not ref:
        continue
    if d.parameterization.exact_value is not None and ref.get("exact_value") is None:
        hallucination_count += 1
        hallucination_details.append((oid, d.parameterization.exact_value))

# ── Positive intent recall ──
positive_correct = 0
positive_total = 0
positive_miss = []
for oid, d in validated_outputs.items():
    ref = references.get(oid)
    if not ref:
        continue
    ref_positive = ref.get("positive_intent")
    if ref_positive:
        positive_total += 1
        # Check if any desired_relation matches the reference positive intent
        model_relations = [r.lower() for r in d.desired_relation_or_change]
        if ref_positive.lower() in model_relations:
            positive_correct += 1
        else:
            positive_miss.append((oid, ref_positive, model_relations))

positive_recall = positive_correct / positive_total if positive_total else 0

# ── Negative constraint recall ──
negative_correct = 0
negative_total = 0
negative_miss = []
for oid, d in validated_outputs.items():
    ref = references.get(oid)
    if not ref:
        continue
    ref_negs = ref.get("negative_constraints", [])
    if ref_negs:
        negative_total += len(ref_negs)
        model_all = ([r.lower() for r in d.desired_relation_or_change] +
                     [m.lower() for m in d.must_avoid] +
                     [p.lower() for p in d.must_preserve])
        for neg in ref_negs:
            neg_key = neg.lower().replace("avoid_", "").replace("preserve_", "")
            if any(neg_key in m for m in model_all):
                negative_correct += 1
            else:
                negative_miss.append((oid, neg))

negative_recall = negative_correct / negative_total if negative_total else 0

# ── Relation direction accuracy (J-cut vs L-cut) ──
direction_correct = 0
direction_total = 0
direction_miss = []
jcut_ids = ["SD-01", "SD-02", "SD-06", "SD-07", "SD-29", "SD-32"]
lcut_ids = ["SD-03", "SD-05"]
for oid in jcut_ids + lcut_ids:
    d = validated_outputs.get(oid)
    ref = references.get(oid)
    if not d or not ref:
        continue
    direction_total += 1
    expected = ref["positive_intent"]
    model_rels = [r.lower() for r in d.desired_relation_or_change]
    if expected.lower() in model_rels:
        direction_correct += 1
    else:
        direction_miss.append((oid, expected, model_rels))

direction_accuracy = direction_correct / direction_total if direction_total else 0

# ── Ambiguity abstention accuracy ──
ambiguous_ids = ["SD-23", "SD-24", "SD-25", "SD-26"]
ambiguous_correct = 0
ambiguous_total = len(ambiguous_ids)
ambiguous_miss = []
for oid in ambiguous_ids:
    d = validated_outputs.get(oid)
    if d and d.status in ("UNDERSPECIFIED", "NEEDS_CONTEXT"):
        ambiguous_correct += 1
    else:
        ambiguous_miss.append((oid, d.status if d else "missing"))

ambiguous_accuracy = ambiguous_correct / ambiguous_total

# ── Status match ──
status_correct = 0
status_total = 0
status_miss = []
for oid, d in validated_outputs.items():
    ref = references.get(oid)
    if not ref:
        continue
    status_total += 1
    if d.status.value == ref["status"]:
        status_correct += 1
    else:
        status_miss.append((oid, ref["status"], d.status.value))

status_accuracy = status_correct / status_total if status_total else 0

# ── Negative examples (SD-04, SD-08, SD-30) ──
negative_example_correct = 0
negative_example_ids = ["SD-04", "SD-08", "SD-30"]
for oid in negative_example_ids:
    d = validated_outputs.get(oid)
    if d:
        rels = [r.lower() for r in d.desired_relation_or_change]
        # Should NOT have the positive action
        if oid in ("SD-04", "SD-08"):
            if "audio_precedes_picture" not in rels:
                negative_example_correct += 1
        elif oid == "SD-30":
            if "extend_visible_duration" not in rels:
                negative_example_correct += 1

# ── Vertical Slice hard case (SD-01) ──
vs = validated_outputs.get("SD-01")
vs_pass = False
vs_details = {}
if vs:
    vs_details = {
        "positive_audio_before_picture": "audio_precedes_picture" in [r.lower() for r in vs.desired_relation_or_change],
        "picture_cut_preserved": "picture_cut_position" in [p.lower() for p in vs.must_preserve],
        "no_transition": any("transition" in m.lower() for m in vs.must_avoid),
        "creative_purpose_retained": len(vs.creative_intent) > 10,
        "exact_offset_not_fabricated": vs.parameterization.exact_value is None,
        "needs_context_explicit": vs.status.value == "NEEDS_CONTEXT",
    }
    vs_pass = all(vs_details.values())

# ── Print results ──
print("=" * 60)
print("DIRECTOR BRAIN SEMANTIC DECISION POC — EVALUATION")
print("=" * 60)
print(f"\nTotal benchmark samples: {len(benchmark)}")
print(f"Model outputs: {len(outputs_raw)}")
print(f"\n--- Schema Validity ---")
print(f"  Valid: {schema_valid}/{len(outputs_raw)} ({schema_valid_rate:.1%})")
if schema_invalid:
    for oid, err in schema_invalid:
        print(f"  INVALID {oid}: {err[:80]}")

print(f"\n--- Positive Intent Recall ---")
print(f"  {positive_correct}/{positive_total} ({positive_recall:.1%})")
for oid, expected, actual in positive_miss:
    print(f"  MISS {oid}: expected={expected}, actual={actual}")

print(f"\n--- Negative Constraint Recall ---")
print(f"  {negative_correct}/{negative_total} ({negative_recall:.1%})")
for oid, neg in negative_miss:
    print(f"  MISS {oid}: {neg}")

print(f"\n--- Relation Direction (J-cut vs L-cut) ---")
print(f"  {direction_correct}/{direction_total} ({direction_accuracy:.1%})")
for oid, expected, actual in direction_miss:
    print(f"  MISS {oid}: expected={expected}, actual={actual}")

print(f"\n--- Ambiguity Abstention ---")
print(f"  {ambiguous_correct}/{ambiguous_total} ({ambiguous_accuracy:.1%})")
for oid, actual in ambiguous_miss:
    print(f"  MISS {oid}: status={actual}")

print(f"\n--- Status Match ---")
print(f"  {status_correct}/{status_total} ({status_accuracy:.1%})")
for oid, expected, actual in status_miss:
    print(f"  MISS {oid}: expected={expected}, actual={actual}")

print(f"\n--- Negative Examples (no false positive) ---")
print(f"  {negative_example_correct}/{len(negative_example_ids)}")

print(f"\n--- Exact Parameter Hallucination ---")
print(f"  Count: {hallucination_count}")
for oid, val in hallucination_details:
    print(f"  HALLUCINATED {oid}: {val}")

print(f"\n--- Tool/API Leakage ---")
print(f"  Count: {tool_leak_count}")
for oid, terms in tool_leak_details:
    print(f"  LEAK {oid}: {terms}")

print(f"\n--- Vertical Slice Hard Case (SD-01) ---")
print(f"  PASS: {vs_pass}")
for k, v in vs_details.items():
    print(f"    {k}: {v}")

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  schema_valid_rate:          {schema_valid_rate:.1%}")
print(f"  positive_intent_recall:     {positive_recall:.1%}")
print(f"  negative_constraint_recall: {negative_recall:.1%}")
print(f"  relation_direction_accuracy:{direction_accuracy:.1%}")
print(f"  ambiguity_abstention:       {ambiguous_accuracy:.1%}")
print(f"  status_match:               {status_accuracy:.1%}")
print(f"  exact_param_hallucination:  {hallucination_count}")
print(f"  tool_leakage:               {tool_leak_count}")
print(f"  vertical_slice_case:        {'PASS' if vs_pass else 'FAIL'}")

# Save results
results = {
    "total_samples": len(benchmark),
    "schema_valid_rate": schema_valid_rate,
    "positive_intent_recall": positive_recall,
    "negative_constraint_recall": negative_recall,
    "relation_direction_accuracy": direction_accuracy,
    "ambiguity_abstention_accuracy": ambiguous_accuracy,
    "status_match_accuracy": status_accuracy,
    "exact_parameter_hallucination_count": hallucination_count,
    "tool_leakage_count": tool_leak_count,
    "vertical_slice_case_pass": vs_pass,
    "vertical_slice_details": vs_details,
    "positive_misses": positive_miss,
    "negative_misses": negative_miss,
    "direction_misses": direction_miss,
}
with open(POC_DIR / "eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nResults saved to eval_results.json")
