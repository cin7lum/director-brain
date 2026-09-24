"""M2 主控独立复验：端到端跑真实素材，不使用 mock。"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

VIDEO = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"
OUT = Path(r"D:\新建豆包\AI-Director\tests\real_output\m2_master_verify")
OUT.mkdir(parents=True, exist_ok=True)

results = {}

# ── 1. 观察服务 ─────────────────────────────────────────────
print("=" * 60)
print("[1] analyze_media + transcribe")
print("=" * 60)
from observation_service.pipeline import analyze_media
from observation_service.asr import transcribe

obs = analyze_media(VIDEO)
asr_obs = transcribe(VIDEO)
all_obs = obs + asr_obs
print(f"  deterministic 观测: {len(obs)}")
print(f"  ASR 观测: {len(asr_obs)}")
print(f"  合计: {len(all_obs)}")
results["obs_count"] = len(all_obs)

# ── 2. Brief Compiler ───────────────────────────────────────
print("\n" + "=" * 60)
print("[2] compile_brief")
print("=" * 60)
from director_brain.brief_compiler import compile_brief
brief = compile_brief("master_verify", VIDEO, all_obs)
print(f"  target_duration: {brief.target_duration}")
print(f"  sound_language: {brief.sound_language}")
print(f"  themes: {brief.themes}")
print(f"  字段数: {len(brief.model_dump())}")
results["brief_fields"] = len(brief.model_dump())
results["brief_target_duration"] = str(brief.target_duration)
with open(OUT / "brief.json", "w", encoding="utf-8") as f:
    json.dump(brief.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)

# ── 3. Story Graph Builder ──────────────────────────────────
print("\n" + "=" * 60)
print("[3] build_story_graph")
print("=" * 60)
from director_brain.story_graph_builder import build_story_graph
graph = build_story_graph(brief, all_obs)
print(f"  节点数: {len(graph.nodes)}")
print(f"  边数: {len(graph.edges)}")
acts = [n.attributes.get("act", "?") for n in graph.nodes]
print(f"  四幕: {acts}")
for n in graph.nodes:
    shots = n.attributes.get("shot_count", "?")
    print(f"    {n.attributes.get('act','?')}: {shots} 镜头")
results["graph_nodes"] = len(graph.nodes)
results["graph_edges"] = len(graph.edges)
results["graph_acts"] = acts
with open(OUT / "story_graph.json", "w", encoding="utf-8") as f:
    json.dump(graph.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)

# ── 4. Relation Inference ───────────────────────────────────
print("\n" + "=" * 60)
print("[4] infer_relations")
print("=" * 60)
from director_brain.relation_inference import infer_relations
relations = infer_relations(all_obs, graph)
print(f"  关系边数: {len(relations)}")
edge_types = {}
for e in relations:
    t = str(e.edge_type)
    edge_types[t] = edge_types.get(t, 0) + 1
print(f"  类型分布: {edge_types}")
inferred_count = sum(1 for e in relations if e.inference_status == "inferred")
print(f"  inferred 状态: {inferred_count}/{len(relations)}")
results["relation_count"] = len(relations)
results["relation_types"] = edge_types
with open(OUT / "relations.json", "w", encoding="utf-8") as f:
    json.dump([e.model_dump(mode="json") for e in relations], f, ensure_ascii=False, indent=2, default=str)

# ── 5. Director Reasoner (heuristic) ────────────────────────
print("\n" + "=" * 60)
print("[5] heuristic_reasoner.generate_plan")
print("=" * 60)
from director_brain.director_reasoner import get_director_reasoner
reasoner = get_director_reasoner("heuristic")
edl, plan = reasoner.generate_plan(brief, graph, all_obs)
total_dur = sum(e.out_frame - e.in_frame for e in edl.ordered_edits) / 1e6
print(f"  EDL 片段数: {len(edl.ordered_edits)}")
print(f"  总时长: {total_dur:.1f}s")
print(f"  timebase: {edl.timebase}")
print(f"  Plan decisions: {len(plan.decisions)}")
print(f"  validation_status: {plan.validation_status}")
results["edl_count"] = len(edl.ordered_edits)
results["edl_duration"] = round(total_dur, 1)
results["plan_decisions"] = len(plan.decisions)
with open(OUT / "heuristic_edl.json", "w", encoding="utf-8") as f:
    json.dump(edl.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)

# ── 6. Plan Validator ───────────────────────────────────────
print("\n" + "=" * 60)
print("[6] validate_plan (heuristic EDL)")
print("=" * 60)
from director_brain.plan_validator import validate_plan
is_valid, errors = validate_plan(edl, plan, all_obs)
print(f"  valid: {is_valid}")
print(f"  errors: {errors}")
results["heuristic_valid"] = is_valid
results["heuristic_errors"] = errors

# ── 7. Validator 抓非法 EDL ─────────────────────────────────
print("\n" + "=" * 60)
print("[7] validate_plan (构造非法 EDL)")
print("=" * 60)
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.director_plan import Decision, DirectorDecisionPlan
import time

bad_edits = [
    EditItem(source_asset_id="nonexistent_shot", source_media_hash="fake", in_frame=0, out_frame=1000000,
             timebase=1000000, shot_function="test", rationale="bad"),
    EditItem(source_asset_id=all_obs[0].media_asset_id, source_media_hash=all_obs[0].media_hash, in_frame=2000000, out_frame=1000000,  # in > out
             timebase=1000000, shot_function="test", rationale="bad"),
    EditItem(source_asset_id=all_obs[0].media_asset_id, source_media_hash=all_obs[0].media_hash, in_frame=500000, out_frame=1500000,  # 与下一个重叠
             timebase=1000000, shot_function="test", rationale="bad"),
    EditItem(source_asset_id=all_obs[0].media_asset_id, source_media_hash=all_obs[0].media_hash, in_frame=1000000, out_frame=2000000,  # 与上一个重叠
             timebase=1000000, shot_function="test", rationale="bad"),
]
bad_edl = EditorialDecisionList(
    edl_id="bad_edl", version="1.0", brief_version="1.0", context_id="test",
    timebase=1000000, approval_state="draft", ordered_edits=bad_edits,
    source_asset_hashes=[], schema_version="1.0", project_id="test",
    created_at=int(time.time()), producer="test", source_ref="test"
)
bad_plan = DirectorDecisionPlan(
    plan_id="bad_plan", version="1.0", brief_version="1.0", film_state_version="test",
    validation_status="pending", approval_state="draft", decisions=[],
    schema_version="1.0", project_id="test", created_at=int(time.time()),
    producer="test", source_ref="test"
)
is_valid2, errors2 = validate_plan(bad_edl, bad_plan, all_obs)
print(f"  valid: {is_valid2}（期望 False）")
print(f"  errors ({len(errors2)}): {errors2}")
results["bad_valid"] = is_valid2
results["bad_error_count"] = len(errors2)
results["bad_errors"] = errors2

# ── 8. Plan Repair ──────────────────────────────────────────
print("\n" + "=" * 60)
print("[8] repair_plan (修复非法 EDL)")
print("=" * 60)
from director_brain.plan_repair import repair_plan
repaired_edl, repaired_plan = repair_plan(bad_edl, bad_plan, all_obs)
is_valid3, errors3 = validate_plan(repaired_edl, repaired_plan, all_obs)
print(f"  修复后 valid: {is_valid3}（期望 True）")
print(f"  修复后 errors: {errors3}")
print(f"  修复后片段数: {len(repaired_edl.ordered_edits)}")
results["repaired_valid"] = is_valid3
results["repaired_errors"] = errors3
results["repaired_count"] = len(repaired_edl.ordered_edits)

# ── 9. LLM Director 骨架检查 ────────────────────────────────
print("\n" + "=" * 60)
print("[9] LLMDirectorReasoner 骨架")
print("=" * 60)
from director_brain.director_reasoner import LLMDirectorReasoner
llm = LLMDirectorReasoner(provider="ollama", config={"model": "test"})
print(f"  实例化成功: provider={llm.provider}")
try:
    llm.generate_plan(brief, graph, all_obs)
    print("  ❌ 应该抛 NotImplementedError 但没有")
    results["llm_raises"] = False
except NotImplementedError:
    print("  ✅ generate_plan 抛 NotImplementedError（不调用 ollama）")
    results["llm_raises"] = True

# ── 总结 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("M2 主控复验总结")
print("=" * 60)
with open(OUT / "verify_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
