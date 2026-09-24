"""M2 真实端到端验证：P0-1/P0-2/P0-3 修复后跑通全链路。"""
import sys
import traceback

sys.path.insert(0, r"D:\新建豆包\AI-Director")

VIDEO = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"

print("=== Step 1: analyze_media ===", flush=True)
from observation_service.pipeline import analyze_media
obs = analyze_media(VIDEO)
print(f"  deterministic_technical obs: {sum(1 for o in obs if o.observation_type == 'deterministic_technical')}", flush=True)

print("=== Step 2: transcribe (ASR) ===", flush=True)
try:
    from observation_service.asr import transcribe
    asr_obs = transcribe(VIDEO)
    print(f"  asr obs: {len(asr_obs)}", flush=True)
except Exception as e:
    print(f"  ASR failed (non-fatal): {e}", flush=True)
    asr_obs = []

all_obs = obs + asr_obs

print("=== Step 3: compile_brief ===", flush=True)
from director_brain.brief_compiler import compile_brief
brief = compile_brief("audit", VIDEO, all_obs)
print(f"  target_duration={brief.target_duration}", flush=True)
print(f"  source_duration_us={brief.source_duration_us}", flush=True)

print("=== Step 4: build_story_graph ===", flush=True)
from director_brain.story_graph_builder import build_story_graph
graph = build_story_graph(brief, all_obs)
print(f"  nodes={len(graph.nodes)}, edges={len(graph.edges)}", flush=True)

print("=== Step 5: infer_relations ===", flush=True)
from director_brain.relation_inference import infer_relations
relations = infer_relations(all_obs, graph)
print(f"  relations={len(relations)}", flush=True)

print("=== Step 6: generate_plan ===", flush=True)
from director_brain.director_reasoner import get_director_reasoner
reasoner = get_director_reasoner("heuristic")
edl, plan = reasoner.generate_plan(brief, graph, all_obs)
print(f"  edits={len(edl.ordered_edits)}, expected_duration={edl.expected_duration}", flush=True)
for e in edl.ordered_edits:
    print(f"    {e.source_asset_id}: {e.in_frame}-{e.out_frame} ({e.out_frame-e.in_frame}us) fn={e.shot_function}", flush=True)

print("=== Step 7: validate_plan (pre-repair) ===", flush=True)
from director_brain.plan_validator import validate_plan
ok1, err1 = validate_plan(edl, plan, all_obs)
print(f"  ok={ok1}, errors={err1}", flush=True)

print("=== Step 8: repair_plan ===", flush=True)
from director_brain.plan_repair import repair_plan
redl, rplan = repair_plan(edl, plan, all_obs)
print(f"  repaired edits={len(redl.ordered_edits)}, expected_duration={redl.expected_duration}", flush=True)

print("=== Step 9: validate_plan (post-repair) ===", flush=True)
ok2, err2 = validate_plan(redl, rplan, all_obs)
print(f"  ok={ok2}, errors={err2}", flush=True)

print("=== Assertions ===", flush=True)
assert brief.target_duration == 15_000_000, f"P0-1 not fixed: {brief.target_duration}"
print(f"  P0-1 OK: target_duration={brief.target_duration}", flush=True)

assert ok2, f"P0-2 not fixed: repair still invalid, errors={err2}"
print(f"  P0-2 OK: validate_plan after repair = True", flush=True)

acts = [d.rationale for d in rplan.decisions if "act=" in (d.rationale or "")]
assert len(acts) >= 4, f"P0-3 not fixed: only {len(acts)} decisions have act label"
print(f"  P0-3 OK: {len(acts)} decisions have act= label", flush=True)
for a in acts:
    print(f"    {a}", flush=True)

print("ALL P0 FIXES VERIFIED", flush=True)
