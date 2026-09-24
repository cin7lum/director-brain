"""M2 主控复验第7-9步：validator抓错 + repair + LLM骨架。"""
import sys, time
sys.path.insert(0, ".")

from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.plan_validator import validate_plan
from director_brain.plan_repair import repair_plan
from director_brain.director_reasoner import LLMDirectorReasoner
from observation_service.pipeline import analyze_media

VIDEO = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"
obs = analyze_media(VIDEO)
print(f"obs: {len(obs)}")

bad_edits = [
    EditItem(source_asset_id="nonexistent_shot", source_media_hash="fake",
             in_frame=0, out_frame=1000000, timebase=1000000, shot_function="t", rationale="b"),
    EditItem(source_asset_id=obs[0].media_asset_id, source_media_hash=obs[0].media_hash,
             in_frame=2000000, out_frame=1000000, timebase=1000000, shot_function="t", rationale="b"),
    EditItem(source_asset_id=obs[0].media_asset_id, source_media_hash=obs[0].media_hash,
             in_frame=500000, out_frame=1500000, timebase=1000000, shot_function="t", rationale="b"),
    EditItem(source_asset_id=obs[0].media_asset_id, source_media_hash=obs[0].media_hash,
             in_frame=1000000, out_frame=2000000, timebase=1000000, shot_function="t", rationale="b"),
]
bad_edl = EditorialDecisionList(
    edl_id="bad", version="1.0", brief_version="1.0", context_id="t",
    timebase=1000000, approval_state="draft", ordered_edits=bad_edits,
    source_asset_hashes=[], schema_version="1.0", project_id="t",
    created_at=int(time.time()), producer="t", source_ref="t")
bad_plan = DirectorDecisionPlan(
    plan_id="bad", version="1.0", brief_version="1.0", film_state_version="t",
    validation_status="pending", approval_state="draft", decisions=[],
    schema_version="1.0", project_id="t", created_at=int(time.time()),
    producer="t", source_ref="t")

print("\n=== [7] Validator 抓非法 EDL ===")
ok, errs = validate_plan(bad_edl, bad_plan, obs)
print(f"valid: {ok} (期望 False)")
print(f"errors ({len(errs)}):")
for e in errs:
    print(f"  - {e}")

print("\n=== [8] Repair 修复 ===")
redl, rplan = repair_plan(bad_edl, bad_plan, obs)
ok2, errs2 = validate_plan(redl, rplan, obs)
print(f"修复后 valid: {ok2} (期望 True)")
print(f"修复后 errors: {errs2}")
print(f"修复后片段数: {len(redl.ordered_edits)}")

print("\n=== [9] LLMDirectorReasoner 骨架 ===")
llm = LLMDirectorReasoner(provider="ollama", config={"model": "test"})
print(f"实例化: provider={llm.provider}")
try:
    llm.generate_plan(None, None, None)
    print("FAIL: 未抛 NotImplementedError")
except NotImplementedError:
    print("PASS: 抛 NotImplementedError")
except Exception as e:
    print(f"其他异常: {type(e).__name__}: {e}")
