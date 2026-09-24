"""M1 真实验收脚本：端到端跑真实素材，不使用 mock。
验证项：镜头切分、确定性分析、B3关键帧抽取、全链路pipeline、ASSET层、缓存层。
"""
import sys
import json
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

VIDEO = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"
OUT_DIR = Path(r"D:\新建豆包\AI-Director\tests\real_output\m1_verify")
OUT_DIR.mkdir(parents=True, exist_ok=True)

results = {}

# ── 1. 镜头切分 ──────────────────────────────────────────────
print("=" * 60)
print("[1] discover_shots 真实镜头切分")
print("=" * 60)
from observation_service.shot_discovery import discover_shots
shots = discover_shots(VIDEO)
print(f"  输入: {VIDEO}")
print(f"  切出镜头数: {len(shots)}")
if shots:
    print(f"  第一个镜头: in={shots[0].get('source_in_us')}us out={shots[0].get('source_out_us')}us "
          f"时长={(shots[0]['source_out_us']-shots[0]['source_in_us'])/1e6:.2f}s")
    print(f"  最后一个镜头: in={shots[-1].get('source_in_us')}us out={shots[-1].get('source_out_us')}us")
    durations = [(s['source_out_us']-s['source_in_us'])/1e6 for s in shots]
    print(f"  镜头时长: min={min(durations):.2f}s max={max(durations):.2f}s avg={sum(durations)/len(durations):.2f}s")
    # 保存镜头列表
    with open(OUT_DIR / "shots.json", "w", encoding="utf-8") as f:
        json.dump(shots, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {OUT_DIR / 'shots.json'}")
    results["shot_count"] = len(shots)
    results["shot_durations"] = durations
else:
    print("  ❌ 切出 0 个镜头！")
    results["shot_count"] = 0

# ── 2. 确定性分析（前3个镜头）────────────────────────────────
print("\n" + "=" * 60)
print("[2] deterministic_analysis 真实分析（前3镜头）")
print("=" * 60)
from observation_service.deterministic_analysis import analyze_shot
if shots:
    obs_list = []
    for i, shot in enumerate(shots[:3]):
        obs = analyze_shot(VIDEO, shot)
        obs_list.append(obs)
        print(f"  镜头{i+1} ({shot['shot_id'][:20]}...):")
        print(f"    claim_kind={obs.claim_kind}, provider={obs.provider}")
        print(f"    confidence={obs.confidence}")
        print(f"    claim={obs.claim[:120] if obs.claim else '(empty)'}")
    # 保存
    with open(OUT_DIR / "deterministic_obs.json", "w", encoding="utf-8") as f:
        json.dump([o.model_dump(mode="json") for o in obs_list], f, ensure_ascii=False, indent=2)
    print(f"  已保存: {OUT_DIR / 'deterministic_obs.json'}")
    results["deterministic_obs_count"] = len(obs_list)
    results["deterministic_claim_kind"] = str(obs_list[0].claim_kind)
else:
    print("  跳过（无镜头）")

# ── 3. B3 关键帧抽取验证 ─────────────────────────────────────
print("\n" + "=" * 60)
print("[3] extract_keyframe B3 修复验证（镜头中点抽帧）")
print("=" * 60)
from observation_service.keyframe import extract_keyframe
if shots:
    shot = shots[0]
    mid_us = (shot["source_in_us"] + shot["source_out_us"]) // 2
    with extract_keyframe(VIDEO, shot["source_in_us"], shot["source_out_us"]) as frame_path:
        print(f"  镜头范围: {shot['source_in_us']}us - {shot['source_out_us']}us")
        print(f"  理论中点: {mid_us}us")
        print(f"  抽帧结果: {frame_path}")
        if frame_path and Path(frame_path).exists():
            size = Path(frame_path).stat().st_size
            print(f"  文件大小: {size} bytes")
            # 用 ffprobe 验证帧的时间戳
            import subprocess
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "frame=pkt_pts_time",
                 "-of", "csv=p=0", frame_path],
                capture_output=True, text=True, timeout=10
            )
            print(f"  ffprobe 输出: {r.stdout.strip()[:100]}")
            results["keyframe_extracted"] = True
            results["keyframe_size"] = size
        else:
            print("  ❌ 抽帧失败或文件不存在！")
            results["keyframe_extracted"] = False
else:
    print("  跳过（无镜头）")

# ── 4. 全链路 pipeline.analyze_media ─────────────────────────
print("\n" + "=" * 60)
print("[4] pipeline.analyze_media 全链路（切分+分析）")
print("=" * 60)
from observation_service.pipeline import analyze_media
all_obs = analyze_media(VIDEO)
print(f"  产出观测数: {len(all_obs)}")
if all_obs:
    kinds = {}
    for o in all_obs:
        k = str(o.claim_kind)
        kinds[k] = kinds.get(k, 0) + 1
    print(f"  claim_kind 分布: {kinds}")
    print(f"  第一个观测: type={all_obs[0].observation_type}, provider={all_obs[0].provider}")
    with open(OUT_DIR / "pipeline_obs.json", "w", encoding="utf-8") as f:
        json.dump([o.model_dump(mode="json") for o in all_obs], f, ensure_ascii=False, indent=2)
    print(f"  已保存: {OUT_DIR / 'pipeline_obs.json'}")
    results["pipeline_obs_count"] = len(all_obs)
    results["pipeline_kinds"] = kinds
else:
    print("  ❌ 全链路产出 0 个观测！")

# ── 5. ASSET 层 build_asset_index ────────────────────────────
print("\n" + "=" * 60)
print("[5] build_asset_index 真实资产索引")
print("=" * 60)
from director_brain.context_gateway import build_asset_index
ctx = build_asset_index(VIDEO, all_obs)
print(f"  context_id: {ctx.context_id}")
print(f"  layer: {ctx.layers}")
print(f"  coverage: {ctx.coverage}")
print(f"  asset_refs: {ctx.asset_refs}")
print(f"  source_content_hashes: {ctx.source_content_hashes}")
print(f"  evidence_refs 数: {len(ctx.evidence_refs)}")
results["asset_layer"] = str(ctx.layers)
results["asset_coverage"] = ctx.coverage
results["asset_evidence_count"] = len(ctx.evidence_refs)

# ── 6. 缓存层 analysis_cache ─────────────────────────────────
print("\n" + "=" * 60)
print("[6] analysis_cache 真实缓存验证")
print("=" * 60)
from director_brain.analysis_cache import compute_fingerprint, AnalysisCache
# 计算两个指纹
fp1 = compute_fingerprint("hash_abc", "ollama", "qwen3-vl", "v1", "1fps", "1000000", "1.0")
fp2 = compute_fingerprint("hash_abc", "ollama", "qwen3-vl", "v1", "1fps", "1000000", "1.0")
fp3 = compute_fingerprint("hash_abc", "zhipu", "glm-4.6v", "v1", "1fps", "1000000", "1.0")
print(f"  相同输入指纹一致: {fp1 == fp2}")
print(f"  provider变化指纹不同: {fp1 != fp3}")
# 真实缓存 put/get
cache = AnalysisCache(str(OUT_DIR / "cache.json"))
test_obs = all_obs[:2] if all_obs else []
cache.put(fp1, test_obs)
hit = cache.get(fp1)
miss = cache.get(fp3)
print(f"  缓存命中: {hit is not None and len(hit) == len(test_obs)}")
print(f"  缓存未命中: {miss is None}")
print(f"  命中率: {cache.hit_rate():.2f}")
results["cache_fingerprint_consistent"] = fp1 == fp2
results["cache_provider_change"] = fp1 != fp3
results["cache_hit"] = hit is not None
results["cache_miss"] = miss is None

# ── 7. ASR 真实转写（检查模型权重）──────────────────────────
print("\n" + "=" * 60)
print("[7] ASR 真实转写能力检查")
print("=" * 60)
asr_model_path = r"D:\新建豆包\gen1-roughcut\assets\asr\large-v3-turbo\model.bin"
asr_exists = Path(asr_model_path).exists()
print(f"  ASR 模型权重: {asr_model_path}")
print(f"  存在: {asr_exists}")
if asr_exists:
    size_gb = Path(asr_model_path).stat().st_size / 1e9
    print(f"  大小: {size_gb:.2f} GB")
    # 尝试真实转写（用一个短片段，可能较慢）
    try:
        from observation_service.asr import transcribe
        print("  尝试真实转写（可能需要几分钟）...")
        asr_obs = transcribe(VIDEO)
        print(f"  转写产出: {len(asr_obs)} 个 segment")
        if asr_obs:
            print(f"  第一个: '{asr_obs[0].claim[:80]}...'")
            with open(OUT_DIR / "asr_result.json", "w", encoding="utf-8") as f:
                json.dump([o.model_dump(mode="json") for o in asr_obs], f, ensure_ascii=False, indent=2)
        results["asr_real_run"] = True
        results["asr_segment_count"] = len(asr_obs)
    except Exception as e:
        print(f"  ⚠️ 真实转写异常: {type(e).__name__}: {e}")
        results["asr_real_run"] = False
        results["asr_error"] = str(e)
else:
    print("  ⚠️ ASR 模型权重不存在，无法真实转写")
    results["asr_model_exists"] = False

# ── 8. VLM 真实调用环境检查 ──────────────────────────────────
print("\n" + "=" * 60)
print("[8] VLM 真实调用环境检查")
print("=" * 60)
import os
zhipu_key = os.environ.get("ZHIPU_API_KEY", "")
ollama_running = False
try:
    import urllib.request
    urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
    ollama_running = True
except Exception:
    pass
print(f"  智谱 API key 配置: {'是' if zhipu_key else '否'}")
print(f"  Ollama 服务运行: {'是' if ollama_running else '否'}")
results["zhipu_key_configured"] = bool(zhipu_key)
results["ollama_running"] = ollama_running

# ── 总结 ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("M1 真实验收总结")
print("=" * 60)
with open(OUT_DIR / "verify_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(f"  结果已保存: {OUT_DIR / 'verify_results.json'}")
print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
