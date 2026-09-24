"""Arsenal 媒体导入与代理生成单元测试。

使用真实素材 sintel_trailer.mp4 验证 probe_media / generate_proxy / import_media。
标记 slow 的测试会实际调用 ffmpeg，耗时较长。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from arsenal.media_import import generate_proxy, import_media, probe_media
from arsenal.project_layout import create_project

# 真实测试素材路径
REAL_VIDEO = Path(r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4")


@pytest.fixture(scope="module")
def real_video() -> str:
    """真实素材路径（str），若不存在则跳过整个模块的 slow 测试。"""
    if not REAL_VIDEO.is_file():
        pytest.skip(f"真实素材不存在: {REAL_VIDEO}", allow_module_level=False)
    return str(REAL_VIDEO)


def _ffprobe_width(path: str) -> int:
    """用 ffprobe 读取视频宽度，用于交叉验证代理分辨率。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width",
        "-of", "csv=p=0",
        path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


@pytest.mark.slow
class TestProbeMedia:
    """probe_media 元数据读取。"""

    def test_returns_metadata(self, real_video: str) -> None:
        """用真实 sintel_trailer.mp4 断言各字段。"""
        meta = probe_media(real_video)

        # 时长约 52.2s（52208333 微秒，±100000）
        assert abs(meta["duration_us"] - 52_208_333) < 100_000
        assert meta["width"] == 854
        assert meta["height"] == 480
        assert abs(meta["fps"] - 24.0) < 0.5
        assert meta["audio_present"] is True
        assert len(meta["sha256"]) == 64
        assert meta["file_size"] > 0
        assert meta["path"] == real_video
        assert isinstance(meta["codec"], str) and len(meta["codec"]) > 0


@pytest.mark.slow
class TestGenerateProxy:
    """generate_proxy 代理生成与幂等。"""

    def test_creates_file(self, real_video: str, tmp_path: Path) -> None:
        """生成代理后文件存在，ffprobe 可解析，宽度 <= 480。"""
        proxy_path = str(tmp_path / "test_proxy.mp4")
        result = generate_proxy(real_video, proxy_path)

        assert result == proxy_path
        assert Path(proxy_path).is_file()
        assert Path(proxy_path).stat().st_size > 0
        # ffprobe 能解析且宽度 <= 480
        width = _ffprobe_width(proxy_path)
        assert width <= 480
        # meta 旁车文件存在（文件名 = 代理名 + ".proxy_meta.json"）
        meta_path = Path(proxy_path + ".proxy_meta.json")
        assert meta_path.is_file()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "source_sha256" in meta
        assert meta["width"] == 480

    def test_idempotent_skip(self, real_video: str, tmp_path: Path) -> None:
        """二次调用应跳过生成，mtime 不变。"""
        proxy_path = str(tmp_path / "idempotent_proxy.mp4")
        generate_proxy(real_video, proxy_path)
        mtime_before = Path(proxy_path).stat().st_mtime_ns

        # 再次调用应命中缓存跳过
        generate_proxy(real_video, proxy_path)
        mtime_after = Path(proxy_path).stat().st_mtime_ns

        assert mtime_before == mtime_after, "代理文件不应被重新生成"


@pytest.mark.slow
class TestImportMedia:
    """import_media 完整导入流程。"""

    def test_full_flow(self, real_video: str, tmp_path: Path) -> None:
        """create_project + import_media，校验 manifest 与目录产物。"""
        project_root = create_project(str(tmp_path), "import_demo")
        entry = import_media(real_video, project_root)

        # entry 应包含所有关键字段
        expected_fields = {
            "duration_us", "width", "height", "fps", "codec",
            "audio_present", "file_size", "sha256",
            "original_path", "proxy_path", "imported_at",
        }
        assert expected_fields.issubset(entry.keys()), (
            f"缺少字段: {expected_fields - set(entry.keys())}"
        )
        assert len(entry["sha256"]) == 64
        assert entry["audio_present"] is True

        # originals/ 与 proxies/ 都应有文件
        originals_dir = project_root / "media" / "originals"
        proxies_dir = project_root / "media" / "proxies"
        assert any(originals_dir.iterdir()), "originals/ 应为空"
        assert any(proxies_dir.iterdir()), "proxies/ 应为空"

        # original_path / proxy_path 指向真实存在的文件
        assert Path(entry["original_path"]).is_file()
        assert Path(entry["proxy_path"]).is_file()
        assert Path(entry["original_path"]).parent == originals_dir
        assert Path(entry["proxy_path"]).parent == proxies_dir

        # media_manifest.json 存在且结构正确
        manifest_path = project_root / "media_manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "media" in manifest
        assert len(manifest["media"]) == 1
        assert manifest["media"][0]["sha256"] == entry["sha256"]

    def test_import_idempotent(self, real_video: str, tmp_path: Path) -> None:
        """重复导入同一文件，代理 mtime 不变，manifest 不重复追加。"""
        project_root = create_project(str(tmp_path), "import_idem")
        entry1 = import_media(real_video, project_root)

        proxy_file = Path(entry1["proxy_path"])
        mtime_before = proxy_file.stat().st_mtime_ns

        entry2 = import_media(real_video, project_root)
        mtime_after = proxy_file.stat().st_mtime_ns

        # 代理未被重新生成
        assert mtime_before == mtime_after
        # sha256 相同，manifest 中应只有一条
        assert entry2["sha256"] == entry1["sha256"]
        manifest = json.loads(
            (project_root / "media_manifest.json").read_text(encoding="utf-8")
        )
        same_sha = [m for m in manifest["media"] if m["sha256"] == entry1["sha256"]]
        assert len(same_sha) == 1, "同一 sha256 应覆盖而非追加"
