"""
VramService のフォーマット推定ロジックに対するユニットテスト。

RenderDoc 実体に依存せず、`renderdoc` モジュールをスタブ注入したうえで
`vram_service.py` を直接ロードし、純粋関数 (format_layout / estimate_texture_bytes)
の数値が正しいことを検証する。

実行: python tests/test_vram_format.py
"""

import importlib.util
import os
import sys
import types


def _load_vram_module():
    """renderdoc スタブを注入して vram_service を単体ロードする。"""
    # RenderDoc 内蔵 Python にしか存在しない `renderdoc` を最小スタブで代替する。
    if "renderdoc" not in sys.modules:
        stub = types.ModuleType("renderdoc")

        class _ResourceId:
            @staticmethod
            def Null():
                return "ResourceId::Null()"

        stub.ResourceId = _ResourceId
        sys.modules["renderdoc"] = stub

    module_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "renderdoc_extension",
        "services",
        "vram_service.py",
    )
    spec = importlib.util.spec_from_file_location("vram_service_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vram = _load_vram_module()


class FakeFormat:
    """RenderDoc ResourceFormat のダックタイプ代替。"""

    def __init__(self, name, comp_count=0, comp_byte_width=0):
        self._name = name
        self.compCount = comp_count
        self.compByteWidth = comp_byte_width
        self.specialFormat = "Regular"
        self.compType = "Float"

    def Name(self):
        return self._name


class FakeTexture:
    """RenderDoc TextureDescription のダックタイプ代替。"""

    def __init__(self, width, height, fmt, depth=1, arraysize=1, mips=1, samples=1, dimension="2D"):
        self.width = width
        self.height = height
        self.depth = depth
        self.arraysize = arraysize
        self.mips = mips
        self.msSamp = samples
        self.dimension = dimension
        self.format = fmt


class FakeUsage:
    """RenderDoc EventUsage のダックタイプ代替。"""

    def __init__(self, usage_token, event_id):
        self.usage = usage_token
        self.eventId = event_id


class FakeResource:
    """resourceId を持つ最小リソース。"""

    def __init__(self, rid):
        self.resourceId = rid


class FakeController:
    """GetTextures/GetBuffers/GetUsage を備えた最小コントローラ。"""

    def __init__(self, textures, buffers, usage_map):
        self._textures = textures
        self._buffers = buffers
        self._usage_map = usage_map

    def GetTextures(self):
        return self._textures

    def GetBuffers(self):
        return self._buffers

    def GetUsage(self, rid):
        return self._usage_map.get(rid, [])

    def GetRootActions(self):
        return []

    def GetStructuredFile(self):
        return None


def _make_estimator(controller):
    return vram._VramEstimator(ctx=None, controller=controller, enable_mesh_detection=True)


def _check(name, got, expected):
    ok = got == expected
    status = "PASS" if ok else "FAIL"
    print("[%s] %s: got=%s expected=%s" % (status, name, got, expected))
    return ok


def run():
    failures = 0

    # --- P0 回帰: 多分量フォーマットが分量幅から正しく算出されること ---
    bpp = lambda fmt: vram.format_layout(fmt)["bytes_per_pixel"]

    failures += not _check("RGBA8 4comp*1B", bpp(FakeFormat("R8G8B8A8_UNORM", 4, 1)), 4)
    failures += not _check("RGBA16F 4comp*2B", bpp(FakeFormat("R16G16B16A16_FLOAT", 4, 2)), 8)
    failures += not _check("RGBA32F 4comp*4B", bpp(FakeFormat("R32G32B32A32_FLOAT", 4, 4)), 16)
    failures += not _check("RG16F 2comp*2B", bpp(FakeFormat("R16G16_FLOAT", 2, 2)), 4)
    failures += not _check("R32F 1comp*4B", bpp(FakeFormat("R32_FLOAT", 1, 4)), 4)
    failures += not _check("R16 1comp*2B", bpp(FakeFormat("R16_UNORM", 1, 2)), 2)
    failures += not _check("R8 1comp*1B", bpp(FakeFormat("R8_UNORM", 1, 1)), 1)
    failures += not _check("D16 1comp*2B", bpp(FakeFormat("D16_UNORM", 1, 2)), 2)
    failures += not _check("D32F 1comp*4B", bpp(FakeFormat("D32_FLOAT", 1, 4)), 4)

    # --- packed フォーマット (分量幅では表せない) ---
    failures += not _check("R10G10B10A2 packed", bpp(FakeFormat("R10G10B10A2_UNORM", 4, 0)), 4)
    failures += not _check("R11G11B10 packed", bpp(FakeFormat("R11G11B10_FLOAT", 3, 0)), 4)
    failures += not _check("R5G6B5 packed", bpp(FakeFormat("R5G6B5_UNORM", 3, 0)), 2)
    failures += not _check("D24S8 packed", bpp(FakeFormat("D24S8", 2, 0)), 4)
    failures += not _check("D32S8 packed", bpp(FakeFormat("D32S8", 2, 0)), 8)

    # --- block-compressed フォーマット ---
    bc1 = vram.format_layout(FakeFormat("BC1_UNORM"))
    failures += not _check("BC1 block_bytes", bc1["block_bytes"], 8)
    bc7 = vram.format_layout(FakeFormat("BC7_UNORM"))
    failures += not _check("BC7 block_bytes", bc7["block_bytes"], 16)
    astc = vram.format_layout(FakeFormat("ASTC_8x8_UNORM"))
    failures += not _check("ASTC 8x8 block_w", astc["block_w"], 8)

    # --- estimate_texture_bytes: 総量計算 ---
    # 256x256 RGBA8, mip1, 単一スライス = 256*256*4 = 262144
    tex = FakeTexture(256, 256, FakeFormat("R8G8B8A8_UNORM", 4, 1))
    size, _ = vram.estimate_texture_bytes(tex)
    failures += not _check("256x256 RGBA8 no-mip", size, 256 * 256 * 4)

    # 256x256 RGBA16F = 256*256*8 = 524288 (P0 修正前は 262144 に過小評価されていた)
    tex_hdr = FakeTexture(256, 256, FakeFormat("R16G16B16A16_FLOAT", 4, 2))
    size_hdr, _ = vram.estimate_texture_bytes(tex_hdr)
    failures += not _check("256x256 RGBA16F (P0)", size_hdr, 256 * 256 * 8)

    # フルmipチェーン 256x256 RGBA8: 等比級数 ~ base * 4/3 - 端数
    tex_mip = FakeTexture(256, 256, FakeFormat("R8G8B8A8_UNORM", 4, 1), mips=9)
    size_mip, _ = vram.estimate_texture_bytes(tex_mip)
    expected_mip = sum((max(1, 256 >> m) ** 2) * 4 for m in range(9))
    failures += not _check("256x256 RGBA8 full-mip", size_mip, expected_mip)

    # MSAA: 4x, mip1 → サンプル倍率は mip0 のみ。base*4
    tex_msaa = FakeTexture(128, 128, FakeFormat("R8G8B8A8_UNORM", 4, 1), samples=4)
    size_msaa, _ = vram.estimate_texture_bytes(tex_msaa)
    failures += not _check("128x128 RGBA8 4xMSAA", size_msaa, 128 * 128 * 4 * 4)

    # 配列テクスチャ: arraysize 倍
    tex_arr = FakeTexture(64, 64, FakeFormat("R8G8B8A8_UNORM", 4, 1), arraysize=6)
    size_arr, _ = vram.estimate_texture_bytes(tex_arr)
    failures += not _check("64x64 RGBA8 array6", size_arr, 64 * 64 * 4 * 6)

    # --- GetUsage ベースの分類ロジック ---
    usage_map = {
        "tex_color": [FakeUsage("ResourceUsage.ColorTarget", 10)],
        "tex_depth": [FakeUsage("ResourceUsage.DepthStencilTarget", 11)],
        "tex_both": [
            FakeUsage("ResourceUsage.DepthStencilTarget", 12),
            FakeUsage("ResourceUsage.PS_Resource", 13),
        ],
        "buf_vb": [FakeUsage("ResourceUsage.VertexBuffer", 20), FakeUsage("ResourceUsage.VertexBuffer", 21)],
        "buf_ib": [FakeUsage("ResourceUsage.IndexBuffer", 20)],
        "buf_mesh": [
            FakeUsage("ResourceUsage.VertexBuffer", 30),
            FakeUsage("ResourceUsage.IndexBuffer", 30),
        ],
        "buf_cb": [FakeUsage("ResourceUsage.VS_Constants", 40)],
    }
    textures = [FakeResource("tex_color"), FakeResource("tex_depth"), FakeResource("tex_both")]
    buffers = [FakeResource(r) for r in ["buf_vb", "buf_ib", "buf_mesh", "buf_cb"]]
    est = _make_estimator(FakeController(textures, buffers, usage_map))
    est._usage_by_rid = est._collect_usage_by_rid()

    failures += not _check("usage map size", len(est._usage_by_rid), 7)

    # RT/Depth 分類は usage が正典
    failures += not _check("color -> RT/Color", est._usage_texture_category("tex_color"), "RT/Color")
    failures += not _check("depth -> RT/DepthStencil", est._usage_texture_category("tex_depth"), "RT/DepthStencil")
    failures += not _check("depth+SRV -> RT/DepthStencil", est._usage_texture_category("tex_both"), "RT/DepthStencil")
    failures += not _check("cbuffer -> None (no RT)", est._usage_texture_category("buf_cb"), None)

    # VB/IB 役割の導出
    geom, gstats = est._derive_geometry_usage({})
    failures += not _check("geometry buffers detected", gstats["buffers_detected"], 3)
    failures += not _check("buf_vb roles", vram.geometry_category_from_roles(geom["buf_vb"]["roles"]), "Mesh/VertexBuffer")
    failures += not _check("buf_ib roles", vram.geometry_category_from_roles(geom["buf_ib"]["roles"]), "Mesh/IndexBuffer")
    failures += not _check("buf_mesh roles", vram.geometry_category_from_roles(geom["buf_mesh"]["roles"]), "Mesh/Vertex+Index")
    failures += not _check("buf_vb event count", len(geom["buf_vb"]["events"]), 2)
    failures += not _check("cbuffer not geometry", "buf_cb" in geom, False)

    # --- live-set 峰値 (スイープライン) と未引用資源 ---
    # A[1..3]=100, B[2..4]=200 (A と event2,3 で重複), C[10..11]=50 (分離),
    # D=999 は usage なし(未引用)。全量和=1349、峰値=event2 の A+B=300。
    ls_est = vram._VramEstimator(ctx=None, controller=None, enable_mesh_detection=True, enable_live_set=True)
    ls_est._usage_by_rid = {
        "A": {"roles": set(), "events": {1, 2, 3}},
        "B": {"roles": set(), "events": {2, 3, 4}},
        "C": {"roles": set(), "events": {10, 11}},
    }
    ls_rows = [
        {"kind": "Texture", "category": "RT/Color", "resource_id": "A", "bytes": 100},
        {"kind": "Texture", "category": "RT/Color", "resource_id": "B", "bytes": 200},
        {"kind": "Buffer", "category": "Mesh/VertexBuffer", "resource_id": "C", "bytes": 50},
        {"kind": "Texture", "category": "Texture/Regular", "resource_id": "D", "bytes": 999},
    ]
    live = ls_est._compute_live_set(ls_rows, {2: "GBufferPass"})
    failures += not _check("live-set peak bytes", live["peak_bytes"], 300)
    failures += not _check("live-set peak event", live["peak_event_id"], 2)
    failures += not _check("live-set peak name", live["peak_event_name"], "GBufferPass")
    failures += not _check("live-set peak live count", live["peak_live_resources"], 2)
    failures += not _check("live-set considered (excl. unref)", live["resources_considered"], 3)
    # 全量和(1349)より峰値(300)が小さいこと = aliasing を重複計上しない効果
    failures += not _check("live-set peak < grand total", live["peak_bytes"] < 1349, True)
    failures += not _check("live-set peak category", live["peak_categories"][0]["category"], "RT/Color")

    unref = ls_est._collect_unreferenced(ls_rows)
    failures += not _check("unreferenced count", unref["count"], 1)
    failures += not _check("unreferenced bytes", unref["total_bytes"], 999)
    failures += not _check("unreferenced top rid", unref["top_resources"][0]["resource_id"], "D")

    # --- residency 驻留拆解 (持久 vs 瞬态 vs 未引用) ---
    # 帧事件跨度 [1,100]=99。阈值0.5 → span>=49.5 判持久。
    #   P: events{1,100} span99 → 持久, 1000B
    #   T1: {1,2} span1 → 瞬态, 100B, 区间[1,2]
    #   T2: {50,51} span1 → 瞬态, 200B, 区间[50,51] (与T1不重叠)
    #   T3: {1,2} span1 → 瞬态, 50B, 区间[1,2] (与T1重叠)
    #   U: 无usage → 未引用, 999B
    rs_est = vram._VramEstimator(ctx=None, controller=None, enable_live_set=True, persistent_span_ratio=0.5)
    rs_est._usage_by_rid = {
        "P": {"roles": set(), "events": {1, 100}},
        "T1": {"roles": set(), "events": {1, 2}},
        "T2": {"roles": set(), "events": {50, 51}},
        "T3": {"roles": set(), "events": {1, 2}},
    }
    rs_rows = [
        {"kind": "Buffer", "category": "Buffer/UAV-Structured", "resource_id": "P", "bytes": 1000},
        {"kind": "Texture", "category": "RT/Color", "resource_id": "T1", "bytes": 100},
        {"kind": "Texture", "category": "RT/Color", "resource_id": "T2", "bytes": 200},
        {"kind": "Texture", "category": "RT/Color", "resource_id": "T3", "bytes": 50},
        {"kind": "Texture", "category": "Texture/Regular", "resource_id": "U", "bytes": 999},
    ]
    res = rs_est._compute_residency(rs_rows)
    failures += not _check("residency frame range", res["frame_event_range"], [1, 100])
    failures += not _check("persistent bytes", res["persistent"]["bytes"], 1000)
    failures += not _check("persistent count", res["persistent"]["count"], 1)
    failures += not _check("transient bytes", res["transient"]["bytes"], 350)
    failures += not _check("transient count", res["transient"]["count"], 3)
    # 瞬态峰值: T1+T3=150 @ [1,2] vs T2=200 @ [50,51] → 200
    failures += not _check("transient pooled peak", res["transient"]["pooled_peak_bytes"], 200)
    failures += not _check("poolable headroom (350-200)", res["transient"]["poolable_headroom_bytes"], 150)
    failures += not _check("unreferenced bytes", res["unreferenced_bytes"], 999)
    # 理论最小驻留 = persistent(1000) + 瞬态峰值(200) = 1200
    failures += not _check("theoretical min resident", res["theoretical_min_resident_bytes"], 1200)
    # 可省上限 = grand(2349) - 1200 = 1149 = poolable(150)+unref(999)
    failures += not _check("reducible upper bound", res["reducible_upper_bound_bytes"], 1149)
    # 三段之和 == 全量和 (自洽校验)
    parts = res["persistent"]["bytes"] + res["transient"]["bytes"] + res["unreferenced_bytes"]
    failures += not _check("persistent+transient+unref == grand", parts, 2349)

    print("\n%d failure(s)" % failures)
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
