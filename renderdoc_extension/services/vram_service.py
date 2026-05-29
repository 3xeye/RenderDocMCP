"""
VRAM estimation service for RenderDoc captures.
"""

import math
import re
from collections import defaultdict

import renderdoc as rd


class VramService:
    """Estimate API-visible captured resource memory and mesh buffers."""

    def __init__(self, ctx, invoke_fn):
        self.ctx = ctx
        self._invoke = invoke_fn

    def estimate_vram(
        self,
        top_n=100,
        show_all=False,
        enable_name_heuristic=True,
        enable_mesh_detection=True,
        enable_live_set=True,
        persistent_span_ratio=0.5,
        collect_draw_names=True,
        max_draw_names_per_buffer=8,
        large_resource_threshold_mb=128,
    ):
        """Estimate texture/buffer memory for the current capture."""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"data": None, "error": None}

        def callback(controller):
            try:
                estimator = _VramEstimator(
                    self.ctx,
                    controller,
                    top_n=top_n,
                    show_all=show_all,
                    enable_name_heuristic=enable_name_heuristic,
                    enable_mesh_detection=enable_mesh_detection,
                    enable_live_set=enable_live_set,
                    persistent_span_ratio=persistent_span_ratio,
                    collect_draw_names=collect_draw_names,
                    max_draw_names_per_buffer=max_draw_names_per_buffer,
                    large_resource_threshold_mb=large_resource_threshold_mb,
                )
                result["data"] = estimator.run()
            except Exception as e:
                import traceback

                result["error"] = "Error: %s\n%s" % (str(e), traceback.format_exc())

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["data"]


class _VramEstimator:
    def __init__(
        self,
        ctx,
        controller,
        top_n=100,
        show_all=False,
        enable_name_heuristic=True,
        enable_mesh_detection=True,
        enable_live_set=True,
        persistent_span_ratio=0.5,
        collect_draw_names=True,
        max_draw_names_per_buffer=8,
        large_resource_threshold_mb=128,
    ):
        self.ctx = ctx
        self.controller = controller
        self.top_n = max(1, int(top_n))
        self.show_all = bool(show_all)
        self.enable_name_heuristic = bool(enable_name_heuristic)
        self.enable_mesh_detection = bool(enable_mesh_detection)
        self.enable_live_set = bool(enable_live_set)
        # Fraction of the frame's event span a resource must cover (by usage
        # lifetime) to be classed as persistent rather than a transient/poolable
        # candidate. Clamped to [0, 1].
        self.persistent_span_ratio = min(1.0, max(0.0, float(persistent_span_ratio)))
        self.collect_draw_names = bool(collect_draw_names)
        self.max_draw_names_per_buffer = max(0, int(max_draw_names_per_buffer))
        self.large_resource_threshold_bytes = (
            max(0, int(large_resource_threshold_mb)) * 1024 * 1024
        )
        # resource_id(str) -> {"roles": set(usage tokens), "events": set(event ids)}
        self._usage_by_rid = {}

    def run(self):
        # Resource usage is precomputed by RenderDoc during replay init, so
        # GetUsage() is a cheap lookup (no per-event replay). One pass over it
        # feeds render-target classification, mesh-buffer detection, the
        # live-set peak estimate, and the unreferenced-resource list.
        self._usage_by_rid = self._collect_usage_by_rid()

        rows, counts = self._collect_resource_rows()

        # Built once (no replay) and shared by mesh annotation and live-set so
        # the action walk is not repeated.
        name_by_event = {}
        if self.collect_draw_names or self.enable_live_set:
            name_by_event = self._build_event_name_map()

        mesh_stats = {
            "enabled": self.enable_mesh_detection,
            "method": "GetUsage",
            "resources_with_usage": len(self._usage_by_rid),
            "buffers_detected": 0,
        }

        if self.enable_mesh_detection:
            geometry_usage, mesh_stats = self._derive_geometry_usage(name_by_event)
            rows = self._annotate_mesh_buffers(rows, geometry_usage)

        live_set = None
        residency = None
        unreferenced = None
        if self.enable_live_set:
            live_set = self._compute_live_set(rows, name_by_event)
            residency = self._compute_residency(rows)
            unreferenced = self._collect_unreferenced(rows)

        return self._build_report(rows, counts, mesh_stats, live_set, unreferenced, residency)

    def _collect_usage_by_rid(self):
        """Map each resource id to its RenderDoc usage roles and event ids.

        Uses ReplayController.GetUsage(), which returns the precomputed
        EventUsage list for a resource across the whole capture. This replaces
        the previous approach of stepping SetFrameEvent() through every draw
        (a full replay per draw) and is both far faster and authoritative about
        how each resource is actually bound.
        """
        usage_by_rid = {}
        if not (self.enable_mesh_detection or self.enable_live_set):
            # The usage pass powers mesh detection, usage-based RT
            # classification, the live-set estimate, and unreferenced detection;
            # skip it only when all of those are disabled.
            return usage_by_rid

        resources = []
        try:
            resources.extend(self.controller.GetTextures())
        except Exception:
            pass
        try:
            resources.extend(self.controller.GetBuffers())
        except Exception:
            pass

        for res in resources:
            rid = safe_get(res, ["resourceId", "id"], None)
            if rid is None or is_null_rid(rid):
                continue
            try:
                usages = self.controller.GetUsage(rid)
            except Exception:
                continue
            if not usages:
                continue

            roles = set()
            events = set()
            for entry in usages:
                token = enum_text(safe_get(entry, ["usage"], None))
                if token:
                    roles.add(token)
                event_id = to_int(safe_get(entry, ["eventId", "eventID"], 0), 0)
                if event_id:
                    events.add(event_id)

            if roles or events:
                usage_by_rid[rid_key(rid)] = {"roles": roles, "events": events}

        return usage_by_rid

    def _collect_resource_rows(self):
        rows = []
        seen = set()
        texture_count = 0
        buffer_count = 0

        try:
            textures = self.controller.GetTextures()
        except Exception:
            textures = []

        try:
            buffers = self.controller.GetBuffers()
        except Exception:
            buffers = []

        for tex in textures:
            try:
                texture_count += 1
                row = self._make_texture_row(tex)
                key = ("Texture", row["resource_id"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
            except Exception:
                pass

        for buf in buffers:
            try:
                buffer_count += 1
                row = self._make_buffer_row(buf)
                key = ("Buffer", row["resource_id"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
            except Exception:
                pass

        return rows, {
            "textures": texture_count,
            "buffers": buffer_count,
            "rows": len(rows),
        }

    def _make_texture_row(self, tex):
        rid = safe_get(tex, ["resourceId", "id"], None)
        rid_s = rid_key(rid)
        name = self._resource_name(rid) or clean_name(safe_get(tex, ["name"], ""))
        if not name:
            name = "<unnamed texture>"

        width = to_int(safe_get(tex, ["width", "Width"], 1), 1)
        height = to_int(safe_get(tex, ["height", "Height"], 1), 1)
        depth = to_int(safe_get(tex, ["depth", "Depth"], 1), 1)
        arraysize = to_int(safe_get(tex, ["arraysize", "arraySize", "ArraySize"], 1), 1)
        mips = to_int(safe_get(tex, ["mips", "mipLevels", "MipLevels"], 1), 1)
        samples = to_int(safe_get(tex, ["msSamp", "samples", "sampleCount"], 1), 1)
        fmt = fmt_name(safe_get(tex, ["format", "Format"], None))

        byte_size = to_int(safe_get(tex, ["byteSize", "bytes", "size"], 0), 0)
        if byte_size > 0:
            size_bytes = byte_size
            note = "byteSize from RenderDoc"
        else:
            size_bytes, note = estimate_texture_bytes(tex)

        return {
            "kind": "Texture",
            "category": self._classify_texture(tex, name, rid_s),
            "resource_id": rid_s,
            "name": name,
            "format": fmt,
            "width": width,
            "height": height,
            "depth": depth,
            "array_size": arraysize,
            "mip_levels": mips,
            "msaa_samples": samples,
            "bytes": size_bytes,
            "mib": round(mib(size_bytes), 4),
            "gib": round(gib(size_bytes), 6),
            "note": note,
        }

    def _make_buffer_row(self, buf):
        rid = safe_get(buf, ["resourceId", "id"], None)
        rid_s = rid_key(rid)
        name = self._resource_name(rid) or clean_name(safe_get(buf, ["name"], ""))
        if not name:
            name = "<unnamed buffer>"

        length = to_int(safe_get(buf, ["length", "byteSize", "size"], 0), 0)
        stride = to_int(safe_get(buf, ["structureByteStride", "stride"], 0), 0)

        return {
            "kind": "Buffer",
            "category": self._classify_buffer(buf, name),
            "resource_id": rid_s,
            "name": name,
            "format": "",
            "width": None,
            "height": None,
            "depth": None,
            "array_size": None,
            "mip_levels": None,
            "msaa_samples": None,
            "bytes": length,
            "mib": round(mib(length), 4),
            "gib": round(gib(length), 6),
            "note": "stride=%d" % stride if stride else "",
        }

    def _resource_name(self, resource_id):
        if resource_id is None or is_null_rid(resource_id):
            return ""

        try:
            return clean_name(self.ctx.GetResourceName(resource_id))
        except Exception:
            return ""

    def _classify_texture(self, tex, name, resource_id=None):
        flags = enum_text(safe_get(tex, ["creationFlags", "flags"], ""))
        fmt = fmt_name(safe_get(tex, ["format"], None))
        dim = enum_text(safe_get(tex, ["dimension", "dim", "type"], ""))
        low = " ".join([flags, fmt, dim, name]).lower()

        if contains_any(low, ["swap", "backbuffer", "back buffer", "present"]):
            return "Texture/Swapchain"

        # Ground-truth render-target classification from RenderDoc usage data.
        # This is authoritative and takes precedence over the name heuristics
        # below, which only act as a fallback when usage data is unavailable.
        usage_category = self._usage_texture_category(resource_id)
        if usage_category:
            return usage_category
        if contains_any(
            low,
            ["depthtarget", "depth target", "depthstencil", "depth stencil", "dsview", "dsv"],
        ):
            return "RT/DepthStencil"
        if contains_any(low, ["colortarget", "color target", "rendertarget", "render target", "rtv"]):
            return "RT/Color"
        if contains_any(fmt.lower(), ["d16", "d24", "d32", "depth", "stencil"]):
            return "RT/DepthStencil"

        if self.enable_name_heuristic:
            n = name.lower()
            if contains_any(
                n,
                [
                    "shadowmap",
                    "shadow map",
                    "cascadeshadow",
                    "cascade shadow",
                    "depth",
                    "stencil",
                    "_cameradepth",
                    "camera depth",
                    "vsm",
                    "virtualshadow",
                    "virtual shadow",
                    "hzbin",
                    "hzb",
                    "hi-z",
                    "hiz",
                ],
            ):
                return "RT/DepthStencil"
            if contains_any(
                n,
                [
                    "rendertexture",
                    "render texture",
                    "rendertarget",
                    "render target",
                    "gbuffer",
                    "g-buffer",
                    "colorbuffer",
                    "color buffer",
                    "framebuffer",
                    "frame buffer",
                    "camera color",
                    "_cameracolor",
                    "history",
                    "velocity",
                    "motionvector",
                    "motion vector",
                    "normalbuffer",
                    "normal buffer",
                    "lighting",
                    "lightbuffer",
                    "light buffer",
                    "postprocess",
                    "post process",
                    "bloom",
                    "taa",
                    "ssao",
                    "ssr",
                    "rt_",
                    "_rt",
                    "temporaryrt",
                    "temporary rt",
                ],
            ):
                return "RT/Color"
            if contains_any(
                n,
                [
                    "virtualtexture",
                    "virtual texture",
                    "physicalpagepool",
                    "physical page pool",
                    "pagepool",
                    "page pool",
                    "pagetable",
                    "page table",
                ],
            ):
                return "Texture/VirtualTexture"

        if contains_any(dim, ["cube"]):
            return "Texture/Cubemap"
        if contains_any(dim, ["3d"]):
            return "Texture/3D"
        if to_int(safe_get(tex, ["arraysize", "arraySize"], 1), 1) > 1:
            return "Texture/Array"
        return "Texture/Regular"

    def _classify_buffer(self, buf, name):
        flags = enum_text(safe_get(buf, ["creationFlags", "flags"], ""))
        low = " ".join([flags, name]).lower()

        if contains_any(low, ["index", "ibuffer", "indexbuffer", "index buffer", "meshib", "mesh ib"]):
            return "Buffer/Index"
        if contains_any(low, ["vertex", "vbuffer", "vertexbuffer", "vertex buffer", "meshvb", "mesh vb"]):
            return "Buffer/Vertex"
        if contains_any(low, ["constant", "cbuffer", "constantbuffer", "constant buffer", "uniform"]):
            return "Buffer/Constant"
        if contains_any(low, ["indirect", "argument", "args", "drawargs", "dispatchargs"]):
            return "Buffer/Indirect"
        if contains_any(low, ["skin", "skinning", "bone", "bones", "blendshape", "morph"]):
            return "Buffer/Skinning-Morph"
        if contains_any(low, ["instance", "instancing", "instance data", "perinstance"]):
            return "Buffer/Instance"
        if contains_any(
            low,
            ["uav", "readwrite", "rwbuffer", "storage", "structured", "append", "consume", "byteaddress"],
        ):
            return "Buffer/UAV-Structured"
        if contains_any(low, ["acceleration", "blas", "tlas", "raytracing", "ray tracing"]):
            return "Buffer/RayTracingAS"
        if contains_any(low, ["gpu scene", "gpuscene", "primitive", "cluster", "meshlet", "visibility", "culling"]):
            return "Buffer/GPUScene-Culling"
        return "Buffer/Unknown"

    def _usage_texture_category(self, resource_id):
        """Classify a texture as a render/depth target from RenderDoc usage data.

        Returns a category string when the resource is bound as a color or
        depth-stencil target at any event, otherwise None so the caller can fall
        back to format/name heuristics.
        """
        info = self._usage_by_rid.get(rid_key(resource_id))
        if not info:
            return None

        roles_low = " ".join(info["roles"]).lower()
        # Depth takes precedence: a resource bound as a depth target is a depth
        # target even if also sampled later.
        if "depthstenciltarget" in roles_low:
            return "RT/DepthStencil"
        if "colortarget" in roles_low:
            return "RT/Color"
        return None

    def _derive_geometry_usage(self, name_by_event):
        """Build mesh geometry usage from the precomputed GetUsage map.

        Maps RenderDoc usage tokens to VB/IB roles and records the draw events
        (and optional draw names) that reference each geometry buffer. No replay
        stepping is performed.
        """
        usage = defaultdict(
            lambda: {
                "roles": set(),
                "events": set(),
                "draw_names": set(),
                "attrs": set(),
            }
        )

        for rid_s, info in self._usage_by_rid.items():
            roles = set()
            for token in info["roles"]:
                low = token.lower()
                if "indexbuffer" in low:
                    roles.add("IB")
                elif "vertexbuffer" in low:
                    roles.add("VB")
            if not roles:
                continue

            entry = usage[rid_s]
            entry["roles"].update(roles)
            entry["events"].update(info["events"])

            if self.collect_draw_names:
                for event_id in sorted(info["events"]):
                    if len(entry["draw_names"]) >= self.max_draw_names_per_buffer:
                        break
                    draw_name = name_by_event.get(event_id)
                    if draw_name:
                        entry["draw_names"].add(draw_name)

        stats = {
            "enabled": True,
            "method": "GetUsage",
            "resources_with_usage": len(self._usage_by_rid),
            "buffers_detected": len(usage),
        }
        return usage, stats

    def _build_event_name_map(self):
        """Map draw event ids to action names by walking actions once (no replay)."""
        name_by_event = {}
        try:
            structured_file = self.controller.GetStructuredFile()
        except Exception:
            structured_file = None

        for action in walk_draws(get_root_actions(self.controller)):
            event_id = get_action_event_id(action)
            if not event_id:
                continue

            name = ""
            if structured_file is not None:
                try:
                    name = clean_name(action.GetName(structured_file))
                except Exception:
                    name = ""
            if not name:
                name = clean_name(safe_get(action, ["name"], ""))
            if name:
                name_by_event[event_id] = name

        return name_by_event

    def _annotate_mesh_buffers(self, rows, geometry_usage):
        row_by_rid = {}
        for row in rows:
            if row.get("kind") == "Buffer":
                row_by_rid[row.get("resource_id")] = row

        for rid, info in geometry_usage.items():
            if rid not in row_by_rid:
                continue

            row = row_by_rid[rid]
            row["category"] = geometry_category_from_roles(info["roles"])
            mesh_note_parts = [
                "meshRoles=%s" % ",".join(sorted(info["roles"])),
                "drawEvents=%d" % len(info["events"]),
            ]
            attrs = ",".join(sorted(info["attrs"]))
            draw_names = " | ".join(sorted(info["draw_names"]))
            if attrs:
                mesh_note_parts.append("attrs=[%s]" % attrs)
            if draw_names:
                mesh_note_parts.append("draws=[%s]" % draw_names)

            mesh_note = " ".join(mesh_note_parts)
            old_note = row.get("note") or ""
            row["note"] = old_note + " | " + mesh_note if old_note else mesh_note

        return rows

    def _sweep_live_peak(self, rows):
        """Sweep-line peak of simultaneously-live bytes over the given rows.

        Each resource's lifetime is approximated by the [first, last] span of the
        events that reference it (from GetUsage). Returns
        (peak_bytes, peak_event, lifetimes) where lifetimes is the list of
        (first, last, row) for rows that have usage and positive size.
        """
        # (event, order, delta): +bytes when a resource enters, -bytes after it
        # leaves. Subtractions at a coordinate are applied before additions so
        # back-to-back lifetimes (lastA + 1 == firstB) do not falsely overlap.
        deltas = []
        lifetimes = []
        for row in rows:
            info = self._usage_by_rid.get(row.get("resource_id"))
            if not info or not info["events"]:
                continue
            size = row.get("bytes", 0)
            if size <= 0:
                continue
            first = min(info["events"])
            last = max(info["events"])
            deltas.append((first, 1, size))      # enter (order 1 = after leaves)
            deltas.append((last + 1, 0, -size))  # leave (order 0 = first)
            lifetimes.append((first, last, row))

        peak_bytes = 0
        peak_event = None
        running = 0
        for event_id, _order, delta in sorted(deltas, key=lambda d: (d[0], d[1])):
            running += delta
            if running > peak_bytes:
                peak_bytes = running
                peak_event = event_id

        return peak_bytes, peak_event, lifetimes

    def _build_live_timeline(self, lifetimes, name_by_event):
        """Build a compact event timeline from sweep-line lifetime deltas."""
        deltas = []
        for first, last, row in lifetimes:
            size = row.get("bytes", 0)
            if size <= 0:
                continue
            deltas.append((first, 1, size))
            deltas.append((last + 1, 0, -size))

        timeline = []
        running = 0
        current_event = None
        pending_delta = 0
        for event_id, order, delta in sorted(deltas, key=lambda d: (d[0], d[1])):
            if current_event is not None and event_id != current_event:
                if pending_delta:
                    running += pending_delta
                    timeline.append(
                        {
                            "event_id": current_event,
                            "bytes": running,
                            "mib": round(mib(running), 4),
                            "event_name": name_by_event.get(current_event, ""),
                        }
                    )
                    pending_delta = 0
            current_event = event_id
            pending_delta += delta
        if current_event is not None and pending_delta:
            running += pending_delta
            timeline.append(
                {
                    "event_id": current_event,
                    "bytes": running,
                    "mib": round(mib(running), 4),
                    "event_name": name_by_event.get(current_event, ""),
                }
            )
        return timeline

    def _compute_live_set(self, rows, name_by_event):
        """Estimate the peak simultaneously-live resource memory across the frame.

        A sweep line over usage lifetimes yields, for every event, the sum of
        bytes of all resources whose lifetime covers it; the maximum is the peak
        working-set estimate.

        Unlike the naive grand total this does NOT double-count resources whose
        lifetimes never overlap (e.g. pooled/transient render targets that the
        engine reuses), so it is a closer proxy for real VRAM pressure. It is
        still an estimate: it reflects usage liveness, not driver allocation
        liveness or heap aliasing, and resources with usage gaps are treated as
        live across the whole gap.
        """
        peak_bytes, peak_event, lifetimes = self._sweep_live_peak(rows)
        considered = len(lifetimes)

        # Category breakdown of the resources alive at the peak event.
        peak_by_category = defaultdict(int)
        peak_count = 0
        peak_rows = []
        if peak_event is not None:
            for first, last, row in lifetimes:
                if first <= peak_event <= last:
                    peak_by_category[row["category"]] += row["bytes"]
                    peak_count += 1
                    peak_rows.append(row)

        peak_categories = [
            {
                "category": category,
                "bytes": size,
                "mib": round(mib(size), 4),
                "percent": round((100.0 * size / peak_bytes) if peak_bytes > 0 else 0.0, 4),
            }
            for category, size in sorted(peak_by_category.items(), key=lambda item: item[1], reverse=True)
        ]

        grand = sum(row.get("bytes", 0) for row in rows)
        peak_rows.sort(key=lambda row: row.get("bytes", 0), reverse=True)
        peak_limit = len(peak_rows) if self.show_all else self.top_n
        return {
            "peak_bytes": peak_bytes,
            "peak_mib": round(mib(peak_bytes), 4),
            "peak_gib": round(gib(peak_bytes), 6),
            "peak_event_id": peak_event,
            "peak_event_name": name_by_event.get(peak_event, "") if peak_event is not None else "",
            "peak_live_resources": peak_count,
            "percent_of_grand_total": round((100.0 * peak_bytes / grand) if grand > 0 else 0.0, 4),
            "resources_considered": considered,
            "peak_categories": peak_categories,
            "peak_resources": [compact_row(row) for row in peak_rows[:peak_limit]],
            "timeline": self._build_live_timeline(lifetimes, name_by_event),
            "note": "Peak of simultaneously-live resources by usage lifetime; lower than the grand total by the amount of non-overlapping transient/pooled memory.",
        }

    def _compute_residency(self, rows):
        """Split resident memory into persistent vs transient (poolable) parts.

        Answers "of all this resident memory, how much could be reclaimed?".
        Every referenced resource is classed by how much of the frame its usage
        lifetime spans:

        - persistent: span covers >= persistent_span_ratio of the frame. It must
          stay resident throughout, so it can only be shrunk (smaller format /
          resolution / pool-size settings), not pooled away.
        - transient: a short-lived candidate whose memory could be reused by
          another non-overlapping transient resource.

        Cross-frame caches/pools/history are resident for the whole session but
        are only used in a narrow window of a single frame, so the span test
        alone would misclassify them as transient and overstate the reducible
        figure. When name heuristics are enabled, resources whose names match
        known-persistent patterns are forced into the persistent bucket
        regardless of span (reported as persistent.name_forced_*).

        Running the sweep-line peak over only the transient set gives the memory
        that set would need if perfectly pooled; the difference from its naive
        sum is the poolable headroom. Combined with the persistent floor and the
        unreferenced bytes this yields a theoretical minimum resident estimate
        and an upper bound on what is reducible.

        Caveats: this is usage-lifetime liveness, not driver allocation
        liveness; the "reducible" figure is an optimistic upper bound that
        assumes perfect pooling and that the engine does not already alias these
        resources.
        """
        all_events = set()
        for info in self._usage_by_rid.values():
            all_events.update(info["events"])
        if not all_events:
            return None

        frame_min = min(all_events)
        frame_max = max(all_events)
        frame_span = max(1, frame_max - frame_min)

        persistent_rows = []
        transient_rows = []
        name_forced_count = 0
        name_forced_bytes = 0
        for row in rows:
            info = self._usage_by_rid.get(row.get("resource_id"))
            if not info or not info["events"]:
                continue  # unreferenced; reported separately
            span = max(info["events"]) - min(info["events"])
            spans_frame = (float(span) / frame_span) >= self.persistent_span_ratio
            # Cross-frame caches/pools/history are resident the whole session but
            # are only *used* in a narrow window of this single frame, so the
            # span test alone misclassifies them as transient and inflates the
            # reducible estimate. A name override forces such known-persistent
            # resources back into the persistent bucket.
            forced = (
                not spans_frame
                and self.enable_name_heuristic
                and is_persistent_by_name(row.get("name", ""))
            )
            if spans_frame or forced:
                persistent_rows.append(row)
                if forced:
                    name_forced_count += 1
                    name_forced_bytes += row.get("bytes", 0)
            else:
                transient_rows.append(row)

        persistent_bytes = sum(row.get("bytes", 0) for row in persistent_rows)
        transient_bytes = sum(row.get("bytes", 0) for row in transient_rows)
        transient_peak, _peak_event, _lifetimes = self._sweep_live_peak(transient_rows)
        poolable_headroom = max(0, transient_bytes - transient_peak)

        unreferenced_bytes = sum(
            row.get("bytes", 0)
            for row in rows
            if not self._usage_by_rid.get(row.get("resource_id"), {}).get("events")
        )
        grand = sum(row.get("bytes", 0) for row in rows)
        theoretical_min = persistent_bytes + transient_peak
        reducible = max(0, grand - theoretical_min)  # = poolable_headroom + unreferenced_bytes

        persistent_rows.sort(key=lambda row: row.get("bytes", 0), reverse=True)
        transient_rows.sort(key=lambda row: row.get("bytes", 0), reverse=True)
        p_limit = len(persistent_rows) if self.show_all else self.top_n
        t_limit = len(transient_rows) if self.show_all else self.top_n

        return {
            "frame_event_range": [frame_min, frame_max],
            "persistent_span_ratio": self.persistent_span_ratio,
            "persistent": {
                "bytes": persistent_bytes,
                "mib": round(mib(persistent_bytes), 4),
                "count": len(persistent_rows),
                "percent_of_grand_total": round((100.0 * persistent_bytes / grand) if grand > 0 else 0.0, 4),
                "name_forced_count": name_forced_count,
                "name_forced_mib": round(mib(name_forced_bytes), 4),
                "comment": "Resident throughout the frame; reduce via smaller format/resolution or pool-size settings. name_forced_* are narrow-usage resources reclassified as persistent by name (cross-frame caches/pools/history).",
                "top_resources": [compact_row(row) for row in persistent_rows[:p_limit]],
            },
            "transient": {
                "bytes": transient_bytes,
                "mib": round(mib(transient_bytes), 4),
                "count": len(transient_rows),
                "percent_of_grand_total": round((100.0 * transient_bytes / grand) if grand > 0 else 0.0, 4),
                "pooled_peak_bytes": transient_peak,
                "pooled_peak_mib": round(mib(transient_peak), 4),
                "poolable_headroom_bytes": poolable_headroom,
                "poolable_headroom_mib": round(mib(poolable_headroom), 4),
                "comment": "Short-lived; pooled_peak is what this set needs if perfectly reused, poolable_headroom is what pooling could reclaim.",
                "top_resources": [compact_row(row) for row in transient_rows[:t_limit]],
            },
            "unreferenced_bytes": unreferenced_bytes,
            "unreferenced_mib": round(mib(unreferenced_bytes), 4),
            "theoretical_min_resident_bytes": theoretical_min,
            "theoretical_min_resident_mib": round(mib(theoretical_min), 4),
            "reducible_upper_bound_bytes": reducible,
            "reducible_upper_bound_mib": round(mib(reducible), 4),
            "note": "Optimistic decomposition by usage lifetime: persistent (shrink-only) + transient (poolable) + unreferenced (freeable). reducible_upper_bound = poolable_headroom + unreferenced assumes perfect pooling and no existing engine aliasing.",
        }

    def _collect_unreferenced(self, rows):
        """List resources never referenced by any event in the captured frame.

        A resource is reported as unreferenced when GetUsage returned no events
        for it. These are candidates for wasted memory, but the list may also
        include legitimately idle resources (staging/upload buffers, the current
        backbuffer before first use) and any resource whose usage query failed.
        """
        unreferenced = [
            row
            for row in rows
            if not self._usage_by_rid.get(row.get("resource_id"), {}).get("events")
        ]
        unreferenced.sort(key=lambda row: row.get("bytes", 0), reverse=True)

        total_bytes = sum(row.get("bytes", 0) for row in unreferenced)
        limit = len(unreferenced) if self.show_all else self.top_n
        return {
            "count": len(unreferenced),
            "total_bytes": total_bytes,
            "total_mib": round(mib(total_bytes), 4),
            "note": "Resources with no recorded usage in this frame; may include idle/staging resources or failed usage queries, not only waste.",
            "top_resources": [compact_row(row) for row in unreferenced[:limit]],
        }

    def _build_report(self, rows, counts, mesh_stats, live_set=None, unreferenced=None, residency=None):
        rows_sorted = sorted(rows, key=lambda row: row.get("bytes", 0), reverse=True)
        total_by_kind = defaultdict(int)
        total_by_category = defaultdict(int)
        count_by_category = defaultdict(int)

        for row in rows:
            total_by_kind[row["kind"]] += row["bytes"]
            total_by_category[row["category"]] += row["bytes"]
            count_by_category[row["category"]] += 1

        grand = sum(row["bytes"] for row in rows)
        categories = []
        for category, size in sorted(total_by_category.items(), key=lambda item: item[1], reverse=True):
            categories.append(
                {
                    "category": category,
                    "bytes": size,
                    "mib": round(mib(size), 4),
                    "gib": round(gib(size), 6),
                    "percent": round((100.0 * size / grand) if grand > 0 else 0.0, 4),
                    "count": count_by_category[category],
                }
            )

        mesh_rows = sorted(
            [row for row in rows if str(row.get("category", "")).startswith("Mesh/")],
            key=lambda row: row.get("bytes", 0),
            reverse=True,
        )
        mesh_total = sum(row["bytes"] for row in mesh_rows)
        mesh_by_category = defaultdict(int)
        mesh_count_by_category = defaultdict(int)
        for row in mesh_rows:
            mesh_by_category[row["category"]] += row["bytes"]
            mesh_count_by_category[row["category"]] += 1

        mesh_categories = []
        for category, size in sorted(mesh_by_category.items(), key=lambda item: item[1], reverse=True):
            mesh_categories.append(
                {
                    "category": category,
                    "bytes": size,
                    "mib": round(mib(size), 4),
                    "gib": round(gib(size), 6),
                    "percent": round((100.0 * size / mesh_total) if mesh_total > 0 else 0.0, 4),
                    "count": mesh_count_by_category[category],
                }
            )

        limit = len(rows_sorted) if self.show_all else self.top_n
        large_resources = [
            row for row in rows_sorted if row.get("bytes", 0) >= self.large_resource_threshold_bytes
        ][:limit]

        return {
            "scope": "API-visible captured resource estimated size; not exact driver VRAM usage",
            "options": {
                "top_n": self.top_n,
                "show_all": self.show_all,
                "enable_name_heuristic": self.enable_name_heuristic,
                "enable_mesh_detection": self.enable_mesh_detection,
                "enable_live_set": self.enable_live_set,
                "persistent_span_ratio": self.persistent_span_ratio,
                "collect_draw_names": self.collect_draw_names,
                "max_draw_names_per_buffer": self.max_draw_names_per_buffer,
                "large_resource_threshold_mb": self.large_resource_threshold_bytes // (1024 * 1024),
            },
            "counts": counts,
            "totals": {
                "grand_bytes": grand,
                "grand_mib": round(mib(grand), 4),
                "grand_gib": round(gib(grand), 6),
                "textures_bytes": total_by_kind.get("Texture", 0),
                "textures_mib": round(mib(total_by_kind.get("Texture", 0)), 4),
                "buffers_bytes": total_by_kind.get("Buffer", 0),
                "buffers_mib": round(mib(total_by_kind.get("Buffer", 0)), 4),
            },
            "categories": categories,
            "mesh": {
                "stats": mesh_stats,
                "total_bytes": mesh_total,
                "total_mib": round(mib(mesh_total), 4),
                "count": len(mesh_rows),
                "categories": mesh_categories,
                "top_resources": [compact_row(row) for row in mesh_rows[:limit]],
            },
            "live_set": live_set,
            "residency": residency,
            "unreferenced": unreferenced,
            "top_resources": [compact_row(row) for row in rows_sorted[:limit]],
            "large_resources": [compact_row(row) for row in large_resources],
            "notes": [
                "Driver private allocations, heap padding/alignment, compression metadata, descriptor heaps, shader caches, command allocators, query heaps, and RenderDoc replay overhead are not included.",
                "Summing resources can exceed real frame peak memory when transient resources alias; see live_set.peak_bytes for a tighter peak estimate.",
                "Sparse, tiled, virtual, or partially resident resources may be overestimated from their visible descriptions.",
                "Mesh buffer categories and render-target classification are derived from RenderDoc resource usage (GetUsage); per-instance vertex streams are reported as vertex buffers.",
                "live_set is a peak working-set estimate from usage lifetimes (not driver allocation lifetimes); unreferenced lists resources with no recorded usage this frame.",
                "residency splits resident memory into persistent (shrink-only) vs transient (poolable) and estimates a reducible upper bound; persistent + transient + unreferenced == grand total.",
            ],
        }


def compact_row(row):
    return {
        "kind": row.get("kind"),
        "category": row.get("category"),
        "bytes": row.get("bytes", 0),
        "mib": row.get("mib", 0.0),
        "gib": row.get("gib", 0.0),
        "resource_id": row.get("resource_id"),
        "name": row.get("name"),
        "format": row.get("format"),
        "width": row.get("width"),
        "height": row.get("height"),
        "depth": row.get("depth"),
        "array_size": row.get("array_size"),
        "mip_levels": row.get("mip_levels"),
        "msaa_samples": row.get("msaa_samples"),
        "note": row.get("note"),
    }


def safe_get(obj, names, default=None):
    if isinstance(names, str):
        names = [names]
    for name in names:
        try:
            if hasattr(obj, name):
                return getattr(obj, name)
        except Exception:
            pass
    return default


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def rid_key(resource_id):
    try:
        return str(resource_id)
    except Exception:
        return repr(resource_id)


def is_null_rid(resource_id):
    if resource_id is None:
        return True
    try:
        return resource_id == rd.ResourceId.Null()
    except Exception:
        pass
    return rid_key(resource_id) in ["", "0", "ResourceId::Null()", "ResourceId()"]


def mib(num_bytes):
    return float(num_bytes) / 1024.0 / 1024.0


def gib(num_bytes):
    return float(num_bytes) / 1024.0 / 1024.0 / 1024.0


def clean_name(value):
    if value is None:
        return ""
    try:
        value = str(value)
    except Exception:
        value = repr(value)
    return value.replace("\n", " ").replace("\r", " ").strip()


def enum_text(value):
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return repr(value)


def contains_any(text, keywords):
    text = (text or "").lower()
    return any(keyword.lower() in text for keyword in keywords)


def fmt_name(fmt):
    if fmt is None:
        return "UnknownFormat"
    try:
        name = fmt.Name()
        if name:
            return str(name)
    except Exception:
        pass
    for attr in ["strname", "name"]:
        try:
            name = getattr(fmt, attr)
            if name:
                return str(name)
        except Exception:
            pass
    return str(fmt)


def resource_format_key(fmt):
    name = fmt_name(fmt)
    special = safe_get(fmt, ["specialFormat", "special"], "")
    comp_type = safe_get(fmt, ["compType", "type"], "")
    comp_count = safe_get(fmt, ["compCount", "componentCount"], "")
    comp_width = safe_get(fmt, ["compByteWidth", "componentByteWidth"], "")
    return ("%s %s %s %s %s" % (name, enum_text(special), enum_text(comp_type), enum_text(comp_count), enum_text(comp_width))).upper()


def parse_astc_block_size(key):
    match = re.search(r"ASTC[^0-9]*(\d+)\s*[Xx]\s*(\d+)", key)
    if match:
        return int(match.group(1)), int(match.group(2)), 16
    return None


def format_layout(fmt):
    """Resolve the per-texel memory layout of a texture format.

    Order matters. Block-compressed and specially-packed formats must be matched
    by their distinctive names first, because their byte size cannot be derived
    from component count * component width. Everything else falls through to the
    authoritative component-width product. This avoids the earlier short-token
    matching (e.g. "R16"/"R32") that mis-classified multi-component formats such
    as R16G16B16A16 or R32G32B32A32 and under-counted them by up to 4x.
    """
    key = resource_format_key(fmt)

    # --- Block-compressed formats (size cannot come from component width) ---
    if "BC1" in key or "BC4" in key:
        return {"mode": "block", "block_w": 4, "block_h": 4, "block_bytes": 8, "note": "BC1/BC4"}
    if any(value in key for value in ["BC2", "BC3", "BC5", "BC6", "BC7"]):
        return {"mode": "block", "block_w": 4, "block_h": 4, "block_bytes": 16, "note": "BC2/3/5/6/7"}
    if "ETC1" in key or "ETC2" in key:
        if "RGBA" in key or "A8" in key:
            return {"mode": "block", "block_w": 4, "block_h": 4, "block_bytes": 16, "note": "ETC2 RGBA estimated"}
        return {"mode": "block", "block_w": 4, "block_h": 4, "block_bytes": 8, "note": "ETC1/ETC2 RGB"}
    if "EAC" in key:
        if "RG" in key:
            return {"mode": "block", "block_w": 4, "block_h": 4, "block_bytes": 16, "note": "EAC RG"}
        return {"mode": "block", "block_w": 4, "block_h": 4, "block_bytes": 8, "note": "EAC R"}

    astc = parse_astc_block_size(key)
    if astc:
        block_w, block_h, block_bytes = astc
        return {
            "mode": "block",
            "block_w": block_w,
            "block_h": block_h,
            "block_bytes": block_bytes,
            "note": "ASTC %dx%d" % (block_w, block_h),
        }

    # --- Specially-packed formats (size != compCount * compByteWidth) ---
    # Only distinctive, full tokens are used here so they cannot false-match a
    # wider multi-component name (the cause of the previous under-counting).
    if "D32S8" in key or "R32G8" in key:
        return {"mode": "pixel", "bytes_per_pixel": 8, "note": "D32S8/R32G8 estimated"}
    if any(
        value in key
        for value in ["R10G10B10A2", "B10G10R10A2", "R11G11B10", "R9G9B9E5", "D24S8", "R24G8", "D24"]
    ):
        return {"mode": "pixel", "bytes_per_pixel": 4, "note": "packed/depth 4Bpp estimated"}
    if any(value in key for value in ["R5G6B5", "B5G6R5", "R5G5B5A1", "B5G5R5A1", "R4G4B4A4", "B4G4R4A4"]):
        return {"mode": "pixel", "bytes_per_pixel": 2, "note": "packed 2Bpp estimated"}

    # --- Plain uncompressed formats: authoritative component-width product ---
    # Covers R8 (1*1), R16x* / D16 (n*2), R32x* / D32 (n*4), RGBA16F (4*2=8),
    # RGBA32F (4*4=16), etc. without relying on fragile name substrings.
    comp_count = to_int(safe_get(fmt, ["compCount", "componentCount"], 0), 0)
    comp_width = to_int(safe_get(fmt, ["compByteWidth", "componentByteWidth"], 0), 0)
    if comp_count > 0 and comp_width > 0:
        return {
            "mode": "pixel",
            "bytes_per_pixel": comp_count * comp_width,
            "note": "%d comps * %d bytes" % (comp_count, comp_width),
        }
    return {"mode": "pixel", "bytes_per_pixel": 4, "note": "fallback RGBA8 estimate"}


def texture_mip_bytes(width, height, depth, samples, layout):
    width = max(1, int(width))
    height = max(1, int(height))
    depth = max(1, int(depth))
    samples = max(1, int(samples))
    if layout["mode"] == "block":
        blocks_x = max(1, int(math.ceil(float(width) / float(layout["block_w"]))))
        blocks_y = max(1, int(math.ceil(float(height) / float(layout["block_h"]))))
        return blocks_x * blocks_y * depth * layout["block_bytes"] * samples
    return width * height * depth * layout["bytes_per_pixel"] * samples


def estimate_texture_bytes(tex):
    width = to_int(safe_get(tex, ["width", "Width"], 1), 1)
    height = to_int(safe_get(tex, ["height", "Height"], 1), 1)
    depth0 = to_int(safe_get(tex, ["depth", "Depth"], 1), 1)
    arraysize = to_int(safe_get(tex, ["arraysize", "arraySize", "ArraySize"], 1), 1)
    mips = to_int(safe_get(tex, ["mips", "mipLevels", "MipLevels"], 1), 1)
    samples = to_int(safe_get(tex, ["msSamp", "samples", "sampleCount"], 1), 1)
    dim_text = enum_text(safe_get(tex, ["dimension", "dim", "type"], ""))
    is_3d = "3D" in dim_text.upper()
    layout = format_layout(safe_get(tex, ["format", "Format"], None))
    total = 0
    for mip_level in range(max(1, mips)):
        mip_width = max(1, width >> mip_level)
        mip_height = max(1, height >> mip_level)
        if is_3d:
            mip_depth = max(1, depth0 >> mip_level)
            slice_count = 1
        else:
            mip_depth = max(1, depth0)
            slice_count = max(1, arraysize)
        # MSAA stores resolved samples only at the full-resolution surface; mip
        # chains (when reported alongside samples) are single-sampled, so the
        # sample multiplier is applied at mip 0 only to avoid over-counting.
        mip_samples = samples if mip_level == 0 else 1
        total += texture_mip_bytes(mip_width, mip_height, mip_depth, mip_samples, layout) * slice_count
    return total, layout["note"]


def get_root_actions(controller):
    for method_name in ["GetRootActions", "GetDrawcalls"]:
        try:
            method = getattr(controller, method_name)
            actions = method()
            if actions is not None:
                return actions
        except Exception:
            pass
    return []


def walk_draws(actions):
    for action in actions:
        yield action
        children = safe_get(action, ["children"], None)
        if children:
            for child in walk_draws(children):
                yield child


def get_action_event_id(action):
    return to_int(safe_get(action, ["eventId", "eventID"], 0), 0)


# Name fragments that strongly indicate a cross-frame-persistent resource
# (streaming/virtual-texture caches, shadow/VSM physical pools, temporal history,
# runtime-resident arrays). Such resources are resident the whole session even
# when used only briefly within a single captured frame.
_PERSISTENT_NAME_HINTS = [
    "cache",
    "pool",
    "history",
    "streaming",
    "runtimeonly",
    "runtime only",
    "persistent",
    "vt-",
    "vt_",
    "virtualtexture",
    "virtual texture",
    "pagetable",
    "page table",
    "pagepool",
    "page pool",
    "physicalpage",
    "physical page",
    "vsm",
    "virtualshadow",
    "virtual shadow",
    "nanite",
]


def is_persistent_by_name(name):
    """Heuristic: does the resource name mark it as cross-frame persistent?"""
    return contains_any(name or "", _PERSISTENT_NAME_HINTS)


def geometry_category_from_roles(roles):
    roles = set(roles)
    if "VB" in roles and "IB" in roles:
        return "Mesh/Vertex+Index"
    if "VB" in roles:
        return "Mesh/VertexBuffer"
    if "IB" in roles:
        return "Mesh/IndexBuffer"
    if "Instance" in roles:
        return "Mesh/InstanceBuffer"
    return "Mesh/UnknownGeometry"
