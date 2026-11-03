"""Extract and verify the exact-real interpretation of a saved hard circuit.

Outward intervals enclose initialization and every unrolled tick. Disjoint
value clusters label Boolean predicates; aggregate transfers either establish
their logical meaning for every domain size or retain an opaque enclosing
hull. Only head decisions constant on every potentially reachable case yield
a formula. Current-contract upstream ADF proofs close that formula.

The logical checker handles its declared UCQ/degree-count fragments; unsupported
cases and resource limits are inconclusive. Sampled PyTorch execution is a bug
check, not a proof of IEEE/CUDA runtime equivalence. Reports state this semantic
boundary and the numerical library versions explicitly.
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .adf import ADFLibrary, augment_batch
    from .config import OFFICIAL_CONFIG
    from .model import NeuralNP4G
    from .preprocess import COUNT_CAP, CRISP_SCALE, EXISTS_MARGIN
    from .runtime import configure_logging, default_artifact_root
    from . import program_ir as ir
    from . import validated_numerics as vn
except ImportError:  # direct execution
    from adf import ADFLibrary, augment_batch  # type: ignore
    from config import OFFICIAL_CONFIG  # type: ignore
    from model import NeuralNP4G  # type: ignore
    from preprocess import COUNT_CAP, CRISP_SCALE, EXISTS_MARGIN  # type: ignore
    from runtime import configure_logging, default_artifact_root  # type: ignore
    import program_ir as ir  # type: ignore
    import validated_numerics as vn  # type: ignore

_CRISP_SCALE = CRISP_SCALE
_EXISTS_MARGIN = EXISTS_MARGIN
_COUNT_CAP = COUNT_CAP

logger = logging.getLogger("official_neural_np4g.extract")


# ---------------------------------------------------------------------------
# Memory guard: cone enumeration / boolean minimisation has no
# a-priori memory bound, so extraction must be able to give up SAFELY instead
# of freezing the machine.
# ---------------------------------------------------------------------------


def _process_rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        class _PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        # argtypes matter: without them the 64-bit pseudo-handle is truncated
        # and the call fails silently (returns 0).
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(_PMC), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return int(pmc.WorkingSetSize)
    except Exception:
        pass
    return None


def _system_available_bytes() -> int | None:
    """Probe available memory on supported platforms, with a Windows fallback."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        class _MSX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        st = _MSX()
        st.dwLength = ctypes.sizeof(_MSX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys)
    except Exception:
        pass
    return None


def _start_memory_guard(*, cap_gb: float, min_avail_gb: float, report_out: Path | None,
                        reports: list, current_task: list) -> None:
    """Watchdog thread: when this process's RSS exceeds ``cap_gb`` or system
    available memory drops below ``min_avail_gb``, record the current task as
    EXTRACTION_FAILED(memory cap), flush the report JSON, and hard-exit(97).

    A hard exit is deliberate: the blowup happens deep inside enumeration /
    simplification loops that cannot be unwound cooperatively without
    instrumenting every loop. Exiting with the report already written keeps
    the battery resumable and the failure honestly recorded.
    """
    if _process_rss_bytes() is None:
        logger.warning("Memory guard disabled: RSS probe unavailable")
        return

    def _run() -> None:
        while True:
            time.sleep(2.0)
            rss = _process_rss_bytes()
            avail = _system_available_bytes()
            breach = None
            if rss is not None and rss > cap_gb * 2**30:
                breach = f"process rss {rss / 2**30:.1f}GB > cap {cap_gb:g}GB"
            elif avail is not None and avail < min_avail_gb * 2**30:
                breach = f"system available {avail / 2**30:.1f}GB < floor {min_avail_gb:g}GB"
            if breach is None:
                continue
            task = current_task[0]
            logger.critical("Memory guard tripped (%s); aborting extraction of %r", breach, task)
            if not any(r.get("task") == task for r in reports):
                reports.append({"task": task, "status": "EXTRACTION_FAILED",
                                "reason": f"memory cap: {breach}"})
            if report_out is not None:
                try:
                    report_out.parent.mkdir(parents=True, exist_ok=True)
                    report_out.write_text(json.dumps(reports, indent=2, default=str),
                                          encoding="utf-8")
                    logger.info("Wrote memory-guard partial report=%s", report_out)
                except Exception:
                    logger.exception("Memory guard could not write report=%s", report_out)
            os._exit(97)

    threading.Thread(target=_run, daemon=True, name="np4g-mem-guard").start()


def _start_time_guard(*, per_task_sec: float, report_out: Path | None,
                      reports: list, current_task: list, task_started: list) -> None:
    """Watchdog thread: when the CURRENT task has been extracting for longer
    than ``per_task_sec`` wall-clock seconds, record it as
    EXTRACTION_FAILED(time cap) and hard-exit(96).

    Cone enumeration is not only memory-unbounded (see the memory guard) but
    also TIME-unbounded: a wide circuit (e.g. the cexistsp pairwise-join pool
    at full width feeds C^2 join channels into the core, and the per-channel
    value-tracking cone can blow up combinatorially with tiny RSS). Such a run
    pins a CPU core for hours/days without ever tripping the memory guard, so a
    time budget is the honest terminator: 'could not extract within the compute
    budget' is a legitimate EXTRACTION_FAILED under the binary criterion.
    """
    if per_task_sec <= 0:
        return

    def _run() -> None:
        while True:
            time.sleep(5.0)
            t0 = task_started[0]
            task = current_task[0]
            if t0 is None or task is None:
                continue
            elapsed = time.time() - t0
            if elapsed < per_task_sec:
                continue
            logger.critical(
                "Time guard tripped task=%r elapsed=%.0fs cap=%.0fs; aborting extraction",
                task, elapsed, per_task_sec,
            )
            if not any(r.get("task") == task for r in reports):
                reports.append({"task": task, "status": "EXTRACTION_FAILED",
                                "reason": f"time cap: {elapsed:.0f}s > {per_task_sec:.0f}s"})
            if report_out is not None:
                try:
                    report_out.parent.mkdir(parents=True, exist_ok=True)
                    report_out.write_text(json.dumps(reports, indent=2, default=str),
                                          encoding="utf-8")
                    logger.info("Wrote time-guard partial report=%s", report_out)
                except Exception:
                    logger.exception("Time guard could not write report=%s", report_out)
            os._exit(96)

    threading.Thread(target=_run, daemon=True, name="np4g-time-guard").start()


FAMILY_RELS = ["father", "mother", "son", "daughter"]
GRAPH_RELS = ["adj", "eye"]
FAMILY_UNS: list[str] = ["_zero"]     # family input1 is identically zero
GRAPH_UNS = ["red"]

MAX_COMBOS = 1 << 22
MAX_HEAD_COMBOS = 1 << 24   # the head-joint pass stores only a bool table, so
#                             it can afford a larger enumeration than the
#                             per-channel value-tracking passes


# ============================================================================
# Value specs
# ============================================================================


@dataclass
class Choice:
    label: Any            # False/True (Boolean), int m (count), int k (value cluster)
    lo: float
    hi: float


@dataclass
class VarSpec:
    """One input node of the core block = one variable.

    kind[0] distinguishes the semantics; "vstate" marks a MULTI-VALUED state
    variable whose choices are value clusters (labels 0..k-1) -- its atoms are
    per-cluster value predicates, supplied at composition time.
    """
    kind: tuple           # semantic tag, e.g. ("state", c), ("rel", r), ...
    choices: list[Choice]

    @property
    def n(self) -> int:
        return len(self.choices)


def _bool_choices(f_lo, f_hi, t_lo, t_hi) -> list[Choice]:
    return [Choice(False, f_lo, f_hi), Choice(True, t_lo, t_hi)]


def _const_choices(lo, hi, label=None) -> list[Choice]:
    return [Choice(label, lo, hi)]


def merge_clusters(intervals: list[tuple[float, float]], gap: float = 0.01,
                   cap: int = 10) -> list[tuple[float, float]]:
    """Merge (lo, hi) intervals into <= cap clusters (nearest-gap merging)."""
    if not intervals:
        return []
    ivs = sorted(intervals)
    out = [list(ivs[0])]
    for lo, hi in ivs[1:]:
        if lo <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    while len(out) > cap:
        gaps = [(out[i + 1][0] - out[i][1], i) for i in range(len(out) - 1)]
        _, i = min(gaps)
        out[i][1] = max(out[i][1], out[i + 1][1])
        del out[i + 1]
    return [(lo, hi) for lo, hi in out]


MAX_VALUE_CLUSTERS = 24


class ChannelRanges:
    """Per-channel attainable state VALUE CLUSTERS -- no Boolean assumption.

    ok[c]: the attainable values of channel c fit in <= MAX_VALUE_CLUSTERS
    tight clusters (they always do for a hard circuit with pruned fan-in; the
    cap only guards enumeration cost). Each cluster becomes its own choice of
    the state variable, with its own VALUE PREDICATE `P_{c,k}` in the
    composed program. Being multi-valued is fine: the head is analysed
    JOINTLY with the final tick, and preprocess aggregations read a channel
    through sigma(scale*s) with a margin, which partitions the clusters into
    exactly-inactive (contribution 0) and active ones.
    """

    def __init__(self, C: int) -> None:
        self.C = C
        self.ok = [True] * C
        self.clusters: list[list[tuple[float, float]]] = [[(-1.0, 1.0)] for _ in range(C)]
        self.full = [(-1.0, 1.0)] * C

    def q_clusters(self, c: int) -> list[tuple[float, float]]:
        """sigma(scale * cluster) per cluster."""
        return [vn.sigmoid(*vn.scale(lo, hi, _CRISP_SCALE))
                for lo, hi in self.clusters[c]]

    def _partition(self, c: int, thresh: float) -> tuple[list[int], list[int]] | None:
        """Split clusters into (inactive, active) w.r.t. contribution relu(q - thresh):
        inactive -> exactly 0, active -> strictly positive; None if any cluster
        straddles the threshold."""
        inactive, active = [], []
        for k, (q_lo, q_hi) in enumerate(self.q_clusters(c)):
            if q_hi <= thresh:
                inactive.append(k)
            elif q_lo > thresh:
                active.append(k)
            else:
                return None
        return inactive, active

    def exists_spec(self, c: int) -> tuple[list[Choice], tuple[int, ...]] | None:
        if not self.ok[c]:
            return None
        part = self._partition(c, _EXISTS_MARGIN)
        if part is None:
            return None
        _, active = part
        if not active:
            return [Choice(False, 0.0, 0.0)], ()
        qs = self.q_clusters(c)
        lo = vn.add_down(min(qs[k][0] for k in active), -_EXISTS_MARGIN)
        hi = vn.add_up(max(qs[k][1] for k in active), -_EXISTS_MARGIN)
        return [Choice(False, 0.0, 0.0), Choice(True, lo, hi)], tuple(active)

    def forall_spec(self, c: int) -> tuple[list[Choice], tuple[int, ...]] | None:
        """Violation contribution relu((1-q) - margin): returns (choices,
        NON-violating cluster ids); value 1 iff every neighbour's state lies in
        a non-violating cluster."""
        if not self.ok[c]:
            return None
        # violating <=> (1 - q) > margin <=> q < 1 - margin
        inactive, active = [], []   # active = violating
        for k, (q_lo, q_hi) in enumerate(self.q_clusters(c)):
            v_lo, v_hi = vn.add_down(1.0, -q_hi), vn.add_up(1.0, -q_lo)
            if v_hi <= _EXISTS_MARGIN:
                inactive.append(k)
            elif v_lo > _EXISTS_MARGIN:
                active.append(k)
            else:
                return None
        if not active:
            return [Choice(True, 1.0, 1.0)], tuple(inactive)
        qs = self.q_clusters(c)
        f_lo = vn.add_down(0.5, min(qs[k][0] for k in active))
        f_hi = vn.add_up(0.5, max(qs[k][1] for k in active))
        return [Choice(False, f_lo, f_hi), Choice(True, 1.0, 1.0)], tuple(inactive)

    def join_spec(self, c: int) -> tuple[list[Choice], tuple[int, ...]] | None:
        if not self.ok[c]:
            return None
        part = self._partition(c, _EXISTS_MARGIN)  # placeholder split by q vs 0.5
        if part is None:
            return None
        _, active = part
        if not active:
            return [Choice(False, 0.0, 0.0)], ()
        qs = self.q_clusters(c)
        act_lo = min(qs[k][0] for k in active)
        act_hi = max(qs[k][1] for k in active)
        inact_hi = max((qs[k][1] for k in range(len(qs)) if k not in active), default=0.0)
        # product conditions: active*active strictly detected, anything with an
        # inactive side exactly zero
        if vn.mul_down(act_lo, act_lo) <= _EXISTS_MARGIN:
            return None
        if vn.mul_up(act_hi, inact_hi) > _EXISTS_MARGIN:
            return None
        return [Choice(False, 0.0, 0.0),
                Choice(True, vn.add_down(vn.mul_down(act_lo, act_lo), -_EXISTS_MARGIN),
                       vn.add_up(vn.mul_up(act_hi, act_hi), -_EXISTS_MARGIN))], tuple(active)

    def join_spec2(self, c1: int, c2: int) -> tuple[list[Choice], tuple[int, ...], tuple[int, ...]] | None:
        """Two-channel EXISTS-join (cexistsp pair c1, c2): crisp exactness needs
        every product with an inactive side to be zeroed by the margin and the
        active-active product to be strictly detected."""
        if not (self.ok[c1] and self.ok[c2]):
            return None
        p1 = self._partition(c1, _EXISTS_MARGIN)
        p2 = self._partition(c2, _EXISTS_MARGIN)
        if p1 is None or p2 is None:
            return None
        _, act1 = p1
        _, act2 = p2
        if not act1 or not act2:
            return [Choice(False, 0.0, 0.0)], tuple(act1), tuple(act2)
        q1, q2 = self.q_clusters(c1), self.q_clusters(c2)
        a1_lo = min(q1[k][0] for k in act1)
        a1_hi = max(q1[k][1] for k in act1)
        a2_lo = min(q2[k][0] for k in act2)
        a2_hi = max(q2[k][1] for k in act2)
        i1_hi = max((q1[k][1] for k in range(len(q1)) if k not in act1), default=0.0)
        i2_hi = max((q2[k][1] for k in range(len(q2)) if k not in act2), default=0.0)
        if vn.mul_down(a1_lo, a2_lo) <= _EXISTS_MARGIN:
            return None
        if vn.mul_up(a1_hi, i2_hi) > _EXISTS_MARGIN:
            return None
        if vn.mul_up(i1_hi, a2_hi) > _EXISTS_MARGIN:
            return None
        if vn.mul_up(i1_hi, i2_hi) > _EXISTS_MARGIN:
            return None
        return ([Choice(False, 0.0, 0.0),
                 Choice(True, vn.add_down(vn.mul_down(a1_lo, a2_lo), -_EXISTS_MARGIN),
                        vn.add_up(vn.mul_up(a1_hi, a2_hi), -_EXISTS_MARGIN))],
                tuple(act1), tuple(act2))

    def cge_spec(self, c: int) -> tuple[list[Choice], tuple[int, ...]] | None:
        """Binary count-threshold primitive cge{k}: output is EXACTLY {0,1}
        (straight-through forward), so both choices have zero interval width.
        Requires the same crisp partition as EXISTS (no cluster may straddle
        the contribution margin); the count of strictly-positive contributions
        is then an exact integer and never enters the cone as a graded value."""
        if not self.ok[c]:
            return None
        part = self._partition(c, _EXISTS_MARGIN)
        if part is None:
            return None
        _, active = part
        if not active:
            return [Choice(False, 0.0, 0.0)], ()
        return [Choice(False, 0.0, 0.0), Choice(True, 1.0, 1.0)], tuple(active)

    def count_spec(self, c: int) -> tuple[list[Choice], int, tuple[int, ...]] | None:
        if not self.ok[c]:
            return None
        part = self._partition(c, _EXISTS_MARGIN)
        if part is None:
            return None
        _, active = part
        if not active:
            return [Choice(0, 0.0, 0.0)], 0, ()
        qs = self.q_clusters(c)
        lo_c = vn.add_down(min(qs[k][0] for k in active), -_EXISTS_MARGIN)
        hi_c = vn.add_up(max(qs[k][1] for k in active), -_EXISTS_MARGIN)
        # Establish the tail by a lower bound, never by a rounded division.
        cap_m = next((k for k in range(1, 65)
                      if vn.mul_down(k, lo_c) >= _COUNT_CAP), None)
        if cap_m is None:
            return None
        out = [Choice(0, 0.0, 0.0)]
        for mval in range(1, cap_m):
            out.append(Choice(mval, min(vn.mul_down(mval, lo_c), _COUNT_CAP),
                              min(vn.mul_up(mval, hi_c), _COUNT_CAP)))
        out.append(Choice(cap_m, _COUNT_CAP, _COUNT_CAP))
        return out, cap_m, tuple(active)

# ============================================================================
# Circuit reading
# ============================================================================


def _state_choices(ranges: ChannelRanges, c: int) -> list[Choice]:
    """Multi-valued state variable: one choice per value cluster (int labels)."""
    if not ranges.ok[c]:
        return _const_choices(*ranges.full[c], "?")
    return [Choice(k, lo, hi) for k, (lo, hi) in enumerate(ranges.clusters[c])]


class CircuitReader:
    def __init__(self, model) -> None:
        self.model = model
        if not model.hard_forward:
            raise ValueError("extraction requires a hard-forward circuit")
        if model.state_update not in ("replace", "step"):
            raise ValueError("extraction supports state_update=replace/step")
        self.step_state = model.state_update == "step"
        if model.core.activation_name != "sigmoid" or model.adf_call_specs:
            raise ValueError("extraction requires sigmoid hidden activation and adf_mode=input")
        core = model.core
        with torch.no_grad():
            w_eff = core.effective_weight().detach().cpu().double().numpy()
            _, node_gate = core.soft_structure()
            self.node_gate = node_gate.detach().cpu().double().numpy()
            self.bias = core.bias.detach().cpu().double().numpy()
        self.W = w_eff
        self.Wp = np.clip(w_eff, 0.0, None)
        self.Wn = np.clip(w_eff, None, 0.0)
        if not all(np.isfinite(v).all() for v in (self.W, self.node_gate, self.bias)):
            raise ValueError("nonfinite saved circuit parameter")
        if not np.isin(self.node_gate, [0.0, 1.0]).all():
            raise ValueError("node gates must be binary")
        self.n_nodes = core.n_nodes
        self.d_in = core.d_in
        self.out_start = core.out_start
        self.out_end = core.out_end
        self.hidden_start = core.hidden_start
        self.circuit_ticks = core.num_ticks
        self.C = model.channels

    # ---------------- layout ----------------
    def input_layout(self, ranges: ChannelRanges) -> list[VarSpec]:
        """VarSpec per input node, with per-channel intervals from `ranges`.

        Whenever a preprocess aggregation's exactness condition fails for a channel
        (its false side would not be exactly zero, or its true side not
        strictly detected), the corresponding variable degrades to an opaque
        constant hull -- sound, but that variable then carries no formula.
        """
        model = self.model
        C = self.C

        def scaled(ch: list[Choice], w: float) -> list[Choice]:
            return [Choice(x.label, *vn.scale(x.lo, x.hi, w)) for x in ch]

        specs: list[VarSpec] = []
        if model.target_arity_value == 2:
            gate_w = self._gate_weights("join_aggregation_gate_logits", model.join_aggregation_ops)
            _, rel_gate = model.input_channel_gate_weights(torch.device("cpu"))
            rel_gate = [float(v) for v in rel_gate.detach().cpu()]
            for c in range(C):
                specs.append(VarSpec(("vstate", c), _state_choices(ranges, c)))
            for c in range(C):
                specs.append(VarSpec(("vstate_t", c), _state_choices(ranges, c)))
            for oi, op in enumerate(model.join_aggregation_ops):
                if op not in ("cexists", "cexistsp", "cexistsp_r"):
                    raise ValueError(f"unsupported join aggregation: {op}")
                w = gate_w[oi]
                if op == "cexists":
                    for c in range(C):
                        if w == 0.0:
                            specs.append(VarSpec(("dead",), _const_choices(0.0, 0.0)))
                            continue
                        js = ranges.join_spec(c)
                        if js is None:
                            _, q_hi = vn.sigmoid(*vn.scale(*ranges.full[c], _CRISP_SCALE))
                            hull_hi = max(0.0, vn.add_up(vn.mul_up(q_hi, q_hi), -_EXISTS_MARGIN))
                            specs.append(VarSpec(("join", c, ()), _const_choices(0.0, hull_hi * w, "?")))
                        else:
                            ch, active = js
                            specs.append(VarSpec(("join", c, active), scaled(ch, w)))
                else:  # cexistsp / cexistsp_r: ordered channel pairs (c1,c2)
                    rng = C if op == "cexistsp" else min(model.CEXISTSP_R_BLOCK, C)
                    for c1 in range(rng):
                        for c2 in range(rng):
                            if w == 0.0:
                                specs.append(VarSpec(("dead",), _const_choices(0.0, 0.0)))
                                continue
                            js2 = ranges.join_spec2(c1, c2)
                            if js2 is None:
                                _, q1_hi = vn.sigmoid(*vn.scale(*ranges.full[c1], _CRISP_SCALE))
                                _, q2_hi = vn.sigmoid(*vn.scale(*ranges.full[c2], _CRISP_SCALE))
                                hull_hi = max(0.0, vn.add_up(vn.mul_up(q1_hi, q2_hi), -_EXISTS_MARGIN))
                                specs.append(VarSpec(("join2", c1, c2, (), ()),
                                                     _const_choices(0.0, hull_hi * w, "?")))
                            else:
                                ch, act1, act2 = js2
                                specs.append(VarSpec(("join2", c1, c2, act1, act2),
                                                     scaled(ch, w)))
            d2 = model.d_binary
            for r in range(d2):
                specs.append(
                    VarSpec(("dead",), _const_choices(0.0, 0.0))
                    if rel_gate[r] == 0.0
                    else VarSpec(("rel", r), _bool_choices(0.0, 0.0, 1.0, 1.0))
                )
            if model.bidirectional:
                for r in range(d2):
                    specs.append(
                        VarSpec(("dead",), _const_choices(0.0, 0.0))
                        if rel_gate[r] == 0.0
                        else VarSpec(
                            ("rel_t", r), _bool_choices(0.0, 0.0, 1.0, 1.0)
                        )
                    )
        else:
            gate_w = self._gate_weights("aggregation_gate_logits", model.aggregation_ops)
            _, rel_gate = model.input_channel_gate_weights(torch.device("cpu"))
            rel_gate = [float(v) for v in rel_gate.detach().cpu()]
            for c in range(C):
                specs.append(VarSpec(("vstate", c), _state_choices(ranges, c)))
            d2 = model.d_binary
            dirs = [False, True] if model.bidirectional else [False]
            for r in range(d2):
                for incoming in dirs:
                    for oi, op in enumerate(model.aggregation_ops):
                        w = gate_w[oi]
                        for c in range(C):
                            if w == 0.0 or rel_gate[r] == 0.0:
                                specs.append(VarSpec(("dead",), _const_choices(0.0, 0.0)))
                                continue
                            _, q_full_hi = vn.sigmoid(*vn.scale(*ranges.full[c], _CRISP_SCALE))
                            if op == "cexists":
                                es = ranges.exists_spec(c)
                                if es is None:
                                    hull_hi = max(0.0, vn.add_up(q_full_hi, -_EXISTS_MARGIN))
                                    specs.append(VarSpec(("exists", r, incoming, c, ()),
                                                         _const_choices(0.0, hull_hi * w, "?")))
                                else:
                                    ch, active = es
                                    specs.append(VarSpec(("exists", r, incoming, c, active),
                                                         scaled(ch, w)))
                            elif op == "cforall":
                                fs = ranges.forall_spec(c)
                                if fs is None:
                                    specs.append(VarSpec(("forall", r, incoming, c, ()),
                                                         _const_choices(0.0, 1.0 * w, "?")))
                                else:
                                    ch, nonviol = fs
                                    specs.append(VarSpec(("forall", r, incoming, c, nonviol),
                                                         scaled(ch, w)))
                            elif op == "satcount":
                                cs = ranges.count_spec(c)
                                if cs is None:
                                    specs.append(VarSpec(("count", r, incoming, c, 0, ()),
                                                         _const_choices(0.0, _COUNT_CAP * w, "?")))
                                else:
                                    ch, cap_m, active = cs
                                    specs.append(VarSpec(("count", r, incoming, c, cap_m, active),
                                                         scaled(ch, w)))
                            elif op.startswith("cge"):
                                kk = int(op[3:])
                                gs = ranges.cge_spec(c)
                                if gs is None:
                                    specs.append(VarSpec(("cge", r, incoming, c, kk, ()),
                                                         _const_choices(0.0, 1.0 * w, "?")))
                                else:
                                    ch, active = gs
                                    specs.append(VarSpec(("cge", r, incoming, c, kk, active),
                                                         scaled(ch, w)))
                            else:
                                raise AssertionError(f"unsupported aggregation op {op}")
            for c in range(C):
                # g_max: exactly representable when the channel is two-clustered
                # and sign-split (the max is then "any node in the high cluster");
                # otherwise degrade to an opaque hull
                cl = ranges.clusters[c]
                if ranges.ok[c] and len(cl) == 2 and cl[0][1] < 0 < cl[1][0]:
                    specs.append(VarSpec(("gmax", c),
                                         [Choice(False, *cl[0]), Choice(True, *cl[1])]))
                elif ranges.ok[c] and len(cl) == 1:
                    specs.append(VarSpec(("gmax_const", c), [Choice(0, *cl[0])]))
                else:
                    specs.append(VarSpec(("gmax", c), _const_choices(*ranges.full[c], "?")))
        assert len(specs) == self.d_in, (len(specs), self.d_in)
        return specs

    def _gate_weights(self, attr: str, ops) -> list[float]:
        with torch.no_grad():
            w = self.model._primitive_gate_weights(attr, ops, torch.device("cpu"))
        vals = [float(v) for v in w.detach().cpu()]
        if any(v not in (0.0, 1.0) for v in vals):
            raise ValueError(f"primitive gate weight not hard: {vals}")
        return vals

    # ---------------- cone + enumeration ----------------
    def cone_inputs(self, out_channel: int) -> list[int]:
        """Input nodes reaching output node within circuit_ticks steps."""
        adj = self.W != 0.0
        target = self.out_start + out_channel
        frontier = {target}
        reachable_inputs: set[int] = set()
        for _ in range(self.circuit_ticks):
            nxt: set[int] = set()
            for node in frontier:
                srcs = np.flatnonzero(adj[node])
                for s in srcs:
                    if s < self.d_in:
                        reachable_inputs.add(int(s))
                    else:
                        nxt.add(int(s))
            frontier = nxt | frontier  # deeper ticks can still read earlier layers
        return sorted(reachable_inputs)

    def propagate(self, in_lo: np.ndarray, in_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Outward enclosure of the exact-real unrolled block. in_*: [N,d_in]."""
        N = in_lo.shape[0]
        lo = np.zeros((N, self.n_nodes))
        hi = np.zeros((N, self.n_nodes))
        lo[:, : self.d_in] = in_lo
        hi[:, : self.d_in] = in_hi
        hs = self.hidden_start
        for _ in range(self.circuit_ticks):
            pre_lo, pre_hi = vn.affine(lo, hi, self.W, self.bias)
            new_lo = pre_lo.copy()
            new_hi = pre_hi.copy()
            new_lo[:, hs:], new_hi[:, hs:] = vn.sigmoid(pre_lo[:, hs:], pre_hi[:, hs:])
            lo = np.concatenate([in_lo, new_lo[:, self.d_in:]], axis=1)
            hi = np.concatenate([in_hi, new_hi[:, self.d_in:]], axis=1)
            lo = lo * self.node_gate[None, :]
            hi = hi * self.node_gate[None, :]
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)  # gate is >=0, but be safe
        return lo[:, self.out_start : self.out_end], hi[:, self.out_start : self.out_end]

    def _enumerate_intervals(self, var_nodes: list[int], specs: list[VarSpec],
                             idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Input interval matrices [N, d_in] for combo indices `idx` (mixed radix,
        first var fastest; None = all combos); other inputs pinned to their hull."""
        sizes = [specs[i].n for i in var_nodes]
        if idx is None:
            idx = np.arange(math.prod(sizes) if sizes else 1)
        base_lo = np.array([min(ch.lo for ch in s.choices) for s in specs])
        base_hi = np.array([max(ch.hi for ch in s.choices) for s in specs])
        in_lo = np.tile(base_lo, (idx.size, 1))
        in_hi = np.tile(base_hi, (idx.size, 1))
        rem = idx.copy()
        for vi, node in enumerate(var_nodes):
            k = sizes[vi]
            ci = rem % k
            rem = rem // k
            los = np.array([specs[node].choices[j].lo for j in range(k)])
            his = np.array([specs[node].choices[j].hi for j in range(k)])
            in_lo[:, node] = los[ci]
            in_hi[:, node] = his[ci]
        return in_lo, in_hi

    def _squash(self, out_lo: np.ndarray, out_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.step_state:
            return (np.where(out_lo >= 0, 1.0, -1.0), np.where(out_hi >= 0, 1.0, -1.0))
        return vn.tanh(out_lo, out_hi)

    @staticmethod
    def cluster_assign(s_lo: np.ndarray, s_hi: np.ndarray) -> tuple[bool, list[tuple[float, float]], np.ndarray, Any]:
        """Cluster the per-combo value intervals; assign each combo its cluster id.

        Returns (ok, clusters, table_int, note). Deduplication preserves the
        original endpoints exactly; hull merging never shrinks an interval.
        """
        if (not np.isfinite(s_lo).all() or not np.isfinite(s_hi).all()
                or np.any(s_lo > s_hi) or not s_lo.size):
            return False, [], np.zeros(0, dtype=np.int16), {"reason": "invalid interval"}
        pairs = np.stack([s_lo, s_hi], axis=1)
        uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
        intervals = [(float(a), float(b)) for a, b in uniq]
        clusters = merge_clusters(intervals, cap=MAX_VALUE_CLUSTERS)
        if len(clusters) > MAX_VALUE_CLUSTERS:
            return False, clusters, np.zeros(0, dtype=np.int16), {"reason": "too many value clusters"}
        clo = np.array([c[0] for c in clusters])
        chi = np.array([c[1] for c in clusters])
        u_lo = uniq[:, 0]
        u_hi = uniq[:, 1]
        idx = np.searchsorted(clo, u_lo, side="right") - 1
        idx = np.clip(idx, 0, len(clusters) - 1)
        ok = bool(np.all(u_lo >= clo[idx]) and np.all(u_hi <= chi[idx]))
        if not ok:
            return False, clusters, np.zeros(0, dtype=np.int16), {"reason": "interval spans clusters"}
        table = idx[inverse].astype(np.int16)
        return True, clusters, table, None

    def extract_tick_tables(self, specs: list[VarSpec]) -> dict[int, dict[str, Any]]:
        """For each output channel: variables + truth table (or failure info).

        Exact path: enumerate the full input cone jointly (interval propagation
        through the unrolled block preserves correlations per combo).
        Compositional fallback (cone too large): abstract each hidden source as
        a free Boolean variable defined by its own truth table over its input
        fan-in. Sound (a superset of joint behaviours); may fail classification
        where the exact path would succeed.
        """
        out: dict[int, dict[str, Any]] = {}
        chunk = 1 << 15
        for c in range(self.C):
            cone = self.cone_inputs(c)
            sizes = [specs[i].n for i in cone]
            total = math.prod(sizes) if sizes else 1
            if total <= MAX_COMBOS:
                s_lo = np.zeros(total)
                s_hi = np.zeros(total)
                for start in range(0, total, chunk):
                    idx = np.arange(start, min(start + chunk, total))
                    in_lo, in_hi = self._enumerate_intervals(cone, specs, idx)
                    out_lo, out_hi = self.propagate(in_lo, in_hi)
                    a, b = self._squash(out_lo[:, c], out_hi[:, c])
                    s_lo[idx] = a
                    s_hi[idx] = b
                ok, clusters, table, note = self.cluster_assign(s_lo, s_hi)
                full = (float(s_lo.min()), float(s_hi.max()))
                if not ok:
                    out[c] = {"ok": False, "reason": note.get("reason"), "cone": cone,
                              "mode": "exact", "obs_full": full}
                else:
                    out[c] = {"ok": True, "mode": "exact",
                              "vars": [("input", i) for i in cone],
                              "hidden_defs": {}, "table": table,
                              "clusters": clusters, "obs_full": full}
                continue
            out[c] = self._extract_channel_compositional(c, specs)
        return out

    def _extract_channel_compositional(self, c: int, specs: list[VarSpec]) -> dict[str, Any]:
        if self.circuit_ticks != 2:
            return {"ok": False, "reason": "compositional path requires circuit_ticks=2"}
        alive = self.W != 0.0
        out_node = self.out_start + c
        srcs = np.flatnonzero(alive[out_node])
        direct_inputs = [int(s) for s in srcs if s < self.d_in]
        nonin_srcs = [int(s) for s in srcs if s >= self.d_in]

        # Abstract every tick-1 non-input source as a free Boolean: sigmoid
        # hiddens split at 0.5 (pre-act sign), linear tick-1 output nodes split
        # at 0. Each carries its own truth table over its input fan-in.
        hidden_defs: dict[int, dict[str, Any]] = {}
        for h in nonin_srcs:
            h_inputs = [int(s) for s in np.flatnonzero(alive[h]) if s < self.d_in]
            # non-input sources of h contribute 0 at tick 1 (values start at 0)
            sizes = [specs[i].n for i in h_inputs]
            total = math.prod(sizes) if sizes else 1
            if total > MAX_COMBOS:
                return {"ok": False, "reason": f"tick-1 node {h} fan-in too large: {total} combos"}
            in_lo, in_hi = self._enumerate_intervals(h_inputs, specs)
            pre_lo, pre_hi = vn.affine(in_lo, in_hi, self.W[h, :self.d_in], self.bias[h])
            if h >= self.hidden_start:
                v_lo, v_hi = vn.scale(*vn.sigmoid(pre_lo, pre_hi), self.node_gate[h])
                split = 0.5 * self.node_gate[h]
            else:
                v_lo, v_hi = vn.scale(pre_lo, pre_hi, self.node_gate[h])
                split = 0.0
            is_true = v_lo > split
            is_false = v_hi < split
            bad = ~(is_true | is_false)
            if bad.any():
                b = int(np.flatnonzero(bad)[0])
                return {"ok": False, "reason": f"tick-1 node {h} not two-valued",
                        "detail": {"combo_index": b, "interval": [float(v_lo[b]), float(v_hi[b])]}}
            t_int = ((float(v_lo[is_true].min()), float(v_hi[is_true].max()))
                     if is_true.any() else (float(split), float(split)))
            f_int = ((float(v_lo[is_false].min()), float(v_hi[is_false].max()))
                     if is_false.any() else (float(split), float(split)))
            hidden_defs[h] = {"sources": h_inputs, "table": is_true,
                              "f_interval": f_int, "t_interval": t_int}

        # enumerate output over direct inputs + abstracted hidden Booleans
        var_list: list[tuple[str, int]] = [("input", i) for i in direct_inputs]
        var_list += [("hidden", h) for h in nonin_srcs]
        d_sizes = [specs[i].n for i in direct_inputs]
        n_direct = math.prod(d_sizes) if d_sizes else 1
        total = n_direct * (2 ** len(nonin_srcs))
        if total > MAX_COMBOS:
            return {"ok": False, "reason": f"compositional cone still too large: {total} combos"}
        s_lo = np.zeros(total)
        s_hi = np.zeros(total)
        chunk = 1 << 15
        for start in range(0, total, chunk):
            idx = np.arange(start, min(start + chunk, total))
            d_idx = idx % max(1, n_direct)
            h_bits = idx // max(1, n_direct)
            in_lo, in_hi = self._enumerate_intervals(direct_inputs, specs, d_idx)
            v_lo = np.zeros((idx.size, self.n_nodes))
            v_hi = np.zeros((idx.size, self.n_nodes))
            v_lo[:, : self.d_in] = in_lo
            v_hi[:, : self.d_in] = in_hi
            for hi_i, h in enumerate(nonin_srcs):
                bit = (h_bits >> hi_i) & 1
                f_int = hidden_defs[h]["f_interval"]
                t_int = hidden_defs[h]["t_interval"]
                v_lo[:, h] = np.where(bit == 1, t_int[0], f_int[0])
                v_hi[:, h] = np.where(bit == 1, t_int[1], f_int[1])
            pre_lo, pre_hi = vn.affine(v_lo, v_hi, self.W[out_node], self.bias[out_node])
            a, b = self._squash(pre_lo, pre_hi)
            s_lo[idx] = a
            s_hi[idx] = b
        ok, clusters, table, note = self.cluster_assign(s_lo, s_hi)
        full = (float(s_lo.min()), float(s_hi.max()))
        if not ok:
            return {"ok": False, "reason": f"{note.get('reason')} (compositional)",
                    "mode": "compositional", "obs_full": full}
        # NOTE the enumeration order is direct-inputs-first (mixed radix, d_idx
        # fastest), i.e. var_list order with hidden bits as higher digits.
        return {"ok": True, "mode": "compositional", "vars": var_list,
                "hidden_defs": hidden_defs, "table": table,
                "clusters": clusters, "obs_full": full}

    def extract_head_joint(self, specs: list[VarSpec]) -> dict[str, Any]:
        """Evaluate sign(head(tanh(core(x)))) jointly over the UNION cone of all
        output channels at the final tick.

        Per-channel cluster enumeration loses cross-channel correlations (a
        channel may take an off-nominal value only in situations where other
        channels dominate the head); composing the head with the final tick
        keeps them exactly.
        """
        model = self.model
        with torch.no_grad():
            hw = model.head.weight.detach().cpu().double().numpy().reshape(-1)
            hb = float(model.head.bias.detach().cpu().double())
        # Only channels the head actually reads matter: a zero-weight channel
        # contributes nothing to the logit, so its cone can be dropped. Shrinking
        # the union cone directly reduces the interval-arithmetic slack.
        cone: set[int] = set()
        for c in range(self.C):
            if hw[c] != 0.0:
                cone |= set(self.cone_inputs(c))
        cone = sorted(cone)
        sizes = [specs[i].n for i in cone]
        total = math.prod(sizes) if sizes else 1
        if total > MAX_HEAD_COMBOS:
            return {"ok": False, "reason": f"head-joint cone too large: {total} combos"}
        table = np.zeros(total, dtype=bool)
        chunk = 1 << 15
        for start in range(0, total, chunk):
            idx = np.arange(start, min(start + chunk, total))
            in_lo, in_hi = self._enumerate_intervals(cone, specs, idx)
            out_lo, out_hi = self.propagate(in_lo, in_hi)
            if self.step_state:
                s_lo = np.where(out_lo >= 0, 1.0, -1.0)
                s_hi = np.where(out_hi >= 0, 1.0, -1.0)
                if (s_lo != s_hi).any():
                    # a state bit is undetermined for some combo: the head cannot
                    # be certified through it; report the first such combo
                    rows = np.flatnonzero((s_lo != s_hi).any(axis=1))
                    b = int(rows[0])
                    return {"ok": False, "reason": "head-joint: undetermined state bit",
                            "detail": {"combo": int(idx[b])}}
            else:
                s_lo, s_hi = vn.tanh(out_lo, out_hi)
            logit_lo, logit_hi = vn.affine(s_lo, s_hi, hw, hb)
            dec_t = logit_lo >= 0
            dec_f = logit_hi < 0
            bad = ~(dec_t | dec_f)
            if bad.any():
                b = int(np.flatnonzero(bad)[0])
                return {"ok": False, "reason": "head-joint not decisively signed",
                        "detail": {"combo": int(idx[b]),
                                   "interval": [float(logit_lo[b]), float(logit_hi[b])]}}
            table[idx] = dec_t
        return {"ok": True, "mode": "joint", "cone": cone, "table": table}

    def extract_head_cluster(self, ranges: "ChannelRanges") -> dict[str, Any]:
        """Readout from the FINAL state's value clusters directly (no cone
        propagation). Sound when every final channel is two-valued/cluster-split:
        enumerate the per-channel cluster combos, decide the head sign on each
        cluster interval. Loses cross-channel correlation (some combos may be
        unreachable) but that only makes it MORE conservative, never unsound --
        an unreachable combo where the head is decisive is still fine, and one
        where it is not fails safely. Avoids the cone-union interval blow-up
        that defeats head-joint on wide circuits."""
        model = self.model
        with torch.no_grad():
            hw = model.head.weight.detach().cpu().double().numpy().reshape(-1)
            hb = float(model.head.bias.detach().cpu().double())
        C = self.C
        choices = [_state_choices(ranges, c) for c in range(C)]
        sizes = [len(ch) for ch in choices]
        total = math.prod(sizes)
        if total > (1 << 20):
            return {"ok": False, "reason": f"head-cluster combos too large: {total}"}
        idx = np.arange(total)
        lo = np.full(total, hb)
        hi = np.full(total, hb)
        rem = idx.copy()
        for c in range(C):
            k = sizes[c]
            ci = rem % k
            rem = rem // k
            los = np.array([ch.lo for ch in choices[c]])
            his = np.array([ch.hi for ch in choices[c]])
            a, b = vn.scale(los[ci], his[ci], hw[c])
            lo = vn.add_down(lo, a)
            hi = vn.add_up(hi, b)
        dec_t = lo >= 0
        dec_f = hi < 0
        bad = ~(dec_t | dec_f)
        if bad.any():
            b_i = int(np.flatnonzero(bad)[0])
            return {"ok": False, "reason": "head-cluster not decisively signed",
                    "detail": {"combo": b_i, "interval": [float(lo[b_i]), float(hi[b_i])]}}
        return {"ok": True, "mode": "cluster", "choices": choices, "table": dec_t}



# ============================================================================
# Init-state extraction (exact enumeration of the embedding)
# ============================================================================


def extract_init_tables(model) -> dict[str, Any]:
    """Exact init state per Boolean input combo -> per-channel table + hulls."""
    C = model.channels
    with torch.no_grad():
        Wg = model.global_embed.weight.detach().cpu().double().numpy()
        bg = model.global_embed.bias.detach().cpu().double().numpy()
        if model.target_arity_value == 2:
            Ws = model.source_node_embed.weight.detach().cpu().double().numpy()
            bs = model.source_node_embed.bias.detach().cpu().double().numpy()
            Wt = model.target_node_embed.weight.detach().cpu().double().numpy()
            bt = model.target_node_embed.bias.detach().cpu().double().numpy()
            We = model.edge_embed.weight.detach().cpu().double().numpy()
            be = model.edge_embed.bias.detach().cpu().double().numpy()
        else:
            Wn_ = model.node_embed.weight.detach().cpu().double().numpy()
            bn_ = model.node_embed.bias.detach().cpu().double().numpy()
        unary_gate, binary_gate = model.input_channel_gate_weights(torch.device("cpu"))
        if not all(bool(torch.isin(g, torch.tensor([0., 1.])).all())
                   for g in (unary_gate, binary_gate)):
            raise ValueError("input gates must be binary")
        unary_gate_np = unary_gate.detach().cpu().double().numpy()[None, :]
        if model.target_arity_value == 2:
            Ws = Ws * unary_gate_np
            Wt = Wt * unary_gate_np
            We = We * binary_gate.detach().cpu().double().numpy()[None, :]
        else:
            Wn_ = Wn_ * unary_gate_np
    step_state = model.state_update == "step"

    def squash(lo, hi):
        return vn.step(lo, hi) if step_state else vn.tanh(lo, hi)

    d1 = model.d_unary
    if model.target_arity_value == 2:
        d2 = model.d_binary
        n_bits = d2 + 2 * d1
        total = 2 ** n_bits
        vals_lo, vals_hi = np.zeros((total, C)), np.zeros((total, C))
        for row in range(total):
            a = np.array([(row >> i) & 1 for i in range(d2)], dtype=float)
            ui = np.array([(row >> (d2 + i)) & 1 for i in range(d1)], dtype=float)
            uj = np.array([(row >> (d2 + d1 + i)) & 1 for i in range(d1)], dtype=float)
            lo, hi = vn.affine(a, a, We, be)
            sl, sh = vn.affine(ui, ui, Ws, bs)
            tl, th = vn.affine(uj, uj, Wt, bt)
            lo = vn.add_down(vn.add_down(lo, sl), tl)
            hi = vn.add_up(vn.add_up(hi, sh), th)
            vals_lo[row], vals_hi[row] = squash(lo, hi)
    else:
        n_bits = d1
        total = 2 ** n_bits
        vals_lo, vals_hi = np.zeros((total, C)), np.zeros((total, C))
        glob_lo, glob_hi = vn.affine(np.zeros(Wg.shape[1]), np.zeros(Wg.shape[1]), Wg, bg)
        for row in range(total):
            u = np.array([(row >> i) & 1 for i in range(d1)], dtype=float)
            lo, hi = vn.affine(u, u, Wn_, bn_)
            vals_lo[row], vals_hi[row] = squash(vn.add_down(lo, glob_lo), vn.add_up(hi, glob_hi))
    tables: list[np.ndarray | None] = []
    statuses = []
    details = []
    clusters_per_channel = []
    for c in range(C):
        lo, hi = vals_lo[:, c], vals_hi[:, c]
        ok, clusters, table, note = CircuitReader.cluster_assign(lo, hi)
        statuses.append(ok)
        tables.append(table if ok else None)
        clusters_per_channel.append(clusters)
        details.append({"n_clusters": len(clusters), "ok": ok,
                        "full": (float(lo.min()), float(hi.max()))})
    return {"n_bits": n_bits, "tables": tables, "statuses": statuses,
            "details": details, "clusters": clusters_per_channel}


# ============================================================================
# Table -> formula composition
# ============================================================================


class NotBooleanFunction(Exception):
    """The table distinguishes same-label clusters -> not a Boolean function."""


def _collapse_duplicate_labels(cone_specs: list[VarSpec], atoms: list[Any],
                               table: np.ndarray) -> tuple[list[VarSpec], list[Any], np.ndarray]:
    """Collapse choices sharing a label (value clusters of the same Boolean).

    Requires the table to be INVARIANT across same-label choices of each
    variable -- otherwise the circuit's behaviour depends on which cluster the
    value came from, i.e. it is not a Boolean function of that input.
    """
    specs = list(cone_specs)
    ats = list(atoms)
    tbl = table
    vi = 0
    while vi < len(specs):
        spec = specs[vi]
        labels = [ch.label for ch in spec.choices]
        uniq = list(dict.fromkeys(labels))
        if len(uniq) == len(labels):
            vi += 1
            continue
        sizes = [s.n for s in specs]
        groups: dict[Any, list[int]] = {}
        for ci, lab in enumerate(labels):
            groups.setdefault(lab, []).append(ci)
        rep_rows: dict[Any, np.ndarray] = {}
        for lab, cis in groups.items():
            base = tbl[_slice_indices(sizes, vi, cis[0])]
            for ci in cis[1:]:
                if not np.array_equal(base, tbl[_slice_indices(sizes, vi, ci)]):
                    raise NotBooleanFunction(f"var {spec.kind} label {lab}")
            rep_rows[lab] = base
        new_choices = [
            Choice(lab,
                   min(spec.choices[ci].lo for ci in groups[lab]),
                   max(spec.choices[ci].hi for ci in groups[lab]))
            for lab in uniq
        ]
        new_sizes = sizes.copy()
        new_sizes[vi] = len(uniq)
        new_tbl = np.zeros(math.prod(new_sizes), dtype=bool)
        for li, lab in enumerate(uniq):
            new_tbl[_slice_indices(new_sizes, vi, li)] = rep_rows[lab]
        specs[vi] = VarSpec(spec.kind, new_choices)
        tbl = new_tbl
        vi += 1
    return specs, ats, tbl


def _mixed_table_to_expr(cone_specs: list[VarSpec], atoms: list[Any], table: np.ndarray) -> ir.Expr:
    """Table over mixed vars -> Expr. Boolean vars use table_to_expr; count vars
    are grouped into NCount membership sets; '?'-vars must be irrelevant."""
    cone_specs, atoms, table = _collapse_duplicate_labels(cone_specs, atoms, table)
    sizes = [s.n for s in cone_specs]
    total = math.prod(sizes) if sizes else 1
    assert table.shape == (total,)

    # peel the first non-Boolean var recursively
    for vi, spec in enumerate(cone_specs):
        labels = [ch.label for ch in spec.choices]
        if spec.kind == ("dead",) or labels == ["?"] or spec.n == 1:
            continue
        if labels == [False, True] and not isinstance(atoms[vi], dict):
            continue
        k = spec.n
        rest_sizes = sizes[:vi] + sizes[vi + 1:]
        rest_specs = cone_specs[:vi] + cone_specs[vi + 1:]
        rest_atoms = atoms[:vi] + atoms[vi + 1:]
        rest_total = math.prod(rest_sizes) if rest_sizes else 1
        if isinstance(atoms[vi], dict):
            # multi-valued state var: expr = OR_k (value predicate_k AND rest_k)
            indicators: dict[Any, ir.Expr] = atoms[vi]
            parts = []
            for ci, lab in enumerate(labels):
                sub = table[_slice_indices(sizes, vi, ci)]
                if not sub.any():
                    continue
                cond = _mixed_table_to_expr(rest_specs, rest_atoms, sub)
                ind = indicators.get(lab, ir.FALSE)
                parts.append(ir.simplify(ir.And(ind, cond)))
            return ir.simplify(ir.Or(*parts) if len(parts) != 1 else parts[0]) if parts else ir.FALSE
        # count var: group rows by which m-values are true
        view = np.zeros((rest_total, k), dtype=bool)
        for m_i in range(k):
            sel = _slice_indices(sizes, vi, m_i)
            view[:, m_i] = table[sel]
        groups: dict[tuple, np.ndarray] = {}
        for r in range(rest_total):
            key = tuple(np.flatnonzero(view[r]).tolist())
            groups.setdefault(key, np.zeros(rest_total, dtype=bool))[r] = True
        parts = []
        count_atom_builder = atoms[vi]  # callable: allowed -> Expr
        for key, mask in groups.items():
            if not key:
                continue
            cond = _mixed_table_to_expr(rest_specs, rest_atoms, mask)
            catom = count_atom_builder(frozenset(key))
            parts.append(ir.simplify(ir.And(cond, catom)))
        return ir.simplify(ir.Or(*parts) if len(parts) != 1 else parts[0]) if parts else ir.FALSE

    # all vars Boolean / constant: check '?'/const vars are irrelevant, then boolean table
    bool_idx = [i for i, s in enumerate(cone_specs) if [c.label for c in s.choices] == [False, True]]
    const_idx = [i for i, s in enumerate(cone_specs) if s.n == 1]
    assert len(bool_idx) + len(const_idx) == len(cone_specs)
    # reduce constants (their single choice is fixed)
    bool_total = 2 ** len(bool_idx)
    red = np.zeros(bool_total, dtype=bool)
    for row in range(bool_total):
        full = 0
        mult = 1
        bi = 0
        for i, s in enumerate(cone_specs):
            if i in const_idx:
                ci = 0
            else:
                ci = (row >> bi) & 1
                bi += 1
            full += ci * mult
            mult *= s.n
        red[row] = table[full]
    return ir.table_to_expr([atoms[i] for i in bool_idx], red)


def _slice_indices(sizes: list[int], vi: int, choice: int) -> np.ndarray:
    """Flat indices (mixed radix, LSB-first) where var vi == choice, ordered by
    the remaining vars' mixed radix value."""
    total = math.prod(sizes)
    idx = np.arange(total)
    digits = []
    rem = idx.copy()
    for k in sizes:
        digits.append(rem % k)
        rem //= k
    mask = digits[vi] == choice
    sel = idx[mask]
    # order by remaining digits LSB-first
    order_key = np.zeros(sel.size, dtype=np.int64)
    mult = 1
    for i, k in enumerate(sizes):
        if i == vi:
            continue
        order_key += digits[i][mask] * mult
        mult *= k
    return sel[np.argsort(order_key, kind="stable")]


def swap_pair(expr: ir.Expr) -> ir.Expr:
    """F(x,y) -> F(y,x)."""
    if isinstance(expr, ir.Const):
        return expr
    if isinstance(expr, ir.Rel):
        return ir.Rel(expr.name, not expr.swapped)
    if isinstance(expr, ir.Un):
        return ir.Un(expr.name, 1 - expr.arg)
    if isinstance(expr, ir.Eq):
        return expr
    if isinstance(expr, ir.Not):
        return ir.Not(swap_pair(expr.term))
    if isinstance(expr, ir.And):
        return ir.And(*[swap_pair(t) for t in expr.terms])
    if isinstance(expr, ir.Or):
        return ir.Or(*[swap_pair(t) for t in expr.terms])
    if isinstance(expr, ir.Join):
        return ir.Join(swap_pair(expr.right), swap_pair(expr.left))
    raise TypeError(f"cannot swap: {expr!r}")


def _or_clusters(state_c: dict[int, ir.Expr] | None, ids: tuple[int, ...]) -> ir.Expr | None:
    """OR of the value predicates for the given cluster ids."""
    if state_c is None:
        return None
    terms = [state_c[k] for k in ids if k in state_c]
    if not terms:
        return ir.FALSE
    return ir.simplify(ir.Or(*terms) if len(terms) > 1 else terms[0])


def compose_program(model, init: dict[str, Any],
                    per_tick_specs: list[list[VarSpec]],
                    per_tick_tables: list[dict[int, dict[str, Any]]],
                    head: dict[str, Any],
                    rel_names: list[str], un_names: list[str],
                    ranges_t: "ChannelRanges | None" = None) -> dict[str, Any]:
    """Compose init/tick/head tables into a closed formula (per-tick unrolling).

    Channel state at each tick is a MULTI-VALUED variable: `cur[c]` maps each
    value-cluster id to its VALUE PREDICATE (an Expr); the head, analysed
    jointly with the final tick, turns everything back into one Boolean formula.
    """
    C = model.channels
    arity = model.target_arity_value

    # ---- init formulas ----
    if arity == 2:
        d2, d1 = model.d_binary, model.d_unary
        init_atoms = [ir.Rel(rel_names[r]) for r in range(d2)]
        init_atoms += [ir.Un(un_names[i], 0) for i in range(d1)]
        init_atoms += [ir.Un(un_names[i], 1) for i in range(d1)]
    else:
        d1 = model.d_unary
        init_atoms = [ir.UnSelf(un_names[i]) for i in range(d1)]
    cur: list[dict[int, ir.Expr] | None] = []
    for c in range(C):
        if init["statuses"][c]:
            spec_list = [VarSpec(("b",), _bool_choices(0, 0, 1, 1)) for _ in init_atoms]
            table = init["tables"][c]
            d: dict[int, ir.Expr] = {}
            for k in range(len(init["clusters"][c])):
                d[k] = _mixed_table_to_expr(spec_list, list(init_atoms),
                                            np.asarray(table) == k)
            cur.append(d)
        else:
            cur.append(None)

    # ---- tick composition ----
    def var_atom(kind: tuple, state: list[dict[int, ir.Expr] | None]):
        tag = kind[0]
        if tag == "vstate":
            return state[kind[1]]                                # dict: cluster -> P
        if tag == "vstate_t":
            s = state[kind[1]]
            return None if s is None else {k: swap_pair(e) for k, e in s.items()}
        if tag == "join":
            c, active = kind[1], kind[2]
            inner = _or_clusters(state[c], active)
            return None if inner is None else ir.Join(inner, inner)
        if tag == "join2":
            c1, c2, act1, act2 = kind[1], kind[2], kind[3], kind[4]
            left = _or_clusters(state[c1], act1)
            right = _or_clusters(state[c2], act2)
            return None if left is None or right is None else ir.Join(left, right)
        if tag == "rel":
            return ir.Rel(rel_names[kind[1]])
        if tag == "rel_t":
            return ir.Rel(rel_names[kind[1]], swapped=True)
        if tag == "exists":
            r, incoming, c, active = kind[1], kind[2], kind[3], kind[4]
            inner = _or_clusters(state[c], active)
            return None if inner is None else ir.NExists(rel_names[r], incoming, inner)
        if tag == "forall":
            r, incoming, c, nonviol = kind[1], kind[2], kind[3], kind[4]
            inner = _or_clusters(state[c], nonviol)
            return None if inner is None else ir.NForall(rel_names[r], incoming, inner)
        if tag == "count":
            r, incoming, c, cap, active = kind[1], kind[2], kind[3], kind[4], kind[5]
            inner = _or_clusters(state[c], active)
            if inner is None or cap <= 0:
                return None
            return lambda allowed: ir.NCount(rel_names[r], incoming, inner, allowed, cap)
        if tag == "cge":
            # binary "count >= k": a plain Boolean atom (min(count, cap) in {k..cap})
            r, incoming, c, kk, active = kind[1], kind[2], kind[3], kind[4], kind[5]
            inner = _or_clusters(state[c], active)
            if inner is None:
                return None
            cap = int(_COUNT_CAP)
            return ir.NCount(rel_names[r], incoming, inner,
                             frozenset(range(kk, cap + 1)), cap)
        if tag == "gmax":
            s = state[kind[1]]
            if s is None or 1 not in s:
                return None
            return ir.GExists(s[1])                              # "some node in the high cluster"
        if tag in ("dead", "gmax_const"):
            return ir.FALSE  # constant; never used as a live boolean
        raise AssertionError(kind)

    def gather_atoms(var_ids, specs, state):
        """(var_specs, atoms) for _mixed_table_to_expr; None if a needed formula is missing."""
        var_specs: list[VarSpec] = []
        atoms: list[Any] = []
        for i in var_ids:
            a = var_atom(specs[i].kind, state)
            if a is None and specs[i].n > 1:
                return None
            var_specs.append(specs[i])
            atoms.append(a if a is not None else ir.FALSE)
        return var_specs, atoms

    def channel_exprs(tt: dict[str, Any], specs: list[VarSpec],
                      state: list[dict[int, ir.Expr] | None]) -> dict[int, ir.Expr] | None:
        var_specs: list[VarSpec] = []
        atoms: list[Any] = []
        for tag, ref in tt["vars"]:
            if tag == "input":
                a = var_atom(specs[ref].kind, state)
                if a is None and specs[ref].n > 1:
                    return None
                var_specs.append(specs[ref])
                atoms.append(a if a is not None else ir.FALSE)
            else:  # abstracted tick-1 node Boolean: its formula over ITS input sources
                hdef = tt["hidden_defs"][ref]
                got = gather_atoms(hdef["sources"], specs, state)
                if got is None:
                    return None
                h_specs, h_atoms = got
                h_expr = _mixed_table_to_expr(h_specs, h_atoms, hdef["table"])
                var_specs.append(VarSpec(("b",), _bool_choices(0, 0, 1, 1)))
                atoms.append(h_expr)
        table = np.asarray(tt["table"])
        return {k: _mixed_table_to_expr(var_specs, atoms, table == k)
                for k in range(len(tt["clusters"]))}

    if not head.get("ok") or head.get("mode") not in ("joint", "cluster"):
        return {"ok": False, "reason": head.get("reason", "head unavailable"), "trace": []}

    # For head-joint we compose all ticks then the head reads the FINAL tick's
    # core inputs; for head-cluster the head reads the FINAL STATE clusters, so
    # we must compose ALL num_ticks ticks (the last one included).
    n_state_ticks = (model.num_ticks - 1) if head["mode"] == "joint" else model.num_ticks
    trace = [list(cur)]
    for tick in range(n_state_ticks):
        tts = per_tick_tables[tick]
        specs = per_tick_specs[tick]
        nxt: list[dict[int, ir.Expr] | None] = []
        for c in range(C):
            tt = tts[c]
            if not tt.get("ok"):
                nxt.append(None)
                continue
            try:
                nxt.append(channel_exprs(tt, specs, cur))
            except NotBooleanFunction:
                nxt.append(None)
        cur = nxt
        trace.append(list(cur))

    if head["mode"] == "cluster":
        # Head over the FINAL STATE clusters, restricted to REACHABLE cluster
        # combinations. A combination (k_0,...,k_{C-1}) is reachable iff the
        # conjunction of its per-channel cluster predicates cur[c][k_c] is
        # satisfiable; unsatisfiable ones cannot occur so the head need not be
        # decided on them (this is what beats the cross-channel-correlation
        # failure of a naive cluster product). On a reachable combination the
        # head interval must be decisively signed, else extraction fails.
        assert ranges_t is not None
        with torch.no_grad():
            hw = model.head.weight.detach().cpu().double().numpy().reshape(-1)
            hb = float(model.head.bias.detach().cpu().double())
        var_channels = [c for c in range(C)
                        if len(ranges_t.clusters[c]) > 1 and hw[c] != 0.0]
        for c in var_channels:
            if cur[c] is None:
                return {"ok": False, "reason": f"final channel {c} has no cluster formula",
                        "trace": trace}
        # fixed (single-cluster or head-weight-0) channels contribute a constant
        # interval to the logit
        base_lo, base_hi = hb, hb
        for c in range(C):
            if c in var_channels:
                continue
            k_lo = min(lo for lo, _ in ranges_t.clusters[c])
            k_hi = max(hi for _, hi in ranges_t.clusters[c])
            a, b = vn.scale(k_lo, k_hi, hw[c])
            base_lo = vn.add_down(base_lo, a)
            base_hi = vn.add_up(base_hi, b)
        # Order channels by fewest clusters first so the reachability conjunction
        # collapses to FALSE (pruning a whole subtree) as early as possible.
        var_channels.sort(key=lambda c: len(ranges_t.clusters[c]))
        # Per-channel head-contribution intervals, and the running min/max the
        # UNVISITED tail can still add -- used to decide the head sign before the
        # whole combination is fixed (branch-and-bound), so a k-hop circuit's
        # astronomically many cluster products never materialise.
        contrib = []
        for c in var_channels:
            ivs = [vn.scale(lo, hi, hw[c])
                   for lo, hi in ranges_t.clusters[c]]
            contrib.append(ivs)
        m = len(var_channels)
        tail_lo = [0.0] * (m + 1)
        tail_hi = [0.0] * (m + 1)
        for i in range(m - 1, -1, -1):
            tail_lo[i] = vn.add_down(tail_lo[i + 1], min(v[0] for v in contrib[i]))
            tail_hi[i] = vn.add_up(tail_hi[i + 1], max(v[1] for v in contrib[i]))

        parts_true: list[ir.Expr] = []
        undecided: list[Any] = []
        NODE_CAP = 5_000_000
        nodes = [0]

        def dfs(i: int, cond: ir.Expr, lo: float, hi: float) -> None:
            if undecided or nodes[0] > NODE_CAP:
                return
            nodes[0] += 1
            # branch-and-bound on the head sign: if even the extreme remaining
            # contributions cannot flip it, the whole subtree shares one verdict.
            if vn.add_down(lo, tail_lo[i]) >= 0:
                parts_true.append(cond)
                return
            if vn.add_up(hi, tail_hi[i]) < 0:
                return
            if i == m:
                undecided.append((float(lo), float(hi)))   # reachable, head unsure
                return
            c = var_channels[i]
            for k in range(len(ranges_t.clusters[c])):
                nc = ir.simplify(ir.And(cond, cur[c][k]))
                if isinstance(nc, ir.Const) and not nc.value:
                    continue                                # unreachable subtree
                dfs(i + 1, nc, vn.add_down(lo, contrib[i][k][0]), vn.add_up(hi, contrib[i][k][1]))
                if undecided or nodes[0] > NODE_CAP:
                    return

        dfs(0, ir.TRUE, base_lo, base_hi)
        if nodes[0] > NODE_CAP:
            return {"ok": False, "reason": f"head-cluster search exceeded {NODE_CAP} nodes",
                    "trace": trace}
        if undecided:
            return {"ok": False, "reason": "head-cluster reachable combo not decisively signed",
                    "trace": trace, "detail": {"interval": undecided[0]}}
        final = ir.simplify(ir.Or(*parts_true)) if parts_true else ir.FALSE
        return {"ok": True, "formula": final, "trace": trace}

    # ---- head-joint: head composed with the FINAL tick's core inputs ----
    specs = per_tick_specs[-1]
    got = gather_atoms(head["cone"], specs, cur)
    if got is None:
        return {"ok": False, "reason": "head-joint input var has no formula", "trace": trace}
    var_specs, atoms = got
    try:
        final = _mixed_table_to_expr(var_specs, atoms, np.asarray(head["table"]))
    except NotBooleanFunction as exc:
        return {"ok": False, "reason": f"head-joint distinguishes same-label clusters: {exc}",
                "trace": trace}
    return {"ok": True, "formula": ir.simplify(final), "trace": trace}


# ============================================================================
# Target logic definitions (generator vocabulary; see np4g_program_ir docstring)
# ============================================================================
# Family targets are expressed over the generator's FREE parameters
# (father, mother, male); son/daughter channels appearing in extracted
# formulas are substituted by their definitions before comparison.


def _parent() -> ir.Expr:
    return ir.Or(ir.Rel("father"), ir.Rel("mother"))


def _parent_sw() -> ir.Expr:
    return ir.Or(ir.Rel("father", True), ir.Rel("mother", True))


def _brother_pair() -> ir.Expr:
    # brother(x,y): shared parent p, y male, x != y
    return ir.And(ir.Join(_parent(), _parent_sw()), ir.Un("male", 1), ir.Not(ir.Eq()))


def _sister_pair() -> ir.Expr:
    # sister(x,y) (generator _sister): EXISTS m mother(x,m) AND daughter(m,y), x != y
    # daughter(m,y) = parent(y,m) AND NOT male(y)
    return ir.And(ir.Join(ir.Rel("mother"), _parent_sw()),
                  ir.Not(ir.Un("male", 1)), ir.Not(ir.Eq()))


def _uncle_pair() -> ir.Expr:
    return ir.Join(_parent(), _brother_pair())


def _connectivity(k: int) -> ir.Expr:
    cur: ir.Expr = ir.Rel("adj")
    for _ in range(k - 1):
        cur = ir.Or(ir.Rel("adj"), ir.Join(ir.Rel("adj"), cur))
    return cur


def _two_hop() -> ir.Expr:
    return ir.Join(ir.Rel("adj"), ir.Rel("adj"))


def _binary_adf(expr: ir.Expr) -> ir.Expr:
    """Apply the public binary-ADF contract: predictions are zero on x == y."""

    return ir.And(expr, ir.Not(ir.Eq()))


TARGETS: dict[str, dict[str, Any]] = {
    "has_father": {"arity": 1, "expr": ir.NExists("father", False, ir.TRUE)},
    "is_sister": {"arity": 2, "expr": _sister_pair()},
    "is_parent": {"arity": 2, "expr": _parent()},
    "is_father": {"arity": 1, "expr": ir.NExists("father", True, ir.TRUE)},
    "is_mother": {"arity": 1, "expr": ir.NExists("mother", True, ir.TRUE)},
    "co_parent": {
        "arity": 2,
        "expr": ir.Join(_parent_sw(), _parent()),
    },
    "has_sister": {"arity": 1, "expr": ir.NExists(_sister_pair(), False, ir.TRUE)},
    "is_brother": {"arity": 2, "expr": _brother_pair()},
    "is_grandparent": {"arity": 2, "expr": ir.Join(_parent(), _parent())},
    "is_uncle": {"arity": 2, "expr": _uncle_pair()},
    "is_mguncle": {"arity": 2, "expr": ir.Join(ir.Rel("mother"), _uncle_pair())},
    "adjacent_to_red": {"arity": 1, "expr": ir.NExists("adj", False, ir.UnSelf("red"))},
    "connectivity_2": {"arity": 2, "expr": _connectivity(2)},
    "connectivity_4": {"arity": 2, "expr": _connectivity(4)},
    "connectivity_6": {"arity": 2, "expr": _connectivity(6)},
    "outdegree_1": {"arity": 1, "count_profile": ("adj", False, 1)},
    "outdegree_2": {"arity": 1, "count_profile": ("adj", False, 2)},
    # dILP-suite tasks (predecessor / undirected edge / coloured graph / husband
    # analogues; see docs/paper_guide.md and exp_torch_nlm_np4g_google_graph).
    "inverse_edge": {"arity": 2, "expr": ir.Rel("adj", True)},
    "undirected_edge": {"arity": 2, "expr": ir.Or(ir.Rel("adj"), ir.Rel("adj", True))},
    "graph_colouring": {"arity": 1, "expr": ir.Or(
        ir.And(ir.UnSelf("red"), ir.NExists("adj", False, ir.UnSelf("red"))),
        ir.And(ir.Not(ir.UnSelf("red")),
               ir.NExists("adj", False, ir.Not(ir.UnSelf("red")))),
    )},
    "husband": {"arity": 2, "expr": ir.Join(ir.Rel("father", True), ir.Rel("mother"))},
    "two_hop": {"arity": 2, "expr": _two_hop()},
    "four_hop_via_adf": {
        "arity": 2,
        # The learned two_hop predicate is consumed through augment_batch,
        # which clears every binary ADF diagonal.  Those inner inequalities
        # affect off-diagonal four-hop answers and therefore belong in the
        # exact target, not only in the final output mask.
        "expr": ir.Join(_binary_adf(_two_hop()), _binary_adf(_two_hop())),
    },
}


def subst_names(expr: ir.Expr, pair_map: dict[str, ir.Expr], node_map: dict[str, ir.Expr]) -> ir.Expr:
    """Replace named atoms (ADF channels, class-constant channels) by formulas."""

    def sub(e: ir.Expr) -> ir.Expr:
        if isinstance(e, ir.Const) or isinstance(e, ir.Eq):
            return e
        if isinstance(e, ir.Rel):
            if e.name in pair_map:
                f = pair_map[e.name]
                return swap_pair(f) if e.swapped else f
            return e
        if isinstance(e, ir.Un):
            if e.name in node_map:
                return ir.NodeAt(node_map[e.name], e.arg)
            return e
        if isinstance(e, ir.UnSelf):
            if e.name in node_map:
                return node_map[e.name]
            return e
        if isinstance(e, ir.NodeAt):
            return ir.NodeAt(sub(e.inner), e.arg)
        if isinstance(e, ir.Not):
            return ir.Not(sub(e.term))
        if isinstance(e, ir.And):
            return ir.And(*[sub(t) for t in e.terms])
        if isinstance(e, ir.Or):
            return ir.Or(*[sub(t) for t in e.terms])
        if isinstance(e, ir.Join):
            return ir.Join(sub(e.left), sub(e.right))
        if isinstance(e, ir.NExists):
            rel = e.rel
            if isinstance(rel, str) and rel in pair_map:
                rel = pair_map[rel]
            elif not isinstance(rel, str):
                rel = sub(rel)
            return ir.NExists(rel, e.incoming, sub(e.inner))
        if isinstance(e, ir.NForall):
            rel = e.rel
            if isinstance(rel, str) and rel in pair_map:
                rel = pair_map[rel]
            elif not isinstance(rel, str):
                rel = sub(rel)
            return ir.NForall(rel, e.incoming, sub(e.inner))
        if isinstance(e, ir.NCount):
            rel = e.rel
            if isinstance(rel, str) and rel in pair_map:
                rel = pair_map[rel]
            elif not isinstance(rel, str):
                rel = sub(rel)
            return ir.NCount(rel, e.incoming, sub(e.inner), e.allowed, e.cap)
        if isinstance(e, ir.GExists):
            return ir.GExists(sub(e.inner))
        raise TypeError(f"cannot substitute: {e!r}")

    return ir.simplify(sub(expr))


def substitute_family_derived(expr: ir.Expr) -> ir.Expr:
    """Replace son/daughter atoms by their (father, mother, male) definitions."""
    son = ir.And(_parent_sw(), ir.Un("male", 1))       # son(x,y): y child of x AND male(y)
    daughter = ir.And(_parent_sw(), ir.Not(ir.Un("male", 1)))

    def sub_pair(e: ir.Expr) -> ir.Expr:
        if isinstance(e, ir.Rel):
            if e.name == "son":
                return swap_pair(son) if e.swapped else son
            if e.name == "daughter":
                return swap_pair(daughter) if e.swapped else daughter
            return e
        if isinstance(e, ir.Const) or isinstance(e, (ir.Un, ir.Eq, ir.UnSelf)):
            return e
        if isinstance(e, ir.NodeAt):
            return ir.NodeAt(sub_pair(e.inner), e.arg)
        if isinstance(e, ir.Not):
            return ir.Not(sub_pair(e.term))
        if isinstance(e, ir.And):
            return ir.And(*[sub_pair(t) for t in e.terms])
        if isinstance(e, ir.Or):
            return ir.Or(*[sub_pair(t) for t in e.terms])
        if isinstance(e, ir.Join):
            return ir.Join(sub_pair(e.left), sub_pair(e.right))
        if isinstance(e, ir.NExists):
            rel = e.rel if isinstance(e.rel, str) else sub_pair(e.rel)
            if rel == "son":
                rel = son
            elif rel == "daughter":
                rel = daughter
            return ir.NExists(rel, e.incoming, sub_pair(e.inner))
        if isinstance(e, ir.NForall):
            rel = e.rel if isinstance(e.rel, str) else sub_pair(e.rel)
            if rel in ("son", "daughter"):
                rel = son if rel == "son" else daughter
            return ir.NForall(rel, e.incoming, sub_pair(e.inner))
        if isinstance(e, ir.NCount):
            rel = e.rel if isinstance(e.rel, str) else sub_pair(e.rel)
            if rel in ("son", "daughter"):
                rel = son if rel == "son" else daughter
            return ir.NCount(rel, e.incoming, sub_pair(e.inner), e.allowed, e.cap)
        if isinstance(e, ir.GExists):
            return ir.GExists(sub_pair(e.inner))
        raise TypeError(f"cannot substitute: {e!r}")

    return ir.simplify(sub_pair(expr))


def substitute_graph_eye(expr: ir.Expr) -> ir.Expr:
    """Replace the eye channel by the equality atom (its exact semantics)."""
    return subst_names(expr, pair_map={"eye": ir.Eq()}, node_map={})


# ============================================================================
# Sanity check: extracted formula must reproduce the model exactly on samples
# ============================================================================


@dataclass
class _Batch:
    input0: torch.Tensor
    input1: torch.Tensor
    input2: torch.Tensor
    input3: torch.Tensor
    target: torch.Tensor
    pair_mask: torch.Tensor


def _structure_to_batch(domain: str, rels: dict[str, np.ndarray], uns: dict[str, np.ndarray]) -> _Batch:
    n = next(iter(rels.values())).shape[0]
    if domain == "family":
        input2 = np.stack([rels["father"], rels["mother"], rels["son"], rels["daughter"]], axis=-1)
        input1 = np.zeros((n, 1), dtype=np.float32)
    else:
        input2 = np.stack([rels["adj"], rels["eye"]], axis=-1)
        input1 = uns["red"].astype(np.float32)[:, None]
    return _Batch(
        input0=torch.zeros(1, 1),
        input1=torch.as_tensor(input1, dtype=torch.float32).unsqueeze(0),
        input2=torch.as_tensor(input2.astype(np.float32)).unsqueeze(0),
        input3=torch.zeros(1, 1, 1, 1, 1),
        target=torch.zeros(1),
        pair_mask=torch.ones(1),
    )


def sanity_check_formula(task: str, domain: str, arity: int, expr_raw: ir.Expr,
                         model, upstream_entries, *, sizes=(6, 9, 13, 20), samples=12,
                         seed: int = 0) -> dict[str, Any]:
    """Formula (raw vocabulary) vs model predictions; must match on every cell."""
    rng = np.random.default_rng(seed)
    checked = 0
    for n in sizes:
        for _ in range(samples):
            if domain == "family":
                rels, uns = ir.random_family_structure(rng, n)
            else:
                rels, uns = ir.random_graph_structure(rng, n)
            batch = _structure_to_batch(domain, rels, uns)
            batch = augment_batch(batch, upstream_entries)
            model.set_num_objects(n)
            with torch.no_grad():
                logits = model(batch)[0].cpu().numpy()
            # ADF reuse, held-out evaluation, and training metrics all define
            # a non-negative logit as true.  The extraction sanity check must
            # use that same boundary convention, including the exact-zero case.
            pred = logits >= 0.0
            uns_eval = dict(uns)
            vals = (ir.eval_pair if arity == 2 else ir.eval_node)(expr_raw, rels, uns_eval)
            if arity == 2:
                mask = ~np.eye(n, dtype=bool)
                bad = (pred != vals) & mask
            else:
                bad = pred != vals
            if bad.any():
                idx = np.argwhere(bad)[0]
                return {"ok": False, "n": n, "cell": idx.tolist(),
                        "model": bool(pred[tuple(idx)]), "formula": bool(vals[tuple(idx)]),
                        "checked_cells": checked}
            checked += int(bad.size)
    return {"ok": True, "checked_cells": checked}


def falsify_formula_vs_truth(task: str, arity: int, expr_class: ir.Expr,
                             *, sizes=(50, 100, 200, 500), samples=6, seed: int = 1) -> dict[str, Any]:
    """Counterexample search of the VERIFIED formula against generator ground truth.

    With the formula in hand this is pure numpy, so large n is cheap. Any
    mismatch falsifies the equivalence verdict (should never happen if the
    engine is sound) -- kept as an independent battery.
    """
    rng = np.random.default_rng(seed)
    wrong_total = 0
    cells_total = 0
    per_size: dict[str, Any] = {}
    for n in sizes:
        wrong = 0
        cells = 0
        for _ in range(samples):
            if task in (
                "has_father",
                "is_sister",
                "is_parent",
                "is_father",
                "is_mother",
                "co_parent",
                "has_sister",
                "is_brother",
                "is_grandparent",
                "is_uncle",
                "is_mguncle",
                "husband",
            ):
                rels, uns = ir.random_family_structure(rng, n)
                truth = _family_truth(task, rels, uns)
                ev_rels, ev_uns = rels, uns
            else:
                rels, uns = ir.random_graph_structure(rng, n)
                truth = _graph_truth(task, rels, uns)
                ev_rels, ev_uns = rels, uns
            vals = (ir.eval_pair if arity == 2 else ir.eval_node)(expr_class, ev_rels, ev_uns)
            if arity == 2:
                mask = ~np.eye(n, dtype=bool)
                wrong += int(((vals != truth) & mask).sum())
                cells += int(mask.sum())
            else:
                wrong += int((vals != truth).sum())
                cells += n
        per_size[f"n{n}"] = {"wrong": wrong, "cells": cells}
        wrong_total += wrong
        cells_total += cells
    return {"wrong": wrong_total, "cells": cells_total, "per_size": per_size}


def _bool_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Boolean relation composition without narrow-integer overflow."""

    # Formal falsification includes structures with as many as 500 objects.
    # int8 accumulation can wrap once more than 127 witnesses exist and turn a
    # genuinely reachable pair into false.  float32 exactly represents every
    # possible witness count at these sizes and delegates to NumPy's fast
    # matrix multiplication implementation.
    return left.astype(np.float32) @ right.astype(np.float32) > 0.0


def _family_truth(task: str, rels: dict[str, np.ndarray], uns: dict[str, np.ndarray]) -> np.ndarray:
    father, mother = rels["father"], rels["mother"]
    son, daughter = rels["son"], rels["daughter"]
    parent = father | mother
    n = father.shape[0]
    eye = np.eye(n, dtype=bool)
    sister = _bool_matmul(mother, daughter) & ~eye
    child_of_parent = (son | daughter).T
    brother = _bool_matmul(child_of_parent, son) & ~eye
    uncle = _bool_matmul(parent, brother)
    if task == "has_father":
        return father.any(axis=1)
    if task == "is_sister":
        return sister
    if task == "is_parent":
        return parent
    if task == "is_father":
        return father.any(axis=0)
    if task == "is_mother":
        return mother.any(axis=0)
    if task == "co_parent":
        return _bool_matmul(parent.T, parent)
    if task == "has_sister":
        return sister.any(axis=1)
    if task == "is_brother":
        return brother
    if task == "is_grandparent":
        return _bool_matmul(parent, parent)
    if task == "is_uncle":
        return uncle
    if task == "is_mguncle":
        return _bool_matmul(mother, uncle)
    if task == "husband":
        return _bool_matmul(father.T, mother)
    raise ValueError(task)


def _graph_truth(task: str, rels: dict[str, np.ndarray], uns: dict[str, np.ndarray]) -> np.ndarray:
    adj = rels["adj"]
    if task == "adjacent_to_red":
        return (adj & uns["red"][None, :]).any(axis=1)
    if task.startswith("connectivity_"):
        k = int(task.split("_")[1])
        reach = adj.copy()
        power = adj.copy()
        for _ in range(2, k + 1):
            power = _bool_matmul(power, adj)
            reach = reach | power
        return reach
    if task.startswith("outdegree_"):
        k = int(task.split("_")[1])
        return adj.sum(axis=1) == k
    if task == "inverse_edge":
        return adj.T.copy()
    if task == "undirected_edge":
        return adj | adj.T
    if task == "graph_colouring":
        red = uns["red"]
        same = red[:, None] == red[None, :]
        return (adj & same).any(axis=1)
    if task == "two_hop":
        return _bool_matmul(adj, adj)
    if task == "four_hop_via_adf":
        two = _bool_matmul(adj, adj)
        diagonal = np.arange(two.shape[-1])
        two[..., diagonal, diagonal] = False
        return _bool_matmul(two, two)
    raise ValueError(task)


# ============================================================================
# Target verification
# ============================================================================


def verify_against_target(task: str, arity: int, expr_class: ir.Expr) -> dict[str, Any]:
    """Exact equivalence of the class-vocabulary formula with the target logic."""
    target = TARGETS[task]
    if "count_profile" in target:
        rel, incoming, k = target["count_profile"]
        bound = ir.count_saturation_bound(expr_class)
        if bound is None:
            return {"status": "NOT_PROVED", "engine": "count",
                    "reason": "unsupported counting grammar"}
        cap = max(bound, k + 1)
        prof = node_to_profile_or_none(expr_class, cap)
        if prof is None:
            return {"status": "NOT_PROVED", "reason": "extracted formula outside counting fragment",
                    "engine": "count"}
        e_rel, e_inc, e_allowed = prof
        if e_rel != rel or e_inc != incoming:
            return {"status": "NOT_PROVED", "engine": "count",
                    "reason": f"counts over ({e_rel},{'in' if e_inc else 'out'}) instead of ({rel},{'in' if incoming else 'out'})"}
        t_allowed = np.array([c == k for c in range(len(e_allowed))], dtype=bool)
        if np.array_equal(e_allowed, t_allowed):
            return {"status": "VERIFIED_EQUIVALENT", "engine": "count",
                    "profile": e_allowed.tolist(), "saturation_bound": cap}
        return {"status": "NOT_EQUIVALENT", "engine": "count",
                "extracted_profile": e_allowed.tolist(), "target_profile": t_allowed.tolist(),
                "saturation_bound": cap,
                "counterexample_degree": int(np.flatnonzero(e_allowed != t_allowed)[0])}

    tgt = target["expr"]
    is_family = task in (
        "has_father",
        "is_sister",
        "is_parent",
        "is_father",
        "is_mother",
        "co_parent",
        "has_sister",
        "is_brother",
        "is_grandparent",
        "is_uncle",
        "is_mguncle",
        "husband",
    )
    if is_family:
        rel_names = ["father", "mother"]
        un_names = ["male"]
    else:
        rel_names = ["adj"]
        un_names = ["red"]

    def _decide(ext: ir.Expr) -> tuple[str, dict[str, Any]]:
        if arity == 2:
            neq = ir.Not(ir.Eq())
            q_ext = ir.pair_to_ucq(ir.simplify(ir.And(ext, neq)))
            q_tgt = ir.pair_to_ucq(ir.simplify(ir.And(tgt, neq)))
        else:
            q_ext = ir.node_to_ucq(ext)
            q_tgt = ir.node_to_ucq(tgt)
        q_ext = ir.minimize_ucq(q_ext, rel_names, un_names)
        q_tgt = ir.minimize_ucq(q_tgt, rel_names, un_names)
        # Family predicates are defined only over the generator's structure
        # class, so decide equivalence there (father/mother functional +
        # irreflexive + disjoint + sex); graph tasks keep all-structure UCQ.
        class_model = ir.family_class_model if is_family else ir.graph_class_model
        ok, witness = ir.ucq_equivalent(q_ext, q_tgt, rel_names, un_names,
                                        class_model=class_model)
        if ok:
            extra = {"extracted_cqs": len(q_ext), "target_cqs": len(q_tgt)}
            if is_family:
                extra["class_scoped"] = ["father/mother functional", "irreflexive",
                                         "disjoint", "father->male / mother->female", "acyclic"]
            return "VERIFIED_EQUIVALENT", extra
        return "NOT_EQUIVALENT", {"witness": witness}

    try:
        status, extra = _decide(expr_class)
        return {"status": status, "engine": "ucq", **extra}
    except ir.UCQFail as exc:
        # The raw extracted formula fell outside the decidable fragment (a
        # negated binary atom). For family tasks, retry after eliminating the
        # negated parent atoms that the generator's class constraints make
        # redundant (father/mother disjoint + parent antisymmetric). This is a
        # SOUND rewrite on the generator's structure class (paper_guide 2.4(b)),
        # and the caller independently falsifies the reduced formula against
        # ground truth as a safety net (extract_task).
        if not is_family:
            return {"status": "NOT_PROVED", "reason": f"outside UCQ fragment: {exc}", "engine": "ucq"}
        reduced = ir.family_class_axiom_reduce(expr_class)
        try:
            status, extra = _decide(reduced)
        except ir.UCQFail as exc2:
            return {"status": "NOT_PROVED", "engine": "ucq+class",
                    "reason": f"outside UCQ fragment even after class reduction: {exc2}",
                    "class_axioms": [
                        "father/mother disjoint",
                        "parent-role exclusivity",
                        "parent antisymmetric",
                    ],
                    "reduced_formula": repr(reduced),
                    "_reduced_expr_obj": reduced}
        return {"status": status, "engine": "ucq+class",
                "class_axioms": [
                    "father/mother disjoint",
                    "parent-role exclusivity",
                    "parent antisymmetric",
                ],
                "reduced_formula": repr(reduced),
                "_reduced_expr_obj": reduced, **extra}


def node_to_profile_or_none(expr: ir.Expr, cap: int):
    """Wrapper of ir.node_to_count_profile that also normalises the atom cap."""
    prof = ir.node_to_count_profile(expr, cap)
    if prof is None:
        return None
    rel, incoming, allowed = prof
    if not isinstance(rel, str):
        return None
    return rel, incoming, allowed


# ============================================================================
# Per-task extraction pipeline
# ============================================================================


def _hull2(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]))


def _contained(inner: tuple[float, float] | None, outer: tuple[float, float] | None,
               eps: float = 0.0) -> bool:
    if inner is None:
        return True
    if outer is None:
        return False
    return inner[0] >= outer[0] and inner[1] <= outer[1]


def extract_task(task: str, entry, upstream_entries, formulas: dict[str, dict[str, Any]],
                 nominal: float = 0.98, do_sanity: bool = True, *,
                 sanity_seed: int = 0, falsification_seed: int = 1) -> dict[str, Any]:
    model = entry.model
    domain = entry.domain
    arity = entry.arity
    report: dict[str, Any] = {"task": task, "domain": domain, "arity": arity,
                              "verification_contract": vn.metadata()}
    t0 = time.time()

    base_rels = FAMILY_RELS if domain == "family" else GRAPH_RELS
    base_uns = FAMILY_UNS if domain == "family" else GRAPH_UNS
    rel_names = list(base_rels) + [f"ADF:{e.task}" for e in upstream_entries if e.arity == 2]
    un_names = list(base_uns) + [f"ADF:{e.task}" for e in upstream_entries if e.arity == 1]
    assert len(rel_names) == model.d_binary, (rel_names, model.d_binary)
    assert len(un_names) == model.d_unary, (un_names, model.d_unary)

    reader = CircuitReader(model)
    init = extract_init_tables(model)
    report["init_statuses"] = init["statuses"]

    # Propagate the actual outward cluster enclosures through each tick.
    # The nominal argument remains for compatibility with historical callers;
    # no nominal margin is substituted for a proven enclosure.
    C = model.channels
    tick_notes: dict[str, Any] = {}

    ranges_t = ChannelRanges(C)
    for c in range(C):
        ranges_t.ok[c] = bool(init["statuses"][c])
        ranges_t.clusters[c] = init["clusters"][c] if ranges_t.ok[c] else [(-1.0, 1.0)]
        ranges_t.full[c] = init["details"][c]["full"]
    per_tick_specs: list[list[VarSpec]] = []
    per_tick_tables: list[dict[int, dict[str, Any]]] = []
    tick_status_trace: list[list[bool]] = [list(ranges_t.ok)]
    for t in range(model.num_ticks):
        specs = reader.input_layout(ranges_t)
        tts = reader.extract_tick_tables(specs)
        per_tick_specs.append(specs)
        per_tick_tables.append(tts)
        new = ChannelRanges(C)
        for c in range(C):
            tt = tts[c]
            if not tt.get("ok"):
                tick_notes[f"tick{t + 1}_ch{c}"] = {
                    k: v for k, v in tt.items()
                    if k not in ("table", "hidden_defs", "vars", "clusters")
                }
                new.ok[c] = False
                new.clusters[c] = [(-1.0, 1.0)]
                new.full[c] = (-1.0, 1.0)
            else:
                new.ok[c] = True
                new.clusters[c] = tt["clusters"]
                new.full[c] = tt.get("obs_full") or (-1.0, 1.0)
        ranges_t = new
        tick_status_trace.append(list(ranges_t.ok))
    report["tick_status_trace"] = tick_status_trace
    report["tick_table_notes"] = tick_notes
    report["final_clusters"] = {
        c: {"ok": ranges_t.ok[c], "clusters": [list(x) for x in ranges_t.clusters[c]]}
        for c in range(C)
    }
    statuses = list(ranges_t.ok)
    report["channel_statuses"] = statuses

    head = reader.extract_head_joint(per_tick_specs[-1])
    if not head.get("ok"):
        # head-joint blows up on wide circuits (cone-union interval slack). Fall
        # back to the head-over-final-state-clusters reading, decided per state
        # cluster and filtered to REACHABLE cluster combinations (compose_program
        # handles it using the per-channel cluster predicates).
        report["head_joint_note"] = head.get("reason")
        head = {"ok": True, "mode": "cluster"}
    program = compose_program(model, init, per_tick_specs, per_tick_tables, head,
                              rel_names, un_names, ranges_t)
    if not program.get("ok"):
        report["status"] = "EXTRACTION_FAILED"
        report["reason"] = program.get("reason")
        report["detail"] = program.get("detail")
        report["elapsed_sec"] = time.time() - t0
        return report

    formula = program["formula"]
    report["formula_extracted"] = repr(formula)

    # close over upstream ADF formulas + class-constant channels
    pair_map: dict[str, ir.Expr] = {}
    node_map: dict[str, ir.Expr] = {"_zero": ir.FALSE}
    for e in upstream_entries:
        up = formulas.get(e.task)
        upstream_report = up.get("report", {}) if up else {}
        if (up is None or up.get("expr_raw") is None
                or upstream_report.get("status") != "VERIFIED_EQUIVALENT"
                or upstream_report.get("verification_contract") != vn.metadata()):
            report["status"] = "BLOCKED_ON_UPSTREAM"
            report["reason"] = f"upstream '{e.task}' has no proof under the current numerical contract"
            report["elapsed_sec"] = time.time() - t0
            return report
        if e.arity == 2:
            # downstream sees the upstream prediction with a zeroed diagonal
            pair_map[f"ADF:{e.task}"] = ir.And(up["expr_raw"], ir.Not(ir.Eq()))
        else:
            node_map[f"ADF:{e.task}"] = up["expr_raw"]
    expr_raw = ir.boolean_minimize(subst_names(formula, pair_map, node_map))
    report["formula_raw_closed"] = repr(expr_raw)

    if do_sanity:
        sanity = sanity_check_formula(
            task,
            domain,
            arity,
            expr_raw,
            model,
            upstream_entries,
            seed=sanity_seed,
        )
        report["sanity_vs_model"] = sanity
        if not sanity["ok"]:
            report["status"] = "EXTRACTION_MISMATCH"
            report["elapsed_sec"] = time.time() - t0
            return report

    if domain == "family":
        expr_class = substitute_family_derived(expr_raw)
    else:
        expr_class = substitute_graph_eye(expr_raw)
    expr_class = ir.boolean_minimize(expr_class)
    report["formula_class"] = repr(expr_class)

    verdict = verify_against_target(task, arity, expr_class)
    # If class axioms were used to reach the fragment, independently falsify the
    # REDUCED formula against generator ground truth. A sound reduction leaves
    # the formula behaviourally identical on the class, so any nonzero mismatch
    # means the class rewrite changed meaning -> revoke the VERIFIED claim.
    reduced_expr = verdict.pop("_reduced_expr_obj", None)
    if reduced_expr is not None:
        red_fals = falsify_formula_vs_truth(
            task,
            arity,
            reduced_expr,
            seed=falsification_seed,
        )
        verdict["reduced_falsification"] = red_fals
        if verdict["status"] == "VERIFIED_EQUIVALENT" and red_fals["wrong"] != 0:
            verdict = {"status": "EXTRACTION_MISMATCH", "engine": "ucq+class",
                       "reason": "class-axiom reduction changed truth value "
                                 f"({red_fals['wrong']}/{red_fals['cells']} wrong) — unsound, "
                                 "VERIFIED revoked",
                       "reduced_falsification": red_fals}
    report["verification"] = verdict
    report["status"] = verdict["status"]

    fals = falsify_formula_vs_truth(
        task,
        arity,
        expr_class,
        seed=falsification_seed,
    )
    report["falsification_formula_vs_truth"] = fals
    if report["status"] == "VERIFIED_EQUIVALENT" and fals["wrong"]:
        report["status"] = "EXTRACTION_MISMATCH"
        report["reason"] = "formula disagrees with target on the falsification dataset"

    report["expr_raw_obj"] = expr_raw           # for downstream substitution (not serialised)
    report["elapsed_sec"] = time.time() - t0
    return report


# ============================================================================
# CLI
# ============================================================================


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=default_artifact_root(OFFICIAL_CONFIG.protocol_version))
    p.add_argument("--task", default="all")
    p.add_argument("--nominal-margin", type=float, default=0.98,
                   help="Nominal state domain: class values must stay in [-1,-m] u [m,1]. "
                        "Must match the SYN_STATE_LO the circuit was trained with.")
    p.add_argument("--no-sanity", action="store_true")
    p.add_argument("--report-out", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=None, help="Diagnostic log directory (default: RUN_DIR/logs).")
    args = p.parse_args()
    log_path = configure_logging("extract", log_dir=args.log_dir or args.run_dir / "logs")
    logger.info("Extraction started run_dir=%s task=%s log=%s", args.run_dir, args.task, log_path)

    lib = ADFLibrary(args.run_dir / "adf_library")
    tasks = lib.tasks()
    if not tasks:
        logger.error("No tasks found in ADF library=%s", lib.root)
        raise SystemExit(f"no tasks in {lib.root}")
    wanted = tasks if args.task == "all" else [args.task]

    # ---- memory guard --------------------------------------------------
    # Cone enumeration / boolean minimisation is not memory-bounded (a bad
    # circuit can eat all system RAM: connectivity_4 took >6GB, and an
    # is_grandparent seed froze a 32GB machine). A watchdog thread aborts
    # THIS process safely when its RSS exceeds the cap or system available
    # memory gets low, recording the current task as EXTRACTION_FAILED so
    # the run is resumable and never re-enters the same blowup blindly.
    current_task: list[str | None] = [None]
    task_started: list[float | None] = [None]
    reports: list[dict[str, Any]] = []
    _start_memory_guard(
        cap_gb=float(os.environ.get("NP4G_EXTRACT_MEM_GB", "6")),
        min_avail_gb=float(os.environ.get("NP4G_MIN_AVAIL_GB", "3")),
        report_out=args.report_out, reports=reports, current_task=current_task,
    )
    # Wall-clock terminator for CPU-bound enumeration hangs (default 30 min per
    # task; the memory guard does not catch these — a wide cone can spin for a
    # day at ~0.25GB RSS). Override via NP4G_EXTRACT_TIME_SEC (0 disables).
    _start_time_guard(
        per_task_sec=float(os.environ.get("NP4G_EXTRACT_TIME_SEC", "1800")),
        report_out=args.report_out, reports=reports, current_task=current_task,
        task_started=task_started,
    )

    entries_by_domain: dict[str, list] = {}
    for domain in ("family", "graph"):
        entries_by_domain[domain] = lib.load_entries(
            domain=domain,
            before_task=None,
            device="cpu",
            model_factory=lambda record, dependencies: NeuralNP4G.from_config(
                record["build_config"], dependencies
            ),
        )
    entry_by_task = {e.task: e for d in entries_by_domain.values() for e in d}

    formulas: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if task not in entry_by_task:
            continue
        entry = entry_by_task[task]
        by_name = {e.task: e for e in entries_by_domain[entry.domain]}
        upstream = [by_name[name] for name in entry.dependencies]
        if task not in wanted:
            # still extract (cheap) so downstream substitution works
            pass
        current_task[0] = task
        task_started[0] = time.time()
        logger.info("Extracting task=%s", task)
        rep = extract_task(task, entry, upstream, formulas, nominal=args.nominal_margin,
                           do_sanity=not args.no_sanity)
        formulas[task] = {"expr_raw": rep.pop("expr_raw_obj", None), "report": rep}
        status = rep.get("status")
        logger.info("Extraction task=%s status=%s", task, status)
        if "verification" in rep:
            v = rep["verification"]
            logger.info("Verification task=%s status=%s engine=%s", task, v.get("status"), v.get("engine"))
        if "sanity_vs_model" in rep:
            s = rep["sanity_vs_model"]
            diagnostic = logger.info if s.get("ok") else logger.warning
            diagnostic("Sanity check task=%s matches_model=%s cells=%s", task, s.get("ok"), s.get("checked_cells", 0))
        if "falsification_formula_vs_truth" in rep:
            f = rep["falsification_formula_vs_truth"]
            logger.info("Falsification task=%s wrong=%s cells=%s", task, f["wrong"], f["cells"])
        if "formula_raw_closed" in rep:
            print(f"  formula: {rep['formula_raw_closed'][:400]}")
        reports.append(rep)

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote extraction report=%s", args.report_out)

    print("\n==== SUMMARY ====")
    for rep in reports:
        print(f"  {rep['task']:>16}: {rep.get('status')}")
    logger.info("Extraction completed tasks=%s", len(reports))


def main() -> None:
    try:
        _main()
    except Exception:
        logger.exception("Extraction command failed")
        raise


if __name__ == "__main__":
    main()
