"""
roofline.py — Full roofline model with memory hierarchy simulation.

Goes beyond a single bandwidth check.
Models: L1 → L2 → L3 → HBM → DRAM memory hierarchy with
realistic bandwidth and latency at each level.

Key output: which memory level is actually the binding constraint,
and what is the achievable throughput fraction of peak.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.schemas import DataMovementSimResult

# Memory hierarchy defaults (AI accelerator, circa 2024-2025)
MEMORY_HIERARCHY = {
    "L1_cache": {"bw_gb_s": 20000,  "latency_ns": 0.5,   "capacity_mb": 0.064},
    "L2_cache": {"bw_gb_s": 5000,   "latency_ns": 2.0,   "capacity_mb": 4.0},
    "L3_cache": {"bw_gb_s": 2000,   "latency_ns": 10.0,  "capacity_mb": 96.0},
    "HBM3":     {"bw_gb_s": 1200,   "latency_ns": 30.0,  "capacity_mb": 96*1024},
    "DRAM":     {"bw_gb_s": 100,    "latency_ns": 70.0,  "capacity_mb": 1e6},
}

# Signal integrity: bandwidth degrades with distance (rough model)
# Based on PCIe5 retimer data and NVLink4 specs
def _signal_bw_limit(bw_gb_s: float, distance_mm: float) -> float:
    if distance_mm <= 50:
        return bw_gb_s  # on-package: no degradation
    elif distance_mm <= 200:
        attn = 1.0 - (distance_mm - 50) / 1000  # 0.1% per mm beyond 50mm
        return bw_gb_s * max(attn, 0.7)
    else:
        attn = 0.7 * (200 / distance_mm) ** 0.5  # sqrt rolloff at long distance
        return bw_gb_s * attn


def run_data_movement_sim(
    dm_params: Optional[dict] = None,
    memory_hierarchy: Optional[dict] = None,
) -> DataMovementSimResult:
    result = DataMovementSimResult()

    # Extract params
    bw_claimed   = (dm_params or {}).get("bandwidth_gb_s") or 1200.0
    compute_tf   = (dm_params or {}).get("compute_tflops") or 100.0
    latency_ns   = (dm_params or {}).get("latency_ns") or None
    distance_mm  = None  # not in current schema but could be added

    hierarchy = memory_hierarchy or MEMORY_HIERARCHY

    # ── Arithmetic intensity ──────────────────────────────────────────────────
    # AI = FLOP / byte — the fundamental axis of the roofline
    ai = (compute_tf * 1e12) / (bw_claimed * 1e9)  # FLOP/byte
    result.arithmetic_intensity_flop_per_byte = round(ai, 2)

    # Ridge point: where compute-bound meets memory-bound
    ridge_point = compute_tf * 1e12 / (bw_claimed * 1e9)
    result.ridge_point_flop_per_byte = round(ridge_point, 2)

    # ── Per-level analysis ────────────────────────────────────────────────────
    bw_util: Dict[str, float] = {}
    binding_level = None
    binding_bw    = float("inf")

    for level, props in hierarchy.items():
        level_bw   = props["bw_gb_s"]
        achievable = min(compute_tf, level_bw * ai / 1e3)
        utilization = achievable / compute_tf * 100
        bw_util[level] = round(utilization, 1)

    # Binding level = the level whose BW is >= bw_claimed AND closest to it
    # (i.e., the narrowest bottleneck the data actually flows through)
    # If bw_claimed is within a level range, that level is the bottleneck.
    binding_level = None
    best_diff = float("inf")
    for level, props in hierarchy.items():
        diff = abs(props["bw_gb_s"] - bw_claimed)
        if props["bw_gb_s"] <= bw_claimed * 1.5 and diff < best_diff:
            best_diff = diff
            binding_level = level

    result.bandwidth_utilization = bw_util
    result.binding_memory_level  = binding_level or list(hierarchy.keys())[0]
    result.peak_compute_tflops   = round(compute_tf, 2)

    # Achievable throughput = min(compute_roof, memory_roof at claimed BW)
    achievable_tf = min(compute_tf, bw_claimed * ai / 1e3)

    result.achievable_tflops = round(achievable_tf, 2)
    result.efficiency_pct    = round((achievable_tf / compute_tf) * 100, 1)

    # Compute vs memory bound classification
    if ai > ridge_point:
        result.bottleneck = "compute"
    elif ai > ridge_point * 0.1:
        result.bottleneck = "bandwidth"
    else:
        result.bottleneck = "memory_latency"

    # ── Bandwidth wall check ──────────────────────────────────────────────────
    # Is claimed bandwidth physically achievable?
    hbm3_4stack = 1200 * 4  # 4× HBM3 stacks = 4.8 TB/s max
    bw_feasible = bw_claimed <= hbm3_4stack

    # ── Latency model ─────────────────────────────────────────────────────────
    if latency_ns:
        result.effective_latency_ns = latency_ns
        # Latency-bound: if AI < 1 FLOP/byte, latency dominates
        lat_bound_pct = max(0, (1 - ai) * 100) if ai < 1 else 0
        result.latency_bound_pct = round(min(lat_bound_pct, 100), 1)
    else:
        # Estimate effective latency from hierarchy
        if binding_level:
            result.effective_latency_ns = hierarchy[binding_level]["latency_ns"]

    # ── Signal integrity ──────────────────────────────────────────────────────
    result.signal_integrity_ok = bw_feasible
    # Practical max distance for claimed bandwidth
    if bw_claimed > 128:  # beyond PCIe5
        result.max_practical_distance_mm = 50.0  # on-package only
    elif bw_claimed > 50:
        result.max_practical_distance_mm = 200.0

    # ── Status ────────────────────────────────────────────────────────────────
    notes = []
    if not bw_feasible:
        result.status = "fail"
        notes.append(f"Bandwidth {bw_claimed} GB/s exceeds 4×HBM3 limit ({hbm3_4stack} GB/s) — not achievable")
    elif result.efficiency_pct < 20:
        result.status = "critical"
        notes.append(f"Only {result.efficiency_pct:.1f}% of peak compute utilized — severely memory-bound")
    elif result.efficiency_pct < 50:
        result.status = "warning"
        notes.append(f"{result.efficiency_pct:.1f}% efficiency — binding: {binding_level}")
    else:
        result.status = "pass"
        notes.append(f"{result.efficiency_pct:.1f}% efficiency, {result.bottleneck}-bound (AI={ai:.1f} FLOP/byte)")

    notes.append(f"Arithmetic intensity: {ai:.1f} FLOP/byte (ridge point: {ridge_point:.1f})")
    notes.append(f"Binding memory level: {binding_level} ({binding_bw:.0f} GB/s)")
    if result.effective_latency_ns:
        notes.append(f"Effective latency: {result.effective_latency_ns} ns")

    result.notes = notes
    return result
