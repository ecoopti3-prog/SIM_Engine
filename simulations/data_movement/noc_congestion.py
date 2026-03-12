"""
noc_congestion.py — Network-on-Chip Congestion Model

The roofline model assumes bandwidth is always available.
Reality: data physically travels through a mesh NoC inside the chip.
At high utilization, router queues saturate → latency explodes (Little's Law).

This module adds congestion-aware throughput degradation on top of the roofline.

Physics:
  M/M/1 queueing model:
    λ = arrival rate [requests/ns]
    μ = service rate [requests/ns] = link_bw / packet_size
    ρ = λ/μ = utilization (0-1)
    W = 1/(μ - λ) = mean waiting time    [diverges as ρ → 1]
    L = λW = mean queue length (Little's Law)

  When ρ > 0.7: latency grows super-linearly
  When ρ > 0.9: effective bandwidth collapses to <50% of peak

For 2D mesh NoC (most common in AI chips):
  - Each router has 5 ports (N/S/E/W + local)
  - Bisection bandwidth = sqrt(N_routers) × link_bw × 2
  - Hotspot traffic pattern (all-to-one) is 3× worse than uniform

Reference: published NoC congestion analysis from:
  - NVIDIA A100 whitepaper (NVLink latency vs load)
  - MIT 6.004 (M/D/1 vs M/M/1 queueing)
  - Balfour & Dally 2006 "Design Tradeoffs for Tiled CMP On-chip Networks"
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ── NoC topology parameters ───────────────────────────────────────────────────
NOC_CONFIGS = {
    "mesh_2d_8x8": {
        "n_routers":        64,
        "n_rows":           8,
        "n_cols":           8,
        "link_bw_gb_s":     128,      # per-link bandwidth [GB/s]
        "link_latency_ns":  1.5,      # per-hop latency [ns]
        "max_hops":         14,       # max Manhattan distance on 8×8
        "packet_size_b":    64,       # cache line size [bytes]
    },
    "mesh_2d_16x16": {
        "n_routers":        256,
        "n_rows":           16,
        "n_cols":           16,
        "link_bw_gb_s":     128,
        "link_latency_ns":  1.5,
        "max_hops":         30,
        "packet_size_b":    64,
    },
    "ring": {
        "n_routers":        16,
        "link_bw_gb_s":     256,
        "link_latency_ns":  1.0,
        "max_hops":         8,
        "packet_size_b":    64,
    },
}

TRAFFIC_PATTERNS = {
    "uniform_random": 1.0,       # baseline
    "nearest_neighbor": 0.5,     # mostly local → low congestion
    "hotspot_all_to_one": 3.2,   # all nodes → one memory controller → severe
    "scatter_gather": 2.1,       # reduction ops (all-reduce in training)
    "attention_pattern": 1.8,    # attention: q broadcasts to k,v
}


def _mm1_queueing(rho: float) -> Tuple[float, float, float]:
    """
    M/M/1 queue analysis.
    rho: server utilization (0-1)
    Returns: (mean_wait, mean_queue_length, effective_bw_fraction)
    """
    if rho >= 1.0:
        return float("inf"), float("inf"), 0.0

    mean_wait   = rho / (1.0 - rho)          # normalized to service time
    mean_queue  = rho**2 / (1.0 - rho)       # mean number in queue (Little's Law)

    # Effective bandwidth fraction = throughput / peak
    # As queue builds, effective throughput = λ_effective / μ
    # Under congestion, sources throttle back: effective BW degrades
    if rho < 0.5:
        bw_fraction = 1.0
    elif rho < 0.7:
        bw_fraction = 1.0 - 0.1 * (rho - 0.5) / 0.2
    elif rho < 0.9:
        bw_fraction = 0.9 - 0.3 * (rho - 0.7) / 0.2
    else:
        bw_fraction = 0.6 - 0.5 * (rho - 0.9) / 0.1

    bw_fraction = max(0.05, bw_fraction)
    return round(mean_wait, 3), round(mean_queue, 3), round(bw_fraction, 3)


def run_noc_analysis(
    dm_params: Optional[dict] = None,
    n_compute_units: int = 128,       # number of SM/core clusters
    traffic_pattern: str = "attention_pattern",
    noc_topology: str = "mesh_2d_16x16",
    n_memory_controllers: int = 8,
) -> Dict:
    """
    Analyze NoC congestion and compute effective bandwidth degradation.

    Injects into the roofline model as a bandwidth penalty.
    """
    dm = dm_params or {}
    bw_claimed_gb_s = float(dm.get("bandwidth_gb_s") or 1200.0)
    compute_tflops  = float(dm.get("compute_tflops") or 100.0)
    latency_ns      = float(dm.get("latency_ns") or 30.0)

    noc = NOC_CONFIGS.get(noc_topology, NOC_CONFIGS["mesh_2d_16x16"])
    pattern_factor = TRAFFIC_PATTERNS.get(traffic_pattern, 1.8)

    # ── Link utilization ──────────────────────────────────────────────────────
    # Bandwidth demand per router link
    # Each compute unit generates traffic; routes through avg_hops links
    avg_hops = (noc["n_rows"] + noc["n_cols"]) / 3.0   # average Manhattan distance

    # Nominal bandwidth per link that would serve the whole chip
    total_link_bw = noc["link_bw_gb_s"] * noc["n_routers"] * 4  # 4 links per router (N/S/E/W)
    # Bisection bandwidth (bottleneck for all-to-all traffic)
    bisection_bw  = noc["link_bw_gb_s"] * noc["n_rows"] * 2  # GB/s

    # Traffic load: bandwidth demand per link
    # Each GB/s of memory bandwidth requires (avg_hops) link traversals
    effective_demand = bw_claimed_gb_s * pattern_factor * avg_hops
    link_utilization = effective_demand / max(total_link_bw, 1.0)
    link_utilization = min(link_utilization, 0.999)  # physical upper bound

    # ── Queueing analysis ─────────────────────────────────────────────────────
    mean_wait, queue_len, bw_fraction = _mm1_queueing(link_utilization)

    # Effective bandwidth after congestion
    bw_effective_gb_s = bw_claimed_gb_s * bw_fraction

    # ── Latency under congestion ──────────────────────────────────────────────
    # Base latency = hop_count × link_latency
    base_latency_ns = avg_hops * noc["link_latency_ns"]

    # Congestion adds queueing delay at each hop
    queue_delay_ns  = mean_wait * noc["link_latency_ns"] * avg_hops
    total_latency_ns = latency_ns + base_latency_ns + queue_delay_ns

    # ── Memory controller contention ──────────────────────────────────────────
    # Multiple compute units competing for N memory controllers
    mc_utilization = min(0.999, (n_compute_units / n_memory_controllers) *
                          link_utilization * pattern_factor * 0.3)
    _, mc_queue, mc_bw_frac = _mm1_queueing(mc_utilization)
    mc_latency_ns = mean_wait * noc["link_latency_ns"] * 2

    # Combined effective bandwidth
    bw_final = bw_effective_gb_s * mc_bw_frac
    total_congestion_penalty_pct = round((1 - bw_final / bw_claimed_gb_s) * 100, 1)

    # ── Updated Roofline metrics ──────────────────────────────────────────────
    ai = (compute_tflops * 1e12) / (bw_final * 1e9) if bw_final > 0 else float("inf")
    achievable_tflops = min(compute_tflops, bw_final * ai / 1e3)
    efficiency_pct    = round(achievable_tflops / compute_tflops * 100, 1)

    # ── Status ────────────────────────────────────────────────────────────────
    status = "pass"
    notes  = []

    if link_utilization > 0.85:
        status = "critical"
        notes.append(f"NoC SATURATION: link utilization={link_utilization:.0%} — throughput collapses")
    elif link_utilization > 0.7:
        status = "warning"
        notes.append(f"NoC congested: utilization={link_utilization:.0%}, penalty={total_congestion_penalty_pct:.0f}%")
    else:
        notes.append(f"NoC utilization={link_utilization:.0%} — acceptable")

    notes.append(f"Effective BW: {bw_final:.0f} GB/s ({total_congestion_penalty_pct:.0f}% penalty from {bw_claimed_gb_s:.0f} claimed)")
    notes.append(f"Congestion latency: +{queue_delay_ns:.1f}ns (base {base_latency_ns:.1f}ns + queue {queue_delay_ns:.1f}ns)")
    notes.append(f"Traffic pattern: {traffic_pattern} (factor={pattern_factor}×)")

    if mc_utilization > 0.7:
        if status == "pass": status = "warning"
        notes.append(f"Memory controller contention: {mc_utilization:.0%} utilization, +{mc_latency_ns:.1f}ns")

    return {
        "link_utilization":             round(link_utilization, 3),
        "mc_utilization":               round(mc_utilization, 3),
        "bw_claimed_gb_s":              bw_claimed_gb_s,
        "bw_effective_gb_s":            round(bw_final, 1),
        "congestion_penalty_pct":       total_congestion_penalty_pct,
        "mean_queue_length":            round(queue_len, 2),
        "added_latency_ns":             round(queue_delay_ns, 1),
        "total_latency_ns":             round(total_latency_ns, 1),
        "achievable_tflops":            round(achievable_tflops, 2),
        "efficiency_with_noc_pct":      efficiency_pct,
        "traffic_pattern":              traffic_pattern,
        "noc_topology":                 noc_topology,
        "bisection_bw_gb_s":            round(bisection_bw, 0),
        "status":                       status,
        "notes":                        notes,
    }
