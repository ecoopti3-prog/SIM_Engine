"""
L1 Data Movement Simulator — Roofline + Network Flow
Models:
  Roofline: PERF = min(P_peak, BW × AI)
    AI = arithmetic_intensity (FLOPs/byte)
    If AI < ridge_point → memory bandwidth bound
    If AI > ridge_point → compute bound
  Latency: t_latency = distance / signal_speed + memory_latency
  Bandwidth wall: actual_BW < peak_BW due to overhead
Falsification: sweep AI to find roofline ridge and bandwidth wall crossing
"""
from __future__ import annotations
import math
import numpy as np
from typing import Optional, Dict, List
import sys, os
_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
from core.schemas import DomainSimResult, FalsificationBoundary, ParameterSweepPoint, SimStatus

SIGNAL_SPEED_MM_PER_NS = 200.0   # ~2/3 c in PCB, ~200 mm/ns
DRAM_LATENCY_NS        = 70.0    # typical DDR5 latency
HBM_LATENCY_NS         = 30.0    # HBM3 latency
NVLINK_LATENCY_NS      = 50.0    # NVLink4 latency
PCIe5_BW_GB_S          = 128.0   # PCIe 5.0 x16 unidirectional
HBM3_BW_GB_S           = 1200.0  # HBM3 per stack
BW_EFFICIENCY          = 0.75    # practical efficiency vs theoretical

def simulate_data_movement(
    idea_id: str,
    bandwidth_gb_s: Optional[float] = None,
    latency_ns: Optional[float] = None,
    memory_capacity_gb: Optional[float] = None,
    interconnect_speed_gb_s: Optional[float] = None,
    compute_tflops: Optional[float] = None,
    # Derived
    model_size_b: Optional[float] = None,      # model params in bytes
    connection_distance_mm: Optional[float] = None,
    arithmetic_intensity: Optional[float] = None,  # FLOPs/byte
) -> DomainSimResult:
    warnings: List[str] = []
    metrics: Dict[str, float] = {}
    sweep: List[ParameterSweepPoint] = []
    falsification: Optional[FalsificationBoundary] = None

    if not any([bandwidth_gb_s, latency_ns, interconnect_speed_gb_s, compute_tflops]):
        return DomainSimResult(
            idea_id=idea_id, domain="data_movement", fidelity="L1_analytical",
            status="insufficient_data", score=4.0,
            warnings=["No data movement params — need bandwidth, latency, or compute_tflops"],
        )

    bw = bandwidth_gb_s or interconnect_speed_gb_s or 0.0
    bw_eff = bw * BW_EFFICIENCY

    # ── Roofline model ─────────────────────────────────────────────────────────
    if compute_tflops and bw_eff > 0:
        bw_eff_tbs = bw_eff / 1000.0  # convert GB/s to TB/s for TFLOPS
        ridge_point = compute_tflops / bw_eff_tbs  # FLOPs/byte where compute = bw bound
        metrics["ridge_point_flops_per_byte"]  = round(ridge_point, 2)
        metrics["compute_tflops"]              = compute_tflops
        metrics["effective_bandwidth_gb_s"]    = round(bw_eff, 1)
        if arithmetic_intensity:
            ai = arithmetic_intensity
            perf_achieved = min(compute_tflops, bw_eff_tbs * ai)
            hw_utilization = perf_achieved / compute_tflops * 100
            metrics["arithmetic_intensity"]       = ai
            metrics["perf_achieved_tflops"]        = round(perf_achieved, 2)
            metrics["hw_utilization_pct"]          = round(hw_utilization, 1)
            bottleneck = "memory_bandwidth" if ai < ridge_point else "compute"
            metrics["bottleneck"]                  = 1.0 if bottleneck == "compute" else 0.0

            if hw_utilization < 30:
                warnings.append(
                    f"Hardware utilization only {hw_utilization:.1f}% — "
                    f"{'memory bandwidth starved (AI={ai:.1f} < ridge={ridge_point:.1f})' if ai < ridge_point else 'compute bound, increase batch size'}"
                )
            # AI sweep for roofline curve
            for ai_s in np.logspace(-1, 3, 30):
                p_s = min(compute_tflops, bw_eff_tbs * ai_s)
                sweep.append(ParameterSweepPoint(
                    param_name="arithmetic_intensity_flops_per_byte",
                    param_value=round(float(ai_s), 2),
                    metric_name="perf_tflops",
                    metric_value=round(float(p_s), 3),
                    status="pass" if float(p_s) > 0.5 * compute_tflops else "marginal",
                ))
            falsification = FalsificationBoundary(
                domain="data_movement", breaking_param="arithmetic_intensity",
                nominal_value=round(ai, 2) if ai else ridge_point,
                breaking_value=round(ridge_point, 2),
                safety_margin_pct=round((ai - ridge_point) / ridge_point * 100 if ai > ridge_point else (ai - ridge_point) / ridge_point * 100, 1),
                units="FLOPs/byte",
                equation_used=f"PERF = min({compute_tflops} TFLOPS, {bw_eff_tbs:.3f} TB/s × AI)",
                fix_vector=(
                    f"AI={ai:.1f} is {'below' if ai < ridge_point else 'above'} ridge point {ridge_point:.1f}. "
                    + (f"Increase data reuse (tiling, caching) to push AI above {ridge_point:.1f} FLOPs/byte." if ai < ridge_point else
                       "System is compute bound. Increase HBM bandwidth or reduce precision (FP8).")
                ) if ai else "Provide arithmetic_intensity to get roofline operating point."
            )

    # ── Latency analysis ───────────────────────────────────────────────────────
    if latency_ns:
        metrics["latency_ns"] = latency_ns
        if latency_ns < HBM_LATENCY_NS * 0.5:
            warnings.append(
                f"Claimed latency {latency_ns} ns < HBM3 physics floor {HBM_LATENCY_NS} ns — verify"
            )
        elif latency_ns < DRAM_LATENCY_NS:
            metrics["latency_type_implied"] = 1.0  # 1=HBM-class
        else:
            metrics["latency_type_implied"] = 0.0  # 0=DRAM-class

    # ── Distance-based latency ─────────────────────────────────────────────────
    if connection_distance_mm:
        t_prop_ns = connection_distance_mm / SIGNAL_SPEED_MM_PER_NS
        metrics["propagation_latency_ns"] = round(t_prop_ns, 3)
        if latency_ns and t_prop_ns > latency_ns * 0.5:
            warnings.append(
                f"Propagation delay {t_prop_ns:.2f} ns = {t_prop_ns/latency_ns*100:.0f}% "
                f"of total latency — distance dominates"
            )

    # ── Bandwidth vs model size ────────────────────────────────────────────────
    if model_size_b and bw > 0:
        t_transfer_s = model_size_b / (bw * 1e9)
        metrics["model_load_time_s"] = round(t_transfer_s, 3)
        if t_transfer_s > 1.0:
            warnings.append(
                f"Model transfer time {t_transfer_s:.2f}s at {bw} GB/s — "
                "high latency inference, consider weight streaming or compression"
            )

    # ── Bandwidth headroom vs HBM3 limit ──────────────────────────────────────
    if bw > 0:
        margin_vs_hbm3 = (HBM3_BW_GB_S - bw) / HBM3_BW_GB_S * 100
        metrics["bandwidth_gb_s"]             = bw
        metrics["hbm3_limit_gb_s"]            = HBM3_BW_GB_S
        metrics["bandwidth_vs_hbm3_margin_pct"] = round(margin_vs_hbm3, 1)
        if bw > HBM3_BW_GB_S:
            warnings.append(
                f"Bandwidth claim {bw} GB/s EXCEEDS HBM3 practical limit "
                f"{HBM3_BW_GB_S} GB/s per stack — verify multi-stack or novel interconnect"
            )
        if not falsification:
            falsification = FalsificationBoundary(
                domain="data_movement", breaking_param="bandwidth_gb_s",
                nominal_value=bw, breaking_value=HBM3_BW_GB_S,
                safety_margin_pct=round(margin_vs_hbm3, 1),
                units="GB/s",
                equation_used=f"HBM3_max = {HBM3_BW_GB_S} GB/s per stack",
                fix_vector=(
                    "Multi-stack HBM or novel photonic interconnect to exceed HBM3."
                    if bw > HBM3_BW_GB_S
                    else f"{margin_vs_hbm3:.1f}% bandwidth headroom before HBM3 wall."
                ),
            )

    # ── Score ──────────────────────────────────────────────────────────────────
    fatal    = any("EXCEED" in w for w in warnings)
    marginal = any("only" in w or "starved" in w or "dominates" in w for w in warnings)
    if fatal:
        score, sf = 2.0, "fail"
    elif marginal:
        score, sf = 5.5, "marginal"
    else:
        util = metrics.get("hw_utilization_pct", 70.0)
        score = min(10.0, util / 10.0)
        sf: SimStatus = "pass"
        if score < 3.0: sf = "fail"
        elif score < 6.0: sf = "marginal"

    return DomainSimResult(
        idea_id=idea_id, domain="data_movement", fidelity="L1_analytical",
        status=sf, score=round(score, 2),
        metrics=metrics, sweep_points=sweep, falsification=falsification,
        warnings=warnings,
        solver_notes="Roofline model, latency analysis, bandwidth headroom check",
    )
