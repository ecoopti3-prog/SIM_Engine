"""
L1 PDN Simulator — RLC Network (Frequency Domain + DC)
Physics:
  DC: V_drop = I × R_pdn  (must be < 5% VDD)
  AC: Z(f) = R + j2πfL + 1/(j2πfC)  — impedance profile
  Resonance: f_res = 1/(2π√(LC))  — avoid VDD frequency harmonics here
  Electromigration: J < 1 mA/μm²  at 100°C
  Decap sufficiency: Q_decap ≥ I_peak × t_response
Falsification: sweep current to find IR_drop = 5% VDD
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

IR_DROP_LIMIT_PCT  = 5.0      # % of VDD
EM_CURRENT_DENSITY = 1.0      # mA/μm² Cu at 100°C limit
BUMP_RESISTANCE_OHM = 5e-4   # 0.5 mΩ per C4 bump (typical)

def simulate_pdn(
    idea_id: str,
    ir_drop_mv: Optional[float] = None,
    vdd_v: Optional[float] = None,
    pdn_impedance_mohm: Optional[float] = None,
    frequency_ghz: Optional[float] = None,
    bump_density_per_mm2: Optional[float] = None,
    current_a: Optional[float] = None,
    di_dt_a_per_ns: Optional[float] = None,
    decap_nf: Optional[float] = None,
    n_chiplets: Optional[int] = None,
) -> DomainSimResult:
    warnings: List[str] = []
    metrics: Dict[str, float] = {}
    sweep: List[ParameterSweepPoint] = []
    falsification: Optional[FalsificationBoundary] = None

    if not any([ir_drop_mv, vdd_v, pdn_impedance_mohm, current_a]):
        return DomainSimResult(
            idea_id=idea_id, domain="pdn", fidelity="L1_analytical",
            status="insufficient_data", score=4.0,
            warnings=["No PDN params — need ir_drop_mv, vdd_v, current_a, or impedance"],
        )

    vdd = vdd_v or 0.8   # default to aggressive 0.8V (3nm)

    # ── IR drop (DC) ───────────────────────────────────────────────────────────
    if ir_drop_mv is not None:
        ir_pct = ir_drop_mv / (vdd * 1000) * 100
        metrics["ir_drop_mv"]          = ir_drop_mv
        metrics["ir_drop_pct_vdd"]     = round(ir_pct, 2)
        metrics["ir_drop_limit_pct"]   = IR_DROP_LIMIT_PCT
        if ir_pct > IR_DROP_LIMIT_PCT:
            warnings.append(
                f"IR drop {ir_drop_mv:.1f} mV = {ir_pct:.1f}% of VDD "
                f"EXCEEDS {IR_DROP_LIMIT_PCT}% limit — performance degradation guaranteed"
            )
        elif ir_pct > 3.0:
            warnings.append(f"IR drop {ir_pct:.1f}% of VDD — approaching limit (5%)")

    # ── PDN impedance target check ─────────────────────────────────────────────
    if current_a and vdd:
        allowed_droop_v = vdd * IR_DROP_LIMIT_PCT / 100
        z_target_mohm   = allowed_droop_v / current_a * 1000
        metrics["Z_target_mohm"] = round(z_target_mohm, 3)
        metrics["current_a"]      = current_a
        if pdn_impedance_mohm:
            z_ratio = pdn_impedance_mohm / z_target_mohm
            metrics["Z_actual_mohm"]     = pdn_impedance_mohm
            metrics["Z_vs_target_ratio"] = round(z_ratio, 2)
            if z_ratio > 1.0:
                warnings.append(
                    f"PDN impedance {pdn_impedance_mohm:.2f} mΩ EXCEEDS target "
                    f"{z_target_mohm:.2f} mΩ (ratio = {z_ratio:.2f}×)"
                )
        # Current sweep for IR drop
        for frac in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            i_s = current_a * frac
            # compute IR drop with estimated R_pdn = Z_target as floor
            r_pdn = pdn_impedance_mohm * 0.001 if pdn_impedance_mohm else z_target_mohm * 0.001 * 0.5
            ir_s  = i_s * r_pdn * 1000  # in mV
            ir_pct_s = ir_s / (vdd * 1000) * 100
            sweep.append(ParameterSweepPoint(
                param_name="current_a", param_value=round(i_s, 1),
                metric_name="ir_drop_pct_vdd", metric_value=round(ir_pct_s, 2),
                status="pass" if ir_pct_s < IR_DROP_LIMIT_PCT else "fail",
            ))
        # Find breaking current
        r_est = pdn_impedance_mohm * 0.001 if pdn_impedance_mohm else z_target_mohm * 0.001 * 0.5
        if r_est > 0:
            i_break = allowed_droop_v / r_est
            margin_pct = (i_break - current_a) / i_break * 100
            falsification = FalsificationBoundary(
                domain="pdn", breaking_param="current_a",
                nominal_value=round(current_a, 1), breaking_value=round(i_break, 1),
                safety_margin_pct=round(margin_pct, 1), units="A",
                equation_used=f"IR_drop = I × R_pdn = I × {r_est*1000:.2f}mΩ ≤ {allowed_droop_v*1000:.1f}mV",
                fix_vector=(
                    f"Add decap or reduce R_pdn below {allowed_droop_v/current_a*1000:.2f} mΩ. "
                    "Increase bump density or add on-die capacitance."
                ) if margin_pct < 20 else f"{margin_pct:.0f}% current headroom."
            )

    # ── Frequency-domain impedance ─────────────────────────────────────────────
    if pdn_impedance_mohm and frequency_ghz:
        f_hz = frequency_ghz * 1e9
        # Simple RLC: R=pdn_impedance, L estimated from bump inductance, C=decap
        r_pdn = pdn_impedance_mohm * 1e-3
        l_pdn = 50e-12   # 50 pH typical bump inductance
        c_pdn = (decap_nf * 1e-9) if decap_nf else 100e-9
        f_res = 1 / (2 * math.pi * math.sqrt(l_pdn * c_pdn))
        metrics["PDN_resonance_mhz"] = round(f_res / 1e6, 1)
        metrics["L_pdn_ph"]          = l_pdn * 1e12
        metrics["C_decap_nf"]        = decap_nf or 0.0
        freq_arr = np.logspace(6, 10, 200)
        z_arr = np.abs(r_pdn + 1j*2*math.pi*freq_arr*l_pdn + 1/(1j*2*math.pi*freq_arr*c_pdn))
        z_at_f = float(np.interp(f_hz, freq_arr, z_arr)) * 1000  # mΩ
        metrics["Z_at_freq_mohm"] = round(z_at_f, 3)
        if current_a and vdd:
            z_tgt = vdd * IR_DROP_LIMIT_PCT / 100 / current_a * 1000
            if z_at_f > z_tgt:
                warnings.append(
                    f"Z_PDN at {frequency_ghz} GHz = {z_at_f:.2f} mΩ exceeds target {z_tgt:.2f} mΩ"
                )
        if abs(f_hz - f_res) / f_res < 0.2:
            warnings.append(
                f"Operating frequency {frequency_ghz} GHz is near PDN resonance "
                f"({f_res/1e9:.2f} GHz) — potential noise amplification"
            )

    # ── Electromigration ───────────────────────────────────────────────────────
    if bump_density_per_mm2 and current_a:
        n_bumps = bump_density_per_mm2  # assume 1 mm² die area as minimum
        i_per_bump = current_a / n_bumps * 1000  # mA
        bump_area_um2 = 1.0 / bump_density_per_mm2 * 1e6 * 0.05  # ~5% as conductor
        j_ma_um2 = i_per_bump / bump_area_um2 if bump_area_um2 > 0 else float('inf')
        metrics["current_per_bump_ma"] = round(i_per_bump, 4)
        metrics["EM_current_density_estimate"] = round(j_ma_um2, 3)
        if j_ma_um2 > EM_CURRENT_DENSITY:
            warnings.append(
                f"Estimated EM current density {j_ma_um2:.2f} mA/μm² may EXCEED "
                f"Cu limit {EM_CURRENT_DENSITY} mA/μm² — check bump cross-section"
            )

    # ── Decap sufficiency ──────────────────────────────────────────────────────
    if decap_nf and di_dt_a_per_ns and vdd:
        t_response_ns = 1.0  # assume 1ns response time for PDN
        q_needed_nc  = di_dt_a_per_ns * t_response_ns       # charge needed
        q_decap_nc   = decap_nf * vdd * 0.1 * 1000          # usable charge (10% droop budget)
        metrics["Q_decap_nc"]   = round(q_decap_nc, 3)
        metrics["Q_needed_nc"]  = round(q_needed_nc, 3)
        decap_ratio = q_decap_nc / q_needed_nc
        metrics["decap_ratio"]  = round(decap_ratio, 2)
        if decap_ratio < 1.0:
            warnings.append(
                f"Decap insufficient: Q_decap={q_decap_nc:.2f} nC < Q_needed={q_needed_nc:.2f} nC "
                f"for di/dt={di_dt_a_per_ns} A/ns — VDD droop will EXCEED budget"
            )

    # ── Chiplet PDN scaling ────────────────────────────────────────────────────
    if n_chiplets and n_chiplets > 1:
        metrics["n_chiplets"] = n_chiplets
        warnings.append(
            f"{n_chiplets} chiplets: PDN routing complexity scales nonlinearly — "
            "inter-chiplet ground bounce and shared bump resources need L2 simulation"
        )

    # ── Score ──────────────────────────────────────────────────────────────────
    fatal    = any("EXCEED" in w for w in warnings)
    marginal = any("approaching" in w.lower() or "near" in w.lower() for w in warnings)
    if fatal:
        score, sf = 1.5, "fail"
    elif marginal:
        score, sf = 5.0, "marginal"
    else:
        m = metrics.get("Z_vs_target_ratio", 0.5)
        score = min(10.0, 9.0 - m * 2.0) if m < 1 else 2.0
        sf: SimStatus = "pass" if m < 1 else "fail"
        if not metrics:
            score, sf = 4.0, "insufficient_data"

    return DomainSimResult(
        idea_id=idea_id, domain="pdn", fidelity="L1_analytical",
        status=sf, score=round(score, 2),
        metrics=metrics, sweep_points=sweep, falsification=falsification,
        warnings=warnings,
        solver_notes="DC IR-drop, RLC impedance (numpy), EM check, decap check",
    )
