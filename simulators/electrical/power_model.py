"""
L1 Electrical Simulator — CMOS Power + Landauer + IR Drop
Physics:
  P_dynamic = α × C × V² × f         (switching power)
  P_static  = I_leak × V              (leakage)
  P_density = P_total / A             (W/cm²)
  Landauer  = kT × ln(2) = 0.0174 aJ (minimum energy per irreversible bit op)
  IR_drop   = I × R_pdn               (must be < 5% VDD)
Falsification: sweep frequency to find P_density = 100 W/cm² limit
"""
from __future__ import annotations
import math
from typing import Optional, Dict, List
import sys, os
_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
from core.schemas import DomainSimResult, FalsificationBoundary, ParameterSweepPoint, SimStatus

K_BOLTZMANN   = 1.380649e-23   # J/K
T_ROOM_K      = 300.0
LANDAUER_J    = K_BOLTZMANN * T_ROOM_K * math.log(2)   # ~2.85e-21 J
LANDAUER_AJ   = LANDAUER_J * 1e18                       # ~0.0174 aJ
POWER_DENSITY_CMOS_MAX = 100.0  # W/cm² practical limit at 3nm
CMOS_ALPHA    = 0.1             # activity factor (typical)

def simulate_electrical(
    idea_id: str,
    power_w: Optional[float] = None,
    tdp_watt: Optional[float] = None,
    power_density_w_cm2: Optional[float] = None,
    energy_per_op_pj: Optional[float] = None,
    voltage_v: Optional[float] = None,
    current_a: Optional[float] = None,
    efficiency_pct: Optional[float] = None,
    # derived / context
    frequency_ghz: Optional[float] = None,
    die_area_cm2: Optional[float] = None,
    capacitance_ff: Optional[float] = None,    # switching capacitance in fF
    ops_per_cycle: Optional[float] = None,     # for Landauer check
) -> DomainSimResult:
    warnings: List[str] = []
    metrics: Dict[str, float] = {}
    sweep: List[ParameterSweepPoint] = []
    falsification: Optional[FalsificationBoundary] = None

    p_ref = power_w or tdp_watt
    if not any([p_ref, power_density_w_cm2, energy_per_op_pj, voltage_v]):
        return DomainSimResult(
            idea_id=idea_id, domain="electrical", fidelity="L1_analytical",
            status="insufficient_data", score=4.0,
            warnings=["No electrical params — need power, voltage, or energy_per_op"],
        )

    # ── CMOS dynamic power ─────────────────────────────────────────────────────
    if voltage_v and frequency_ghz and capacitance_ff:
        f_hz = frequency_ghz * 1e9
        c_f  = capacitance_ff * 1e-15
        p_dyn = CMOS_ALPHA * c_f * voltage_v**2 * f_hz
        metrics["P_dynamic_w"]   = round(p_dyn, 3)
        metrics["P_dynamic_formula"] = round(p_dyn, 3)
        if p_ref is None:
            p_ref = p_dyn
        # Voltage scaling: P ∝ V², so halving V → 4x power reduction
        for v_scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            p_s = CMOS_ALPHA * c_f * (voltage_v * v_scale)**2 * f_hz
            sweep.append(ParameterSweepPoint(
                param_name="voltage_v", param_value=round(voltage_v * v_scale, 3),
                metric_name="P_dynamic_w", metric_value=round(p_s, 4),
                status="pass",
            ))
        metrics["voltage_scaling_P_at_0.8V_ratio"] = round((0.8 * voltage_v / voltage_v)**2, 3)

    # ── Power density check ────────────────────────────────────────────────────
    if p_ref and die_area_cm2 and die_area_cm2 > 0:
        pd = p_ref / die_area_cm2
        metrics["power_density_w_cm2"]   = round(pd, 2)
        metrics["power_density_limit"]   = POWER_DENSITY_CMOS_MAX
        margin = (POWER_DENSITY_CMOS_MAX - pd) / POWER_DENSITY_CMOS_MAX * 100
        metrics["power_density_margin_pct"] = round(margin, 1)
        if pd > POWER_DENSITY_CMOS_MAX:
            warnings.append(
                f"Power density {pd:.1f} W/cm² EXCEEDS CMOS practical limit {POWER_DENSITY_CMOS_MAX} W/cm²"
            )
        falsification = FalsificationBoundary(
            domain="electrical", breaking_param="power_density_w_cm2",
            nominal_value=round(pd, 2), breaking_value=POWER_DENSITY_CMOS_MAX,
            safety_margin_pct=round(margin, 1), units="W/cm²",
            equation_used="P_density = P_total / A_die",
            fix_vector=(
                f"Need A_die ≥ {p_ref/POWER_DENSITY_CMOS_MAX:.2f} cm² for {p_ref:.0f}W, "
                "OR reduce frequency/voltage. Voltage scaling P ∝ V² is most effective."
            ) if margin < 0 else f"{margin:.1f}% headroom below thermal runaway density."
        )
    elif power_density_w_cm2:
        pd = power_density_w_cm2
        metrics["power_density_w_cm2"] = pd
        margin = (POWER_DENSITY_CMOS_MAX - pd) / POWER_DENSITY_CMOS_MAX * 100
        metrics["power_density_margin_pct"] = round(margin, 1)
        if pd > POWER_DENSITY_CMOS_MAX:
            warnings.append(f"Power density {pd} W/cm² EXCEEDS limit {POWER_DENSITY_CMOS_MAX} W/cm²")
        falsification = FalsificationBoundary(
            domain="electrical", breaking_param="power_density_w_cm2",
            nominal_value=pd, breaking_value=POWER_DENSITY_CMOS_MAX,
            safety_margin_pct=round(margin, 1), units="W/cm²",
            equation_used=f"limit = {POWER_DENSITY_CMOS_MAX} W/cm² (3nm practical)",
            fix_vector="Reduce switching frequency or increase die area.",
        )

    # ── Frequency sweep for power density ─────────────────────────────────────
    if p_ref and die_area_cm2 and frequency_ghz:
        f_break = frequency_ghz * POWER_DENSITY_CMOS_MAX / (p_ref / die_area_cm2)
        for frac in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
            f_s = frequency_ghz * frac
            p_s = p_ref * frac  # P ∝ f (dynamic power)
            pd_s = p_s / die_area_cm2
            sweep.append(ParameterSweepPoint(
                param_name="frequency_ghz", param_value=round(f_s, 2),
                metric_name="power_density_w_cm2", metric_value=round(pd_s, 2),
                status="pass" if pd_s < POWER_DENSITY_CMOS_MAX else "fail",
            ))
        metrics["max_frequency_ghz_before_thermal_death"] = round(f_break, 2)

    # ── Landauer limit check ───────────────────────────────────────────────────
    if energy_per_op_pj is not None:
        e_pj = energy_per_op_pj
        landauer_pj = LANDAUER_J * 1e12  # convert J to pJ = 2.85e-9 pJ
        ratio = e_pj / landauer_pj
        metrics["energy_per_op_pj"]   = e_pj
        metrics["landauer_limit_pj"]  = round(landauer_pj, 12)
        metrics["energy_vs_landauer_ratio"] = round(ratio, 1)
        if e_pj < landauer_pj * 0.99:
            warnings.append(
                f"energy_per_op {e_pj:.4f} pJ VIOLATES Landauer limit "
                f"{landauer_pj:.4e} pJ — PHYSICALLY IMPOSSIBLE"
            )
        elif ratio < 1000:
            warnings.append(
                f"energy_per_op is only {ratio:.0f}× above Landauer — approaching quantum floor. "
                "Real implementations hit kT overhead at ~10⁶× Landauer."
            )

    # ── IR drop ────────────────────────────────────────────────────────────────
    if current_a and voltage_v:
        # Assume 10mΩ PDN resistance as typical baseline
        r_pdn_est = 0.01
        ir_drop_mv = current_a * r_pdn_est * 1000
        ir_drop_pct = ir_drop_mv / (voltage_v * 1000) * 100
        metrics["IR_drop_estimated_mv"]  = round(ir_drop_mv, 2)
        metrics["IR_drop_pct_of_vdd"]    = round(ir_drop_pct, 2)
        if ir_drop_pct > 5.0:
            warnings.append(
                f"Estimated IR drop {ir_drop_mv:.1f} mV = {ir_drop_pct:.1f}% of VDD — "
                "exceeds 5% safe limit → PDN simulation required (L2)"
            )

    # ── Efficiency check ───────────────────────────────────────────────────────
    if efficiency_pct:
        metrics["efficiency_pct"] = efficiency_pct
        if efficiency_pct > 100:
            warnings.append(f"Claimed efficiency {efficiency_pct}% IMPOSSIBLE — violates conservation of energy")
        elif efficiency_pct > 95:
            warnings.append(f"Claimed efficiency {efficiency_pct}% is extremely optimistic for power electronics")

    # ── Score ──────────────────────────────────────────────────────────────────
    fatal    = any("EXCEED" in w or "VIOLAT" in w or "IMPOSSIBLE" in w for w in warnings)
    marginal = any("5%" in w or "marginal" in w.lower() for w in warnings)
    landauer_fail = any("Landauer" in w and "VIOLATES" in w for w in warnings)

    if landauer_fail:
        score, status_f = 0.0, "fail"
    elif fatal:
        score, status_f = 1.5, "fail"
    elif marginal:
        score, status_f = 5.0, "marginal"
    else:
        m = metrics.get("power_density_margin_pct", 50.0)
        score = min(10.0, 5.0 + m * 0.08)
        status_f: SimStatus = "pass"

    return DomainSimResult(
        idea_id=idea_id, domain="electrical", fidelity="L1_analytical",
        status=status_f, score=round(score, 2),
        metrics=metrics, sweep_points=sweep, falsification=falsification,
        warnings=warnings,
        solver_notes="CMOS power model: P=αCV²f, Landauer check, IR drop estimate",
    )
