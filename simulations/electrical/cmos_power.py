"""
cmos_power.py — CMOS power analysis with Landauer gap and density checks.

Computes power breakdown and checks electrical limits.
Thermal feedback is handled by the thermal RC simulator — not duplicated here.
"""
from __future__ import annotations
import numpy as np
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.schemas import ElectricalSimResult

LANDAUER_PJ      = 2.8e-9    # Landauer limit at 300K [pJ]
CMOS_BREAKDOWN   = 1000.0    # W/cm² oxide breakdown
PRACTICAL_LIMIT  = 100.0     # W/cm² cooling limit in practice


def run_electrical_sim(
    power_params: Optional[dict] = None,
    thermal_params: Optional[dict] = None,
    node_nm: float = 5.0,
) -> ElectricalSimResult:
    result = ElectricalSimResult()
    pp = power_params or {}
    tp = thermal_params or {}

    vdd           = pp.get("voltage_v")      or 0.85
    p_dynamic     = pp.get("watt") or pp.get("tdp_watt") or 100.0
    power_density = pp.get("power_density_w_cm2")
    energy_per_op = pp.get("energy_per_op_pj")
    t_operating   = tp.get("t_junction_c") or (tp.get("t_ambient_c") or 25.0) + 60.0

    # ── Leakage estimate at operating temperature ─────────────────────────────
    # I_leak at 25°C ~ 5% of dynamic. Doubles every 10°C.
    p_leak_25c = p_dynamic * 0.05
    # Realistic: leakage ~25% of dynamic at Tj_max=125°C → alpha=0.02/°C
    exp_arg   = min(10.0, 0.02 * (t_operating - 25.0))
    p_leakage = p_leak_25c * np.exp(exp_arg)
    p_total    = p_dynamic + p_leakage
    result.p_dynamic_w            = round(p_dynamic, 3)
    result.p_leakage_w            = round(p_leakage, 3)
    result.p_total_w              = round(p_total, 3)
    result.useful_power_fraction  = round(p_dynamic / p_total, 4)
    result.t_equilibrium_c        = round(t_operating, 1)
    result.converged              = True
    result.iterations_to_converge = 1

    # Power density
    if power_density:
        result.power_density_w_cm2 = round(power_density, 1)
    elif p_dynamic > 0:
        # Estimate area from TDP heuristic: 200W/cm² is aggressive modern chip
        estimated_area_cm2 = p_total / 120.0   # assuming ~120 W/cm² typical
        result.power_density_w_cm2 = round(p_total / estimated_area_cm2, 1)

    # ── Landauer gap ──────────────────────────────────────────────────────────
    if energy_per_op:
        result.energy_per_op_pj = float(energy_per_op)
        result.landauer_ratio   = round(float(energy_per_op) / LANDAUER_PJ, 2)

    # ── Status ────────────────────────────────────────────────────────────────
    pd = result.power_density_w_cm2
    notes = []

    if energy_per_op and float(energy_per_op) < LANDAUER_PJ:
        result.status = "fail"
        notes.append(f"Energy/op {energy_per_op:.2e} pJ < Landauer limit {LANDAUER_PJ:.2e} pJ — physically impossible")
    elif pd and pd > CMOS_BREAKDOWN:
        result.status = "fail"
        notes.append(f"Power density {pd:.0f} W/cm² > oxide breakdown {CMOS_BREAKDOWN:.0f} W/cm²")
    elif pd and pd > PRACTICAL_LIMIT:
        result.status = "warning"
        notes.append(f"Power density {pd:.0f} W/cm² > practical cooling limit {PRACTICAL_LIMIT:.0f} W/cm²")
    elif p_leakage / p_dynamic > 1.5:
        result.status = "warning"
        notes.append(f"Leakage {p_leakage:.1f}W is {p_leakage/p_dynamic*100:.0f}% of dynamic power at T={t_operating:.0f}°C — thermal management critical")
    else:
        result.status = "pass"

    notes.append(f"P_dynamic={p_dynamic:.1f}W | P_leakage={p_leakage:.1f}W ({(1-result.useful_power_fraction)*100:.1f}% overhead) at T={t_operating:.0f}°C")
    if result.landauer_ratio:
        notes.append(f"Energy/op is {result.landauer_ratio:.1e}× Landauer limit — thermodynamic gap")
    if pd:
        notes.append(f"Power density: {pd:.1f} W/cm²")
    result.notes = notes
    return result
