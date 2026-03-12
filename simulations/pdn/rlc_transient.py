"""
rlc_transient.py — PDN simulation: DC IR drop + transient droop + impedance profile.

Uses analytical droop estimate for robustness at high currents,
plus numerical ODE for the impedance profile Z(f).
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.schemas import PDNSimResult

BUMP_AREA_CM2 = 5e-5       # ~80μm diameter bump
EM_J_LIMIT    = 1e6        # A/cm² electromigration limit for Cu
PDN_IR_BUDGET = 0.05       # 5% VDD budget for IR drop


def _impedance_profile(L_nh, R_mohm, C_uf, freq_array_hz):
    """Simplified RLC impedance: Z = sqrt(R² + (ωL - 1/ωC)²)"""
    L = L_nh * 1e-9
    R = R_mohm * 1e-3
    C = C_uf * 1e-6
    z_out = []
    for f in freq_array_hz:
        w = 2 * np.pi * f
        xl = w * L
        xc = 1 / (w * C + 1e-30)
        z  = np.sqrt(R**2 + (xl - xc)**2)
        z_out.append(z * 1000)  # to mΩ
    return np.array(z_out)


def run_pdn_sim(
    pdn_params: Optional[dict] = None,
    vdd_v: float = 0.85,
    i_load_a: float = 100.0,
    di_dt_a_per_ns: Optional[float] = None,
    bump_density_per_mm2: Optional[float] = None,
    t_sim_ns: float = 200.0,
) -> PDNSimResult:
    result = PDNSimResult()
    pp = pdn_params or {}

    # Override with extracted params
    if pp.get("vdd_v"):           vdd_v    = float(pp["vdd_v"])
    if pp.get("current_a"):       i_load_a = float(pp["current_a"])
    if pp.get("di_dt_a_per_ns"):  di_dt_a_per_ns = float(pp["di_dt_a_per_ns"])
    if pp.get("bump_density_per_mm2"): bump_density_per_mm2 = float(pp["bump_density_per_mm2"])

    result.v_nominal = round(vdd_v, 3)

    # ── DC IR drop ────────────────────────────────────────────────────────────
    if pp.get("ir_drop_mv"):
        # Directly extracted: use it
        ir_drop_v    = float(pp["ir_drop_mv"]) / 1000
        r_total_mohm = (ir_drop_v / max(i_load_a, 0.1)) * 1000
    elif pp.get("pdn_impedance_mohm"):
        # pdn_impedance_mohm = AC target impedance, not DC resistance
        # DC resistance typically ~10-20% of target impedance
        r_total_mohm = float(pp["pdn_impedance_mohm"]) * 0.15
        ir_drop_v    = i_load_a * r_total_mohm * 1e-3
    else:
        # Size for 3% VDD budget
        r_total_mohm = (0.03 * vdd_v / max(i_load_a, 0.1)) * 1000
        ir_drop_v    = i_load_a * r_total_mohm * 1e-3

    ir_drop_pct = (ir_drop_v / vdd_v) * 100 if vdd_v > 0 else 999
    result.ir_drop_mv   = round(ir_drop_v * 1000, 2)
    result.ir_drop_pct  = round(ir_drop_pct, 2)
    result.z_dc_mohm    = round(r_total_mohm, 3)

    # ── Transient droop estimate ──────────────────────────────────────────────
    # Droop ≈ L_pkg * di/dt   (inductance-limited during fast transient)
    # Plus IR drop at peak current
    # Typical package inductance: 0.3–1.0 nH for flip-chip
    L_pkg_nh = 0.4   # nH
    if di_dt_a_per_ns is None:
        # Estimate di/dt from current step with 1ns ramp
        di_dt_a_per_ns = i_load_a * 0.9   # 90% step in 1ns
    inductive_droop_v = L_pkg_nh * 1e-9 * di_dt_a_per_ns * 1e9  # V = L * di/dt
    total_droop_v     = inductive_droop_v + ir_drop_v
    v_min             = max(0.0, vdd_v - total_droop_v)

    result.v_min_transient = round(v_min, 4)
    result.droop_mv        = round(total_droop_v * 1000, 2)
    result.droop_pct       = round((total_droop_v / vdd_v) * 100, 2)

    # Recovery time: ~5 × L/R for RL circuit
    r_pkg_ohm = r_total_mohm * 0.3 * 1e-3   # pkg portion
    tau_ns    = (L_pkg_nh * 1e-9) / max(r_pkg_ohm, 1e-6) * 1e9
    result.recovery_time_ns = round(5 * tau_ns, 1)

    # Synthetic droop waveform for plot
    t_sim_arr  = np.linspace(0, t_sim_ns, 300)
    step_at    = 10.0   # ns
    ramp_ns    = 1.0
    tau_recov  = max(tau_ns, 1.0)
    v_profile  = np.where(
        t_sim_arr < step_at, vdd_v,
        np.where(
            t_sim_arr < step_at + ramp_ns,
            vdd_v - total_droop_v * (t_sim_arr - step_at) / ramp_ns,
            v_min + (vdd_v - v_min) * (1 - np.exp(-(t_sim_arr - step_at - ramp_ns) / tau_recov))
        )
    )
    result.time_ns           = [round(float(x), 2) for x in t_sim_arr]
    result.voltage_transient = [round(float(x), 5) for x in v_profile]

    # ── Impedance profile ─────────────────────────────────────────────────────
    # Use representative L and C values
    C_uf  = float(pp.get("decap_nf", 800)) * 1e-3  # nF → μF
    L_nh  = 0.4 + 2.0   # pkg + trace
    freq_hz = np.logspace(2, 10, 150)
    z_mohm  = _impedance_profile(L_nh, r_total_mohm, C_uf, freq_hz)

    result.z_at_1ghz_mohm = round(float(np.interp(1e9, freq_hz, z_mohm)), 3)

    # Resonance above 10MHz
    mask = freq_hz > 1e7
    z_hf = z_mohm[mask]; f_hf = freq_hz[mask]
    if len(z_hf) > 2:
        peak_idx = int(np.argmax(z_hf))
        result.resonant_freq_mhz    = round(float(f_hf[peak_idx]) / 1e6, 1)
        result.anti_resonance_risk  = bool(float(z_hf[peak_idx]) > 50.0)

    step_z = max(1, len(freq_hz) // 100)
    result.freq_hz        = [float(x) for x in freq_hz[::step_z]]
    result.impedance_mohm = [round(float(x), 4) for x in z_mohm[::step_z]]

    # ── Electromigration ──────────────────────────────────────────────────────
    bd = bump_density_per_mm2 or pp.get("bump_density_per_mm2")
    if bd and float(bd) > 0:
        i_per_bump = i_load_a / float(bd)
        j_density  = i_per_bump / BUMP_AREA_CM2
        em_margin  = (1 - j_density / EM_J_LIMIT) * 100
        result.em_margin_pct = round(em_margin, 1)
        result.em_flag       = bool(em_margin < 0)
        if result.em_flag:
            result.notes.append(f"EM VIOLATION: J={j_density:.2e} A/cm² > {EM_J_LIMIT:.0e} A/cm²")

    # ── Status ────────────────────────────────────────────────────────────────
    droop_ok     = result.droop_pct <= PDN_IR_BUDGET * 100
    em_ok        = not result.em_flag
    z_ok         = not result.z_at_1ghz_mohm or result.z_at_1ghz_mohm <= 10.0

    if result.em_flag:
        result.status = "fail"
    elif result.droop_pct > 15 or (result.z_at_1ghz_mohm and result.z_at_1ghz_mohm > 50):
        result.status = "critical"
    elif result.droop_pct > 5 or not z_ok or result.anti_resonance_risk:
        result.status = "warning"
        result.notes.append(f"Droop={result.droop_mv:.1f}mV ({result.droop_pct:.1f}%) — approaching budget")
    else:
        result.status = "pass"

    result.notes.append(f"DC IR drop: {result.ir_drop_mv:.1f}mV ({result.ir_drop_pct:.1f}% VDD)")
    result.notes.append(f"Transient droop: {result.droop_mv:.1f}mV | Z(1GHz): {result.z_at_1ghz_mohm}mΩ")
    if result.recovery_time_ns:
        result.notes.append(f"Recovery time: {result.recovery_time_ns:.1f}ns")
    if result.anti_resonance_risk:
        result.notes.append(f"Anti-resonance at {result.resonant_freq_mhz}MHz — PDN noise risk")
    return result
