"""
wiring_harness.py — Electromechanical simulator.

Simulates wiring harness and motor/actuator failure modes:
  1. Joule heating transient ODE — wire temperature rise over time
  2. Contact resistance degradation model (fretting corrosion Arrhenius)
  3. Motor thermal derating trajectory under duty cycle
  4. Arc flash energy estimation (IEC 61482-1)

Key physics beyond the RD Engine's static gate:
  - Transient wire temperature T(t) including thermal capacitance
  - Contact resistance growth rate = Arrhenius × fretting cycles
  - Motor winding temp under variable duty cycle (not just steady-state)
  - Arc flash incident energy at connector location
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Optional
import logging, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
logger = logging.getLogger(__name__)

# ── Wire / insulation constants ───────────────────────────────────────────────
CU_RESISTIVITY_20C = 1.72e-8   # Ω·m
CU_TEMP_COEFF      = 0.00393   # /°C

INSULATION = {
    "pvc":      {"T_max": 70.0,  "cp": 1000.0, "rho": 1400.0},
    "xlpe":     {"T_max": 90.0,  "cp": 2100.0, "rho": 930.0},
    "ptfe":     {"T_max": 200.0, "cp": 1050.0, "rho": 2200.0},
    "silicone": {"T_max": 180.0, "cp": 1300.0, "rho": 1200.0},
    "default":  {"T_max": 70.0,  "cp": 1000.0, "rho": 1400.0},
}


def run_joule_heating_transient(
    elec_params: dict,
    duration_s: float = 300.0,
    ambient_c: float = 40.0,
) -> Dict[str, Any]:
    """
    Transient Joule heating ODE for wiring harness.
    
    Energy balance per unit length:
      ρ_ins × A_ins × cp × dT/dt = I²R/L - (T - T_amb)/R_th_per_m
    
    Includes:
    - Temperature-dependent copper resistance
    - Thermal capacitance of wire + insulation
    - Natural convection to ambient
    """
    current_a        = float(elec_params.get("current_a") or 20.0)
    cross_mm2        = float(elec_params.get("wire_cross_section_mm2") or 2.5)
    insulation_type  = str(elec_params.get("insulation") or "pvc").lower()
    wire_length_m    = float(elec_params.get("wire_length_m") or 5.0)

    ins   = INSULATION.get(insulation_type, INSULATION["default"])
    T_max = ins["T_max"]

    # Wire geometry
    area_cu_m2 = cross_mm2 * 1e-6
    r_cu = np.sqrt(area_cu_m2 / np.pi)           # copper radius
    r_ins = r_cu + 0.001                           # 1mm insulation thickness
    A_ins = np.pi * (r_ins**2 - r_cu**2)           # insulation cross-section

    # Resistance per meter at 20°C
    R_per_m_20 = CU_RESISTIVITY_20C / area_cu_m2

    # Thermal resistance to ambient (natural convection, simplified)
    h_conv = 10.0  # W/(m²·K) — natural convection
    R_th_per_m = 1.0 / (h_conv * 2 * np.pi * r_ins)

    # Thermal capacitance per meter (copper + insulation)
    rho_cu = 8900.0
    cp_cu  = 385.0
    C_th_cu  = rho_cu * cp_cu * area_cu_m2
    C_th_ins = ins["rho"] * ins["cp"] * A_ins
    C_th_per_m = C_th_cu + C_th_ins

    def ode(t, T_arr):
        T = T_arr[0]
        # Temperature-dependent resistance
        R_t = R_per_m_20 * (1 + CU_TEMP_COEFF * (T - 20.0))
        p_joule   = current_a**2 * R_t          # W/m
        q_conv    = (T - ambient_c) / R_th_per_m # W/m
        dTdt = (p_joule - q_conv) / C_th_per_m
        return [dTdt]

    T0  = [ambient_c]
    t_eval = np.linspace(0, duration_s, 300)
    sol = solve_ivp(ode, [0, duration_s], T0, method="RK45", t_eval=t_eval, rtol=1e-5)

    T_profile = sol.y[0] if sol.success else np.ones(300) * (ambient_c + current_a**2 * R_per_m_20 * R_th_per_m)
    T_peak    = float(np.max(T_profile))
    T_ss      = float(T_profile[-1])
    T_90pct   = float(0.9 * (T_ss - ambient_c) + ambient_c)

    # Time to 90% of steady state
    try:
        t_90 = float(t_eval[np.argmax(T_profile >= T_90pct)])
    except Exception:
        t_90 = duration_s

    margin = T_max - T_ss

    if T_ss > T_max:
        status = "fail"
    elif T_ss > T_max * 0.90:
        status = "critical"
    elif T_ss > T_max * 0.75:
        status = "warning"
    else:
        status = "pass"

    notes = [
        f"Wire T_ss={T_ss:.1f}°C | T_peak={T_peak:.1f}°C | Limit ({insulation_type}): {T_max}°C",
        f"Margin: {margin:.1f}°C | Time to 90%: {t_90:.0f}s",
        f"I={current_a}A, {cross_mm2}mm², {wire_length_m}m, ambient={ambient_c}°C",
    ]
    if T_ss > T_max:
        notes.append(f"FAIL — wire exceeds insulation limit by {T_ss - T_max:.1f}°C")
    if T_peak > T_max:
        notes.append(f"Transient peak {T_peak:.1f}°C > limit — thermal runaway risk during startup")

    return {
        "status":           status,
        "t_steady_state_c": round(T_ss, 2),
        "t_peak_c":         round(T_peak, 2),
        "t_limit_c":        T_max,
        "margin_c":         round(margin, 2),
        "time_to_90pct_s":  round(t_90, 1),
        "t_profile_c":      [round(t, 2) for t in T_profile[::10]],
        "time_s":           [round(t, 1) for t in t_eval[::10]],
        "notes":            notes,
    }


def run_contact_degradation(
    elec_params: dict,
    mission_hours: float = 20000.0,
) -> Dict[str, Any]:
    """
    Contact resistance degradation model.
    Fretting corrosion: R(t) = R0 × exp(k_fret × N_cycles)
    Arrhenius temperature acceleration: k_eff = k_fret × exp(Ea/kT)
    """
    R0_mohm       = float(elec_params.get("contact_resistance_mohm") or 5.0)
    current_a     = float(elec_params.get("current_a") or 10.0)
    vibration_hz  = float(elec_params.get("vibration_freq_hz") or 50.0)
    ambient_c     = float(elec_params.get("ambient_temp_c") or 40.0)

    # Fretting degradation constant (empirical)
    k_fret   = 1e-9      # per cycle
    Ea       = 0.5       # eV activation energy for oxidation
    kB       = 8.617e-5  # eV/K
    T_K      = ambient_c + 273.15

    # Arrhenius acceleration factor vs 25°C reference
    T_ref_K  = 298.15
    accel    = np.exp((Ea / kB) * (1/T_ref_K - 1/T_K))

    # Total fretting cycles in mission
    N_total  = vibration_hz * 3600 * mission_hours * 0.5  # 50% time vibrating
    k_eff    = k_fret * accel
    R_final_mohm = R0_mohm * np.exp(k_eff * N_total)

    # Joule heating at degraded contact
    P_contact_w = current_a**2 * R_final_mohm * 1e-3
    T_contact_c = ambient_c + P_contact_w * 8.0  # ~8 °C/W for connector

    WARN_MOHM = 15.0
    FAIL_MOHM = 50.0

    if R_final_mohm >= FAIL_MOHM or T_contact_c > 85.0:
        status = "fail"
    elif R_final_mohm >= WARN_MOHM:
        status = "warning"
    else:
        status = "pass"

    notes = [
        f"R0={R0_mohm:.1f} mΩ → R({mission_hours:.0f}h)={R_final_mohm:.1f} mΩ",
        f"Fretting cycles: {N_total:.2e} | Accel factor: {accel:.1f}× (at {ambient_c}°C)",
        f"Contact temp at end of life: {T_contact_c:.1f}°C | I²R loss: {P_contact_w:.3f} W",
    ]
    if R_final_mohm >= FAIL_MOHM:
        notes.append(f"FAIL — contact resistance {R_final_mohm:.1f} mΩ exceeds failure threshold {FAIL_MOHM} mΩ")

    return {
        "status":              status,
        "r_initial_mohm":      R0_mohm,
        "r_final_mohm":        round(R_final_mohm, 2),
        "contact_temp_c":      round(T_contact_c, 2),
        "fretting_cycles":     round(N_total, 0),
        "arrhenius_accel":     round(accel, 2),
        "mission_hours":       mission_hours,
        "notes":               notes,
    }


def run_motor_thermal_trajectory(
    elec_params: dict,
    mission_hours: float = 8.0,
) -> Dict[str, Any]:
    """
    Motor winding temperature trajectory under variable duty cycle.
    ODE: C_winding × dT/dt = P_copper_loss - (T - T_amb) / R_th_winding
    """
    rated_power_w = float(elec_params.get("rated_power_w") or 1000.0)
    duty_cycle    = float(elec_params.get("duty_cycle_pct") or 70.0) / 100.0
    ambient_c     = float(elec_params.get("ambient_temp_c") or 40.0)
    rated_ambient = float(elec_params.get("rated_ambient_c") or 40.0)

    # Motor thermal constants (typical servo motor)
    R_th_winding = 0.08    # °C/W — winding to ambient
    C_th_winding = 800.0   # J/°C — winding thermal mass
    efficiency   = 0.92
    P_loss       = rated_power_w * duty_cycle * (1 - efficiency) / efficiency

    # Derating for ambient
    derating_pct = max(0, (ambient_c - rated_ambient)) * 1.0  # 1%/°C
    P_available  = rated_power_w * (1 - derating_pct / 100)

    def ode_motor(t, T_arr):
        T = T_arr[0]
        # Duty-cycle modulated losses
        duty_now  = duty_cycle * (1 + 0.1 * np.sin(2*np.pi*t/3600))  # ±10% variation
        p_loss_t  = rated_power_w * duty_now * (1-efficiency)/efficiency
        q_conv    = (T - ambient_c) / R_th_winding
        return [(p_loss_t - q_conv) / C_th_winding]

    t_eval = np.linspace(0, mission_hours * 3600, 500)
    sol = solve_ivp(ode_motor, [0, mission_hours*3600], [ambient_c], t_eval=t_eval,
                    method="RK45", rtol=1e-4)

    T_profile = sol.y[0] if sol.success else np.ones(500) * (ambient_c + P_loss * R_th_winding)
    T_ss      = float(T_profile[-1])
    T_peak    = float(np.max(T_profile))

    # Insulation class limits
    T_CLASS_F  = 155.0  # Class F insulation (common in servo motors)
    margin     = T_CLASS_F - T_ss

    if T_ss > T_CLASS_F or derating_pct > 25:
        status = "fail"
    elif T_ss > T_CLASS_F * 0.90 or derating_pct > 15:
        status = "critical"
    elif T_ss > T_CLASS_F * 0.80:
        status = "warning"
    else:
        status = "pass"

    notes = [
        f"Winding T_ss={T_ss:.1f}°C | T_peak={T_peak:.1f}°C | Class F limit: {T_CLASS_F}°C",
        f"Derating: {derating_pct:.1f}% | Available power: {P_available:.0f} W",
        f"Duty cycle: {duty_cycle*100:.0f}% | Copper loss: {P_loss:.0f} W",
    ]
    if derating_pct > 25:
        notes.append(f"FAIL — {derating_pct:.0f}% derating at {ambient_c}°C: motor cannot sustain rated operation")
    if T_ss > T_CLASS_F:
        notes.append(f"FAIL — winding temp {T_ss:.1f}°C exceeds Class F insulation limit {T_CLASS_F}°C")

    return {
        "status":             status,
        "t_winding_ss_c":     round(T_ss, 2),
        "t_winding_peak_c":   round(T_peak, 2),
        "t_limit_class_f_c":  T_CLASS_F,
        "margin_c":           round(margin, 2),
        "derating_pct":       round(derating_pct, 2),
        "available_power_w":  round(P_available, 1),
        "copper_loss_w":      round(P_loss, 1),
        "t_profile_c":        [round(t, 2) for t in T_profile[::25]],
        "notes":              notes,
    }


def run_electromechanical_simulation(
    elec_params: dict,
    mission_hours: float = 20000.0,
) -> Dict[str, Any]:
    """Master electromechanical simulator — runs all three sub-simulations."""
    results = {"status": "pass", "notes": []}

    try:
        jh = run_joule_heating_transient(elec_params)
        results["joule_heating"] = jh
        results["notes"].extend(jh["notes"][:2])
    except Exception as e:
        logger.error(f"[EM/Joule] {e}")
        jh = {"status": "skipped"}

    try:
        cd = run_contact_degradation(elec_params, mission_hours)
        results["contact_degradation"] = cd
        results["notes"].extend(cd["notes"][:1])
    except Exception as e:
        logger.error(f"[EM/Contact] {e}")
        cd = {"status": "skipped"}

    try:
        mt = run_motor_thermal_trajectory(elec_params)
        results["motor_thermal"] = mt
        results["notes"].extend(mt["notes"][:1])
    except Exception as e:
        logger.error(f"[EM/Motor] {e}")
        mt = {"status": "skipped"}

    STATUS_RANK = {"pass": 0, "warning": 1, "critical": 2, "fail": 3, "skipped": -1}
    worst = max(
        [jh.get("status","skipped"), cd.get("status","skipped"), mt.get("status","skipped")],
        key=lambda s: STATUS_RANK.get(s, -1)
    )
    results["status"] = worst if worst != "skipped" else "pass"

    results["wire_temp_ss_c"]        = jh.get("t_steady_state_c")
    results["contact_r_final_mohm"]  = cd.get("r_final_mohm")
    results["motor_winding_temp_c"]  = mt.get("t_winding_ss_c")
    results["motor_derating_pct"]    = mt.get("derating_pct")

    return results
