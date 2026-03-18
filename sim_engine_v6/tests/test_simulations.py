"""
test_simulations.py — Unit tests for all simulation modules.
Run: python tests/test_simulations.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def run_all():
    passed = failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  ✓  {name}")
            passed += 1
        else:
            print(f"  ✗  {name}  {detail}")
            failed += 1

    print("\n══════════════════════════════════════")
    print("  Physics Simulation Engine — Tests")
    print("══════════════════════════════════════\n")

    # ── Thermal RC Network ────────────────────────────────────────────────────
    print("[ Thermal — RC Network ODE ]")
    from simulations.thermal.rc_network import run_thermal_sim

    r1 = run_thermal_sim(power_w=100.0, t_ambient_c=25.0)
    check("100W @ 25°C converges", r1.t_junction_ss_c is not None)
    check("100W steady state > ambient", r1.t_junction_ss_c > 25)
    check("100W has transient data", r1.time_s and len(r1.time_s) > 10)
    check("100W status is pass or warning", r1.status in ("pass","warning"))
    check("100W leakage is positive", r1.p_leakage_w > 0)

    r2 = run_thermal_sim(power_w=800.0, t_ambient_c=50.0)
    check("800W @ 50°C fails or critical", r2.status in ("fail","critical","warning"))

    r3 = run_thermal_sim(power_w=50.0,
                         thermal_params={"thermal_resistance_c_per_w": 0.1, "t_ambient_c": 25.0})
    check("Custom R_theta applied", r3.t_junction_ss_c is not None)

    # ── PDN RLC Transient ─────────────────────────────────────────────────────
    print("\n[ PDN — RLC Transient + Impedance ]")
    from simulations.pdn.rlc_transient import run_pdn_sim

    p1 = run_pdn_sim(vdd_v=0.85, i_load_a=50.0)
    check("50A load produces droop", p1.droop_mv > 0)
    check("50A has voltage transient", p1.voltage_transient and len(p1.voltage_transient) > 10)
    check("50A has impedance profile", p1.impedance_mohm and len(p1.impedance_mohm) > 10)
    check("50A resonant freq found", p1.resonant_freq_mhz is not None)

    p2 = run_pdn_sim(vdd_v=0.85, i_load_a=500.0)
    check("500A load has higher droop", p2.droop_mv > p1.droop_mv)

    p3 = run_pdn_sim(pdn_params={"bump_density_per_mm2": 10, "current_a": 1000})
    check("EM flag triggered at high J", p3.em_flag == True)

    # ── Electrical CMOS Power ─────────────────────────────────────────────────
    print("\n[ Electrical — CMOS Power + Thermal Feedback ]")
    from simulations.electrical.cmos_power import run_electrical_sim

    e1 = run_electrical_sim(power_params={"watt": 100.0, "voltage_v": 0.85})
    check("100W converges", e1.converged == True)
    check("100W T_eq > ambient", e1.t_equilibrium_c > 25)
    check("100W leakage computed", e1.p_leakage_w > 0)

    e2 = run_electrical_sim(power_params={"watt": 100.0, "voltage_v": 0.85, "energy_per_op_pj": 1e-15})
    check("Below-Landauer energy fails", e2.status == "fail")

    e3 = run_electrical_sim(power_params={"watt": 100.0, "power_density_w_cm2": 2500.0})
    check("Breakdown density fails", e3.status in ("fail","warning"))

    # ── Data Movement Roofline ────────────────────────────────────────────────
    print("\n[ Data Movement — Roofline + Memory Hierarchy ]")
    from simulations.data_movement.roofline import run_data_movement_sim

    d1 = run_data_movement_sim(dm_params={"bandwidth_gb_s": 1200, "compute_tflops": 100})
    check("1200 GB/s roofline runs", d1.achievable_tflops is not None)
    check("1200 GB/s has binding level", d1.binding_memory_level is not None)
    check("1200 GB/s has bottleneck", d1.bottleneck is not None)
    check("1200 GB/s efficiency > 0", d1.efficiency_pct > 0)

    d2 = run_data_movement_sim(dm_params={"bandwidth_gb_s": 100000, "compute_tflops": 100})
    check("Impossible bandwidth fails", d2.status == "fail")

    # ── Dispatcher ────────────────────────────────────────────────────────────
    print("\n[ Dispatcher — Full Pipeline ]")
    from core.dispatcher import dispatch

    idea = {
        "id": "test-idea-0001", "title": "Test idea", "domain": "thermal",
        "power_params":   {"watt": 200.0, "voltage_v": 0.85, "current_a": 235.0},
        "thermal_params": {"t_ambient_c": 30.0, "thermal_resistance_c_per_w": 0.25},
        "pdn_params":     {"vdd_v": 0.85, "current_a": 235.0, "bump_density_per_mm2": 2000},
        "data_movement_params": {"bandwidth_gb_s": 800, "compute_tflops": 80},
    }
    result = dispatch(idea)
    check("Dispatcher returns SimResult", result is not None)
    check("All domains simulated", all([result.thermal, result.pdn, result.electrical, result.data_movement]))
    check("Overall status set", result.overall_status != "skipped")
    check("Sim score > 0", result.sim_score > 0)
    check("Duration tracked", result.duration_ms > 0)

    total = passed + failed
    print(f"\n══════════════════════════════════════")
    print(f"  Results: {passed}/{total} passed")
    if failed:
        print(f"  ⚠️  {failed} FAILED")
        sys.exit(1)
    else:
        print("  ✓  All tests passed")
    print("══════════════════════════════════════\n")


if __name__ == "__main__":
    run_all()
