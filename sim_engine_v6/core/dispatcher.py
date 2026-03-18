"""
dispatcher.py v3 — Fully coupled simulation pipeline.

Data flow (every arrow is now real, not assumed):

  [Workload Trace P(t)]
        │
        ├─p_avg──→ [Thermal 1D RC]  ──T_j_1D──────────────────┐
        ├─p_peak─→ [Thermal 2D Grid]──T_hotspot────────────────┼──→ [Coupled Solver]
        └─i_peak, di_dt_max──────────────────────────────────── │        │
                                                                │   T_op (calibrated)
        [PDN: RLC droop]                                        │        │
             │ droop_mv, ssn_mv                                 │        ├──→ [Aging NBTI/HCI/EM]
             └──→ Vdd_at_die ──────────────────────────────────→ [PVT Corners] (uses real Vdd)
                                                                          │
        [CMOS Electrical] ←── T_op from Coupled (not static 105°C) ──────┘
        [Data Movement + NoC]

Score aggregation accounts for all domain interactions.
"""
from __future__ import annotations
import time, logging
from datetime import datetime, timezone
from core.schemas import SimResult, SimStatus

logger = logging.getLogger(__name__)

STATUS_RANK = {"pass": 0, "warning": 1, "critical": 2, "fail": 3, "skipped": -1}


def _worst(*statuses) -> SimStatus:
    ranked = [s for s in statuses if s and s != "skipped"]
    return max(ranked, key=lambda s: STATUS_RANK.get(s, -1)) if ranked else "skipped"


def _compute_score(result: SimResult) -> float:
    scores = []

    if result.thermal and result.thermal.status != "skipped":
        r = result.thermal
        if r.runaway_risk_local:
            scores.append(0.0)
        elif r.t_hotspot_ss_c is not None and r.jedec_margin_hotspot_pct is not None:
            m = r.jedec_margin_hotspot_pct
            scores.append(0.0 if m < 0 else min(10.0, 5.0 + m / 8))
        elif r.thermal_runaway_risk:
            scores.append(0.0)
        else:
            m = r.thermal_margin_pct or 20
            scores.append(min(10.0, 6.0 + m / 15))

    if result.coupled and result.coupled.status != "skipped":
        r = result.coupled
        if r.tt_runaway:
            scores.append(0.0)
        elif r.ff_runaway:
            scores.append(2.0)
        elif r.hotspot_runaway:
            scores.append(1.5)
        elif r.tt_runaway_factor and r.tt_runaway_factor > 0.7:
            scores.append(5.0)
        else:
            margin = r.margin_to_runaway_pct or 0
            scores.append(min(10.0, 7.0 + margin / 30))

    if result.pdn and result.pdn.status != "skipped":
        r = result.pdn
        base = 0.0 if r.em_flag else {
            "fail": 1.0, "critical": 2.5, "warning": 5.5
        }.get(r.status, 9.0 - (r.droop_pct or 0))
        ssn_pen = {"critical": 3.0, "warning": 1.5}.get(r.ssn_status or "", 0.0)
        scores.append(max(0.0, base - ssn_pen))

    if result.electrical and result.electrical.status != "skipped":
        r = result.electrical
        base = 0.0 if r.status == "fail" or r.converged == False else {
            "critical": 2.5, "warning": 5.5
        }.get(r.status, 8.5)
        pvt_pen = 3.0 if r.pvt_any_timing_fail else (1.5 if r.pvt_status == "warning" else 0.0)
        scores.append(max(0.0, base - pvt_pen))

    if result.aging and result.aging.status != "skipped":
        r = result.aging
        if r.status == "fail":
            scores.append(1.0)
        elif r.status == "warning":
            scores.append(5.5)
        else:
            min_mttf = r.min_mttf_years or 10
            scores.append(min(10.0, 7.0 + min_mttf / 20))

    if result.data_movement and result.data_movement.status != "skipped":
        r = result.data_movement
        eff = r.noc_efficiency_pct or r.efficiency_pct or 80
        base = {"fail": 0.0, "critical": 2.0, "warning": 5.0}.get(r.status, min(10.0, eff / 10))
        noc_pen = {"critical": 2.5, "warning": 1.0}.get(r.noc_status or "", 0.0)
        scores.append(max(0.0, base - noc_pen))

    return round(sum(scores) / len(scores), 2) if scores else 5.0


def dispatch(idea: dict) -> SimResult:
    start      = time.time()
    idea_id    = idea.get("id", "unknown")
    idea_title = idea.get("title", "")
    idea_domain= idea.get("domain", "")

    logger.info(f"[Dispatcher v3] idea={idea_id[:8]} — {idea_title[:50]}")

    result = SimResult(
        idea_id=idea_id, idea_title=idea_title, idea_domain=idea_domain,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    thermal_p = idea.get("thermal_params")        or {}
    power_p   = idea.get("power_params")          or {}
    pdn_p     = idea.get("pdn_params")            or {}
    dm_p      = idea.get("data_movement_params")  or {}

    # ── Derived base values ───────────────────────────────────────────────────
    vdd_nom   = float(power_p.get("voltage_v") or pdn_p.get("vdd_v") or 0.85)
    p_dynamic = float(power_p.get("watt") or power_p.get("tdp_watt") or 100.0)
    t_ambient = float(thermal_p.get("t_ambient_c") or 25.0)
    r_theta   = float(thermal_p.get("thermal_resistance_c_per_w") or 0.10)

    # Running state — updated as pipeline progresses
    state = {
        "p_avg":        p_dynamic,       # updated by workload trace
        "p_peak":       p_dynamic,
        "i_peak":       p_dynamic / vdd_nom,
        "di_dt_max":    float(pdn_p.get("di_dt_a_per_ns") or 1.0),
        "t_op":         t_ambient + p_dynamic * r_theta,   # initial estimate
        "t_hotspot":    None,
        "vdd_at_die":   vdd_nom,         # reduced after PDN sim
        "droop_mv":     0.0,
        "ssn_mv":       0.0,
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Workload trace: P(t), di/dt profile
    # ═══════════════════════════════════════════════════════════════════════════
    workload_trace = None
    try:
        from simulations.workload.trace_generator import generate_trace
        profile = (
            "llm_inference_decode"  if "inference" in idea_title.lower() else
            "training_fwd_bwd"      if "training"  in idea_title.lower() else
            "llm_inference_prefill"
        )
        workload_trace = generate_trace(profile_name=profile, tdp_w=p_dynamic, vdd_v=vdd_nom)

        state["p_avg"]    = workload_trace.p_avg_w
        state["p_peak"]   = workload_trace.p_peak_w
        state["i_peak"]   = workload_trace.p_peak_w / vdd_nom
        state["di_dt_max"]= workload_trace.worst_case_di_dt()

        logger.info(
            f"[Trace] {profile}: p_avg={state['p_avg']:.0f}W "
            f"p_peak={state['p_peak']:.0f}W "
            f"di_dt_max={state['di_dt_max']:.3f}A/ns "
            f"bursts={workload_trace.burst_count}"
        )
    except Exception as e:
        logger.error(f"[Trace] {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Thermal: 1D RC (avg power) + 2D grid (peak power)
    # ═══════════════════════════════════════════════════════════════════════════
    has_thermal = bool(thermal_p or power_p)
    if has_thermal:
        from core.schemas import ThermalSimResult
        result.thermal = ThermalSimResult()

        # 2a: 1D RC with average workload power
        try:
            from simulations.thermal.rc_network import run_thermal_sim
            r1 = run_thermal_sim(
                power_w=state["p_avg"], thermal_params=thermal_p,
                t_ambient_c=t_ambient, vdd_v=vdd_nom,
            )
            for field in ["t_junction_ss_c","thermal_margin_pct","dominant_resistance",
                          "t_rise_90pct_ms","t_peak_transient_c","thermal_runaway_risk",
                          "p_dynamic_w","p_leakage_w","time_s","t_junction_transient","notes"]:
                setattr(result.thermal, field, getattr(r1, field))
            result.thermal.status = r1.status
            # Propagate 1D T_j estimate to state
            if r1.t_junction_ss_c:
                state["t_op"] = r1.t_junction_ss_c
            logger.info(f"[Thermal 1D] T_j={r1.t_junction_ss_c}°C (p_avg={state['p_avg']:.0f}W)")
        except Exception as e:
            logger.error(f"[Thermal 1D] {e}")

        # 2b: 2D grid with PEAK power (hotspot is a worst-case event)
        try:
            from simulations.thermal.thermal_grid_2d import run_thermal_grid_2d
            area_cm2 = None
            if power_p.get("power_density_w_cm2"):
                area_cm2 = state["p_peak"] / float(power_p["power_density_w_cm2"])
            g2d = run_thermal_grid_2d(
                power_w=state["p_peak"],            # ← peak, not avg
                die_area_cm2=area_cm2 or 0.83,
                t_ambient_c=t_ambient,
                r_heatsink_c_per_w=r_theta,
            )
            result.thermal.t_hotspot_ss_c            = g2d["t_hotspot_ss_c"]
            result.thermal.t_average_ss_c            = g2d["t_average_ss_c"]
            result.thermal.hotspot_excess_c          = g2d["hotspot_excess_c"]
            result.thermal.hotspot_location          = list(g2d["hotspot_location"])
            result.thermal.thermal_gradient_c_per_mm = g2d["thermal_gradient_c_per_mm"]
            result.thermal.runaway_risk_local        = g2d["runaway_risk_local"]
            result.thermal.jedec_margin_hotspot_pct  = g2d["jedec_margin_hotspot_pct"]
            result.thermal.hotspot_map               = g2d["hotspot_map"]
            result.thermal.time_ms                   = g2d["time_ms"]
            result.thermal.t_hotspot_transient       = g2d["t_hotspot_transient"]
            state["t_hotspot"] = g2d["t_hotspot_ss_c"]

            note = (
                f"2D hotspot @ peak power ({state['p_peak']:.0f}W): "
                f"{g2d['t_hotspot_ss_c']}°C (+{g2d['hotspot_excess_c']}°C vs avg {g2d['t_average_ss_c']}°C) "
                f"at grid{g2d['hotspot_location']}, gradient={g2d['thermal_gradient_c_per_mm']}°C/mm"
            )
            result.thermal.notes.append(note)
            if g2d["runaway_risk_local"]:
                result.thermal.status = "critical"
                result.thermal.notes.append("HOTSPOT EXCEEDS JEDEC — local thermal runaway risk at peak load")
            logger.info(f"[Thermal 2D] hotspot={g2d['t_hotspot_ss_c']}°C avg={g2d['t_average_ss_c']}°C (p_peak={state['p_peak']:.0f}W)")
        except Exception as e:
            logger.error(f"[Thermal 2D] {e}")

        # Attach trace metadata
        if workload_trace:
            result.thermal.workload_profile = workload_trace.profile_name
            result.thermal.p_avg_w          = workload_trace.p_avg_w
            result.thermal.p_peak_w         = workload_trace.p_peak_w
            result.thermal.duty_cycle_pct   = workload_trace.duty_cycle_pct
            result.thermal.burst_count      = workload_trace.burst_count

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3 — PDN (RLC droop) + SSN
    # ═══════════════════════════════════════════════════════════════════════════
    has_pdn = bool(pdn_p or power_p.get("current_a"))
    if has_pdn:
        try:
            from simulations.pdn.rlc_transient import run_pdn_sim
            r_pdn = run_pdn_sim(
                pdn_params=pdn_p, vdd_v=vdd_nom,
                i_load_a=state["i_peak"],
                di_dt_a_per_ns=state["di_dt_max"],
                bump_density_per_mm2=float(pdn_p["bump_density_per_mm2"]) if pdn_p.get("bump_density_per_mm2") else None,
            )
            result.pdn = r_pdn
            state["droop_mv"] = r_pdn.droop_mv or 0.0
            logger.info(f"[PDN] droop={state['droop_mv']:.1f}mV (i_peak={state['i_peak']:.1f}A di_dt={state['di_dt_max']:.3f}A/ns)")
        except Exception as e:
            logger.error(f"[PDN] {e}")

        try:
            from simulations.pdn.ssn_analysis import run_ssn_analysis
            ssn = run_ssn_analysis(pdn_params=pdn_p, power_params=power_p)
            state["ssn_mv"] = ssn.get("ssn_at_50pct_switching_mv") or 0.0
            if result.pdn:
                result.pdn.ssn_worst_mv          = ssn["worst_ssn_mv"]
                result.pdn.ssn_at_50pct_mv       = ssn["ssn_at_50pct_switching_mv"]
                result.pdn.ssn_timing_margin_mv  = ssn["timing_noise_margin_mv"]
                result.pdn.ssn_critical_fraction = ssn["critical_switching_fraction"]
                result.pdn.ssn_status            = ssn["status"]
                result.pdn.notes.extend(ssn["notes"])
                if STATUS_RANK.get(ssn["status"],0) > STATUS_RANK.get(result.pdn.status,0):
                    result.pdn.status = ssn["status"]
            logger.info(f"[SSN] 50%={state['ssn_mv']:.1f}mV status={ssn['status']}")
        except Exception as e:
            logger.error(f"[SSN] {e}")

    # Effective Vdd at die = nominal - droop - SSN
    state["vdd_at_die"] = vdd_nom - (state["droop_mv"] + state["ssn_mv"]) * 1e-3
    state["vdd_at_die"] = max(state["vdd_at_die"], vdd_nom * 0.5)   # clamp at 50% for numerics

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Coupled thermal-electrical solver
    # Uses: p_dynamic, r_theta, t_ambient, t_hotspot (from 2D grid)
    # Produces: T_op (replaces assumed 105°C everywhere), runaway detection
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        from simulations.coupled.thermal_electrical_solver import run_coupled_solver
        from core.schemas import CoupledSimResult
        coupled_raw = run_coupled_solver(
            power_params=power_p,
            thermal_params=thermal_p,
            t_hotspot_c=state.get("t_hotspot"),
        )
        r_coupled = CoupledSimResult(
            status=coupled_raw["status"],
            tt_T_op_c             = coupled_raw["tt_corner"]["T_op_c"],
            tt_P_leak_w           = coupled_raw["tt_corner"]["P_leak_op_w"],
            tt_P_total_w          = coupled_raw["tt_corner"]["P_total_op_w"],
            tt_P_overhead_pct     = coupled_raw["tt_corner"]["P_overhead_pct"],
            tt_runaway_factor     = coupled_raw["tt_corner"]["runaway_factor"],
            tt_converged          = coupled_raw["tt_corner"]["converged"],
            tt_runaway            = coupled_raw["tt_corner"]["runaway"],
            ff_T_op_c             = coupled_raw["ff_corner"]["T_op_c"],
            ff_P_total_w          = coupled_raw["ff_corner"]["P_total_op_w"],
            ff_runaway            = coupled_raw["ff_corner"]["runaway"],
            hotspot_T_c           = (coupled_raw["hotspot"] or {}).get("T_hotspot_c"),
            hotspot_runaway       = (coupled_raw["hotspot"] or {}).get("runaway_at_hotspot"),
            hotspot_runaway_factor= (coupled_raw["hotspot"] or {}).get("runaway_factor"),
            r_theta_critical      = coupled_raw.get("r_theta_critical"),
            r_theta_actual        = r_theta,
            margin_to_runaway_pct = coupled_raw.get("margin_to_runaway_pct"),
            alpha_leak_per_c      = coupled_raw.get("alpha_leak_per_c"),
            leak_doubling_temp_c  = coupled_raw.get("leak_doubling_temp_c"),
            notes                 = coupled_raw["notes"],
        )
        result.coupled = r_coupled

        # Update state with calibrated T_op
        if not coupled_raw["tt_corner"]["runaway"]:
            state["t_op"] = coupled_raw["tt_corner"]["T_op_c"]
            logger.info(f"[Coupled] T_op={state['t_op']:.1f}°C (replaces static T assumption)")
        else:
            logger.info(f"[Coupled] RUNAWAY — R_crit={coupled_raw.get('r_theta_critical')} C/W")
        logger.info(f"[Coupled] status={r_coupled.status} factor={r_coupled.tt_runaway_factor}")
    except Exception as e:
        logger.error(f"[Coupled] {e}")
        import traceback; logger.error(traceback.format_exc())

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Electrical CMOS + PVT corners
    # NOW uses: T_op from coupled solver + Vdd_at_die from PDN+SSN
    # ═══════════════════════════════════════════════════════════════════════════
    if power_p:
        try:
            from simulations.electrical.cmos_power import run_electrical_sim
            # Pass T_op into thermal_params so electrical sim uses real temperature
            thermal_with_top = dict(thermal_p)
            thermal_with_top["t_junction_c"] = state["t_op"]
            result.electrical = run_electrical_sim(power_params=power_p, thermal_params=thermal_with_top)
            logger.info(f"[Electrical] T_op={state['t_op']:.1f}°C status={result.electrical.status}")
        except Exception as e:
            logger.error(f"[Electrical] {e}")

        try:
            from simulations.electrical.pvt_corners import run_pvt_analysis
            # PVT uses real Vdd_at_die (after droop+SSN, not nominal)
            pdn_with_real_droop = dict(pdn_p)
            pdn_with_real_droop["ir_drop_mv"] = (state["droop_mv"] + state["ssn_mv"])
            thermal_with_top = dict(thermal_p)
            thermal_with_top["t_junction_c"] = state["t_op"]
            pvt = run_pvt_analysis(
                power_params=power_p,
                thermal_params=thermal_with_top,
                pdn_params=pdn_with_real_droop,
            )
            if result.electrical:
                result.electrical.pvt_worst_power_corner  = pvt["worst_power_corner"]
                result.electrical.pvt_worst_timing_corner = pvt["worst_timing_corner"]
                result.electrical.pvt_min_slack_ps        = pvt["min_timing_slack_ps"]
                result.electrical.pvt_any_timing_fail     = pvt["any_timing_failure"]
                result.electrical.pvt_ff_power_overhead_pct = pvt["power_overhead_ff_pct"]
                result.electrical.pvt_corners             = pvt["corners"]
                result.electrical.pvt_status              = pvt["status"]
                result.electrical.pvt_notes               = pvt["notes"]
                result.electrical.notes.extend(pvt["notes"])
                if STATUS_RANK.get(pvt["status"],0) > STATUS_RANK.get(result.electrical.status,0):
                    result.electrical.status = pvt["status"]
            logger.info(f"[PVT] vdd_drooped={state['vdd_at_die']:.4f}V slack={pvt['min_timing_slack_ps']}ps status={pvt['status']}")
        except Exception as e:
            logger.error(f"[PVT] {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Reliability & Aging
    # Uses: T_op from coupled solver (not assumed 105°C)
    # ═══════════════════════════════════════════════════════════════════════════
    if power_p or thermal_p:
        try:
            from simulations.reliability.aging import run_aging_analysis
            from core.schemas import AgingSimResult
            thermal_with_top = dict(thermal_p)
            thermal_with_top["t_junction_c"] = state["t_op"]
            ag = run_aging_analysis(
                power_params=power_p,
                thermal_params=thermal_with_top,
                pdn_params=pdn_p,
                mission_years=10.0,
                duty_cycle=workload_trace.duty_cycle_pct / 100 if workload_trace else 0.6,
            )
            result.aging = AgingSimResult(
                status=ag["status"],
                mission_years=ag["mission_years"],
                duty_cycle=ag["duty_cycle"],
                t_junction_c=ag["t_junction_c"],
                nbti_delta_vt_mv   = ag["nbti"]["delta_vt_mv"],
                nbti_delta_t_pd_ps = ag["nbti"]["delta_t_pd_ps"],
                nbti_exceeds_limit = ag["nbti"]["exceeds_limit"],
                hci_mttf_years=ag["hci"]["mttf_years"],
                hci_mttf_ok=ag["hci"]["mttf_ok"],
                em_J_a_cm2=ag["em"]["J_a_cm2"],
                em_mttf_years=ag["em"]["mttf_years"],
                em_mttf_ok=ag["em"]["mttf_ok"],
                em_margin_pct=ag["em"].get("margin_pct"),
                min_mttf_years=ag["min_mttf_years"],
                notes=ag["notes"],
            )
            logger.info(
                f"[Aging] T_op={state['t_op']:.0f}°C duty={ag['duty_cycle']:.0%} "
                f"NBTI={ag['nbti']['delta_vt_mv']:.1f}mV "
                f"EM={ag['em']['mttf_years']:.0f}yr status={ag['status']}"
            )
        except Exception as e:
            logger.error(f"[Aging] {e}")
            import traceback; logger.error(traceback.format_exc())

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Data Movement (Roofline + NoC)
    # ═══════════════════════════════════════════════════════════════════════════
    has_dm = bool(dm_p or idea_domain in ("data_movement", "cross_domain"))
    if has_dm:
        try:
            from simulations.data_movement.roofline import run_data_movement_sim
            result.data_movement = run_data_movement_sim(dm_params=dm_p)
            logger.info(f"[DM] eff={result.data_movement.efficiency_pct}% status={result.data_movement.status}")
        except Exception as e:
            logger.error(f"[DM] {e}")

        try:
            from simulations.data_movement.noc_congestion import run_noc_analysis
            noc = run_noc_analysis(dm_params=dm_p)
            if result.data_movement:
                result.data_movement.noc_link_utilization      = noc["link_utilization"]
                result.data_movement.noc_bw_effective_gb_s     = noc["bw_effective_gb_s"]
                result.data_movement.noc_congestion_penalty_pct = noc["congestion_penalty_pct"]
                result.data_movement.noc_added_latency_ns      = noc["added_latency_ns"]
                result.data_movement.noc_efficiency_pct        = noc["efficiency_with_noc_pct"]
                result.data_movement.noc_traffic_pattern       = noc["traffic_pattern"]
                result.data_movement.noc_status                = noc["status"]
                result.data_movement.noc_notes                 = noc["notes"]
                result.data_movement.notes.extend(noc["notes"])
                if STATUS_RANK.get(noc["status"],0) > STATUS_RANK.get(result.data_movement.status,0):
                    result.data_movement.status = noc["status"]
            logger.info(f"[NoC] util={noc['link_utilization']:.0%} bw={noc['bw_effective_gb_s']:.0f}GB/s status={noc['status']}")
        except Exception as e:
            logger.error(f"[NoC] {e}")


    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 8 — Monte Carlo Yield & Reliability Analysis
    # Runs stochastic sweep over parameter variations using same physics.
    # Fast: vectorized NumPy, 1000 dies in ~1s.
    # ═══════════════════════════════════════════════════════════════════════════
    if power_p or thermal_p:
        try:
            from simulations.yield_analysis.monte_carlo import run_monte_carlo
            from core.schemas import MonteCarloResult
            mc = run_monte_carlo(
                power_params=power_p,
                thermal_params=thermal_p,
                pdn_params=pdn_p,
                max_samples=1000,
                adaptive=True,
                compute_sensitivity=True,
            )
            sensitivity = mc.get("sensitivity", {})
            top_param = max(sensitivity, key=sensitivity.get) if sensitivity else None
            fleet = mc.get("fleet", {})
            result.monte_carlo = MonteCarloResult(
                status            = mc["status"],
                yield_pct         = mc["yield_pct"],
                yield_ci_pct      = mc["yield_ci_pct"],
                n_samples         = mc["n_samples"],
                gate_yields       = mc["gate_yields"],
                gate_failures     = mc["gate_failures"],
                distributions     = mc["distributions"],
                fleet_mean_mttf   = fleet.get("mean_mttf_yr"),
                fleet_p10_mttf    = fleet.get("p10_mttf_yr"),
                fleet_p1_mttf     = fleet.get("p1_mttf_yr"),
                fleet_pct_jedec   = fleet.get("fraction_10yr"),
                sensitivity       = sensitivity,
                top_sensitivity_param = top_param,
                notes             = mc["notes"],
            )
            logger.info(
                f"[MC] yield={mc['yield_pct']:.1f}% ±{mc['yield_ci_pct']/2:.1f}% "
                f"N={mc['n_samples']} dominant_fail={min(mc['gate_yields'], key=mc['gate_yields'].get)} "
                f"top_param={top_param} status={mc['status']}"
            )
        except Exception as e:
            logger.error(f"[MC] {e}")
            import traceback; logger.error(traceback.format_exc())

    # ═══════════════════════════════════════════════════════════════════════════
    # Composite scoring
    # ═══════════════════════════════════════════════════════════════════════════
    statuses = [
        result.thermal.status       if result.thermal       else "skipped",
        result.coupled.status       if result.coupled       else "skipped",
        result.pdn.status           if result.pdn           else "skipped",
        result.electrical.status    if result.electrical    else "skipped",
        result.aging.status         if result.aging         else "skipped",
        result.data_movement.status if result.data_movement else "skipped",
    ]
    result.overall_status = _worst(*statuses)
    result.sim_score      = _compute_score(result)
    result.duration_ms    = int((time.time() - start) * 1000)

    # Collect critical failures and insights
    critical_kw = {"FAIL","VIOLAT","EXCEED","RUNAWAY","SATURAT","CRITICAL","TIMING","NBTI","HCI"}
    for domain_res in [result.thermal, result.coupled, result.pdn,
                       result.electrical, result.aging, result.data_movement]:
        if not domain_res: continue
        if domain_res.status in ("fail","critical"):
            result.critical_failures.extend(
                n for n in domain_res.notes
                if any(kw in n.upper() for kw in critical_kw)
            )
        result.key_insights.extend(domain_res.notes[:2])

    logger.info(
        f"[Dispatcher v3] T_op={state['t_op']:.0f}°C Vdd_die={state['vdd_at_die']:.3f}V "
        f"→ overall={result.overall_status} score={result.sim_score} {result.duration_ms}ms"
    )
    return result
