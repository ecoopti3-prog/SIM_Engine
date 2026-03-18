"""
schemas.py — Pydantic schemas for sim_engine v2.
Full PVT + 2D thermal + workload traces + NoC + SSN.
"""
from __future__ import annotations
from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field
import uuid

SimTier   = Literal["tier1_analytical", "tier2_numerical", "skipped"]
SimStatus = Literal["pass", "warning", "critical", "fail", "skipped", "marginal", "insufficient_data"]


class ThermalSimResult(BaseModel):
    tier: SimTier = "tier1_analytical"
    status: SimStatus = "skipped"
    # 1D RC (fast path)
    t_junction_ss_c: Optional[float] = None
    thermal_margin_pct: Optional[float] = None
    dominant_resistance: Optional[str] = None
    t_rise_90pct_ms: Optional[float] = None
    t_peak_transient_c: Optional[float] = None
    thermal_runaway_risk: bool = False
    p_dynamic_w: Optional[float] = None
    p_leakage_w: Optional[float] = None
    time_s: Optional[List[float]] = None
    t_junction_transient: Optional[List[float]] = None
    # 2D Grid results
    t_hotspot_ss_c: Optional[float] = None
    t_average_ss_c: Optional[float] = None
    hotspot_excess_c: Optional[float] = None
    hotspot_location: Optional[List[int]] = None
    thermal_gradient_c_per_mm: Optional[float] = None
    runaway_risk_local: bool = False
    jedec_margin_hotspot_pct: Optional[float] = None
    hotspot_map: Optional[List[List[float]]] = None   # N×N grid
    time_ms: Optional[List[float]] = None
    t_hotspot_transient: Optional[List[float]] = None
    # Workload trace
    workload_profile: Optional[str] = None
    p_avg_w: Optional[float] = None
    p_peak_w: Optional[float] = None
    duty_cycle_pct: Optional[float] = None
    burst_count: Optional[int] = None
    plot_path: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class PDNSimResult(BaseModel):
    tier: SimTier = "tier1_analytical"
    status: SimStatus = "skipped"
    v_nominal: Optional[float] = None
    ir_drop_mv: Optional[float] = None
    ir_drop_pct: Optional[float] = None
    v_min_transient: Optional[float] = None
    droop_mv: Optional[float] = None
    droop_pct: Optional[float] = None
    recovery_time_ns: Optional[float] = None
    z_dc_mohm: Optional[float] = None
    z_at_1ghz_mohm: Optional[float] = None
    resonant_freq_mhz: Optional[float] = None
    anti_resonance_risk: bool = False
    em_margin_pct: Optional[float] = None
    em_flag: bool = False
    time_ns: Optional[List[float]] = None
    voltage_transient: Optional[List[float]] = None
    freq_hz: Optional[List[float]] = None
    impedance_mohm: Optional[List[float]] = None
    # SSN results
    ssn_worst_mv: Optional[float] = None
    ssn_at_50pct_mv: Optional[float] = None
    ssn_timing_margin_mv: Optional[float] = None
    ssn_critical_fraction: Optional[float] = None
    ssn_status: Optional[str] = None
    plot_path: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class ElectricalSimResult(BaseModel):
    tier: SimTier = "tier1_analytical"
    status: SimStatus = "skipped"
    # TT corner (base)
    p_dynamic_w: Optional[float] = None
    p_leakage_w: Optional[float] = None
    p_total_w: Optional[float] = None
    power_density_w_cm2: Optional[float] = None
    t_equilibrium_c: Optional[float] = None
    converged: Optional[bool] = None
    iterations_to_converge: Optional[int] = None
    energy_per_op_pj: Optional[float] = None
    landauer_ratio: Optional[float] = None
    useful_power_fraction: Optional[float] = None
    # PVT corners
    pvt_worst_power_corner: Optional[str] = None
    pvt_worst_timing_corner: Optional[str] = None
    pvt_min_slack_ps: Optional[float] = None
    pvt_any_timing_fail: Optional[bool] = None
    pvt_ff_power_overhead_pct: Optional[float] = None
    pvt_corners: Optional[Dict[str, Any]] = None
    pvt_status: Optional[str] = None
    pvt_notes: Optional[List[str]] = None
    plot_path: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class DataMovementSimResult(BaseModel):
    tier: SimTier = "tier1_analytical"
    status: SimStatus = "skipped"
    # Roofline
    arithmetic_intensity_flop_per_byte: Optional[float] = None
    ridge_point_flop_per_byte: Optional[float] = None
    bottleneck: Optional[str] = None
    achievable_tflops: Optional[float] = None
    peak_compute_tflops: Optional[float] = None
    efficiency_pct: Optional[float] = None
    binding_memory_level: Optional[str] = None
    bandwidth_utilization: Optional[Dict[str, float]] = None
    effective_latency_ns: Optional[float] = None
    latency_bound_pct: Optional[float] = None
    signal_integrity_ok: Optional[bool] = None
    max_practical_distance_mm: Optional[float] = None
    freq_hz: Optional[List[float]] = None
    impedance_mohm: Optional[List[float]] = None
    # NoC congestion
    noc_link_utilization: Optional[float] = None
    noc_bw_effective_gb_s: Optional[float] = None
    noc_congestion_penalty_pct: Optional[float] = None
    noc_added_latency_ns: Optional[float] = None
    noc_efficiency_pct: Optional[float] = None
    noc_traffic_pattern: Optional[str] = None
    noc_status: Optional[str] = None
    noc_notes: Optional[List[str]] = None
    plot_path: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class SimResult(BaseModel):
    sim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    idea_id: str
    idea_title: str
    idea_domain: str
    thermal:       Optional[ThermalSimResult]       = None
    coupled:       Optional["CoupledSimResult"]      = None
    aging:         Optional["AgingSimResult"]         = None
    monte_carlo:   Optional["MonteCarloResult"]      = None
    pdn:           Optional[PDNSimResult]           = None
    electrical:    Optional[ElectricalSimResult]    = None
    data_movement: Optional[DataMovementSimResult]  = None
    mechanical:     Optional["MechanicalSimResult"]       = None
    fluid:          Optional["FluidSimResult"]             = None
    electromechanical: Optional["ElectromechanicalSimResult"] = None
    overall_status: SimStatus = "skipped"
    sim_score: float = Field(default=0.0, ge=0.0, le=10.0)
    critical_failures: List[str] = Field(default_factory=list)
    warnings: List[str]          = Field(default_factory=list)
    key_insights: List[str]      = Field(default_factory=list)
    duration_ms: Optional[int]   = None
    report_path: Optional[str]   = None
    timestamp: Optional[str]     = None


class CoupledSimResult(BaseModel):
    """Output of the coupled thermal-electrical Newton solver."""
    status: SimStatus = "skipped"
    # TT corner
    tt_T_op_c: Optional[float] = None
    tt_P_leak_w: Optional[float] = None
    tt_P_total_w: Optional[float] = None
    tt_P_overhead_pct: Optional[float] = None
    tt_runaway_factor: Optional[float] = None
    tt_converged: Optional[bool] = None
    tt_runaway: Optional[bool] = None
    # FF corner
    ff_T_op_c: Optional[float] = None
    ff_P_total_w: Optional[float] = None
    ff_runaway: Optional[bool] = None
    # Hotspot
    hotspot_T_c: Optional[float] = None
    hotspot_runaway: Optional[bool] = None
    hotspot_runaway_factor: Optional[float] = None
    # Margins
    r_theta_critical: Optional[float] = None
    r_theta_actual: Optional[float] = None
    margin_to_runaway_pct: Optional[float] = None
    alpha_leak_per_c: Optional[float] = None
    leak_doubling_temp_c: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class AgingSimResult(BaseModel):
    """NBTI + HCI + Electromigration lifetime analysis."""
    status: SimStatus = "skipped"
    mission_years: Optional[float] = None
    duty_cycle: Optional[float] = None
    t_junction_c: Optional[float] = None
    # NBTI
    nbti_delta_vt_mv: Optional[float] = None
    nbti_delta_t_pd_ps: Optional[float] = None
    nbti_exceeds_limit: Optional[bool] = None
    # HCI
    hci_mttf_years: Optional[float] = None
    hci_mttf_ok: Optional[bool] = None
    # EM
    em_J_a_cm2: Optional[float] = None
    em_mttf_years: Optional[float] = None
    em_mttf_ok: Optional[bool] = None
    em_margin_pct: Optional[float] = None
    min_mttf_years: Optional[float] = None
    notes: List[str] = Field(default_factory=list)

class MonteCarloResult(BaseModel):
    status: SimStatus = "skipped"
    yield_pct:        Optional[float] = None
    yield_ci_pct:     Optional[float] = None
    n_samples:        Optional[int]   = None
    gate_yields:      Dict[str, float] = Field(default_factory=dict)
    gate_failures:    Dict[str, int]   = Field(default_factory=dict)
    distributions:    Dict[str, Any]   = Field(default_factory=dict)
    fleet_mean_mttf:  Optional[float] = None
    fleet_p10_mttf:   Optional[float] = None
    fleet_p1_mttf:    Optional[float] = None
    fleet_pct_jedec:  Optional[float] = None
    sensitivity:      Dict[str, float] = Field(default_factory=dict)
    top_sensitivity_param: Optional[str] = None
    notes: List[str] = Field(default_factory=list)



# ── v7: SimReport and DomainSimResult — used by sim_router ───────────────────
# These were missing from v5/v6 schemas. Added for rd_bridge compatibility.

class FalsificationBoundary(BaseModel):
    """The exact parameter value at which the idea transitions from fail to pass."""
    parameter: str = ""           # human-readable parameter name (optional - defaults to breaking_param)
    fail_value: Optional[float] = None
    pass_value: Optional[float] = None
    units: Optional[str] = None
    description: str = ""
    # Simulator fields (used by rc_network, rlc_network, roofline, power_model)
    domain: Optional[str] = None
    breaking_param: Optional[str] = None
    fix_vector: Optional[str] = None          # human-readable fix recommendation
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    margin_pct: Optional[float] = None
    nominal_value: Optional[float] = None     # the idea's proposed value
    breaking_value: Optional[float] = None    # the value at which it breaks
    safety_margin_pct: Optional[float] = None # how much margin is left

class ParameterSweepPoint(BaseModel):
    # Schema-facing fields (used by sim_router for JSON output)
    parameter_value: float = 0.0
    result_value: float = 0.0
    status: str = "pass"
    # Simulator-facing fields (used by rc_network, rlc_network, roofline, power_model)
    param_name: Optional[str] = None
    param_value: Optional[float] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None

class DomainSimResult(BaseModel):
    """Result for a single simulation domain (thermal/pdn/electrical/data_movement)."""
    idea_id: Optional[str] = None             # set by simulators for traceability
    domain: str
    fidelity: str = "L1_analytical"           # e.g. L1_analytical, L2_numerical
    solver_notes: str = ""                    # short note on which solver path ran
    status: SimStatus = "skipped"
    score: float = Field(default=0.0, ge=0.0, le=10.0)
    critical_failures: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)   # key numerical outputs
    falsification: Optional[FalsificationBoundary] = None   # simulator-facing alias
    falsification_boundary: Optional[FalsificationBoundary] = None  # schema-facing alias
    sweep_points: List[ParameterSweepPoint] = Field(default_factory=list)   # simulator-facing alias
    parameter_sweeps: List[ParameterSweepPoint] = Field(default_factory=list)  # schema-facing alias
    details: Dict[str, Any] = Field(default_factory=dict)   # raw sim output (legacy)

class SimReport(BaseModel):
    """Full simulation report for one idea — output of sim_router.route_idea()."""
    sim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    idea_id: str
    idea_title: str = ""
    idea_domain: str = ""
    domain_results: List[DomainSimResult] = Field(default_factory=list)
    overall_status: SimStatus = "skipped"
    overall_score: float = Field(default=0.0, ge=0.0, le=10.0)
    recommendation: str = ""   # proceed_to_prototype / kill_physics / revise / marginal
    revision_targets: List[str] = Field(default_factory=list)  # text targets (legacy)
    cross_domain_couplings: List[str] = Field(default_factory=list)
    fidelity_used: Dict[str, str] = Field(default_factory=dict)  # domain → fidelity level
    duration_ms: Optional[int] = None
    timestamp: Optional[str] = None
    total_sim_time_ms: Optional[int] = None   # alias for duration_ms set by sim_router

class BatchSimResult(BaseModel):
    """Batch result from running sim on multiple ideas."""
    timestamp: str
    ideas_submitted: int
    ideas_completed: int
    reports: List[SimReport]
    summary: Dict[str, Any] = Field(default_factory=dict)


class MechanicalSimResult(BaseModel):
    """Fatigue life + bearing reliability + vibration for robotics domains."""
    status: SimStatus = "skipped"
    # Fatigue
    miner_damage: Optional[float] = None
    predicted_life_h: Optional[float] = None
    stress_amplitude_mpa: Optional[float] = None
    endurance_limit_mpa: Optional[float] = None
    paris_life_cycles: Optional[float] = None
    # Bearing
    bearing_l10_h: Optional[float] = None
    bearing_b10_h: Optional[float] = None
    bearing_b1_h: Optional[float] = None
    bearing_reliability_pct: Optional[float] = None
    fleet_survival_pct: Optional[float] = None
    # Vibration
    vibration_amplification: Optional[float] = None
    frequency_ratio_r: Optional[float] = None
    in_resonance_band: Optional[bool] = None
    material: Optional[str] = None
    mission_hours: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class FluidSimResult(BaseModel):
    """Cold plate 2D + cavitation + corrosion for liquid cooling domains."""
    status: SimStatus = "skipped"
    # Cold plate
    t_outlet_c: Optional[float] = None
    t_hotspot_c: Optional[float] = None
    t_inlet_c: Optional[float] = None
    delta_t_c: Optional[float] = None
    t_gradient_c_m: Optional[float] = None
    reynolds_number: Optional[float] = None
    turbulent_flow: Optional[bool] = None
    # Cavitation
    npsh_available_m: Optional[float] = None
    npsh_margin_m: Optional[float] = None
    cavitation_number: Optional[float] = None
    transient_cavitation: Optional[bool] = None
    # Corrosion
    emf_diff_v: Optional[float] = None
    corrosion_rate_mm_yr: Optional[float] = None
    wall_loss_mm: Optional[float] = None
    anode_metal: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class ElectromechanicalSimResult(BaseModel):
    """Joule heating transient + contact degradation + motor thermal."""
    status: SimStatus = "skipped"
    # Joule heating
    wire_temp_ss_c: Optional[float] = None
    wire_temp_peak_c: Optional[float] = None
    wire_temp_limit_c: Optional[float] = None
    wire_margin_c: Optional[float] = None
    time_to_90pct_s: Optional[float] = None
    # Contact degradation
    contact_r_initial_mohm: Optional[float] = None
    contact_r_final_mohm: Optional[float] = None
    contact_temp_c: Optional[float] = None
    fretting_cycles: Optional[float] = None
    # Motor thermal
    motor_winding_ss_c: Optional[float] = None
    motor_winding_peak_c: Optional[float] = None
    motor_derating_pct: Optional[float] = None
    motor_available_power_w: Optional[float] = None
    mission_hours: Optional[float] = None
    notes: List[str] = Field(default_factory=list)
