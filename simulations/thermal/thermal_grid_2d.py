"""
thermal_grid_2d.py — 2D Finite Difference Thermal Simulation

Solves the 2D heat equation on an N×N die grid:
  ρCp * dT/dt = k * (∂²T/∂x² + ∂²T/∂y²) + Q(x,y,t)

Each cell has its own power density Q(x,y).
Adjacent cells exchange heat through thermal conductance.
Bottom boundary = heatsink (fixed via convection BC).

This reveals what the 1D model hides:
  - Hotspot location and peak temperature (always > average T_j)
  - Lateral thermal spreading resistance
  - Thermal crosstalk between compute and SRAM
  - Hotspot-induced local runaway even when average T is OK

Die layout for AI accelerators (10×10 grid):
  ┌──────────────────────────────┐
  │  IO  │  IO  │  IO  │  IO   │  Row 0 (IO ring: low power)
  │ SRAM │ CORE │ CORE │ SRAM  │  Row 1-2
  │ CORE │ CORE │ CORE │ CORE  │  Row 3-6 (compute cores: HIGH)
  │ SRAM │ CORE │ CORE │ SRAM  │  Row 7-8
  │  IO  │ CTRL │ CTRL │  IO   │  Row 9 (IO + controller)
  └──────────────────────────────┘
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ── Material properties ───────────────────────────────────────────────────────
SILICON = {
    "k_w_cm_k":   1.48,      # thermal conductivity [W/(cm·K)]
    "rho_g_cm3":  2.33,      # density [g/cm³]
    "cp_j_g_k":   0.712,     # specific heat [J/(g·K)]
    "alpha_cm2_s": 1.48 / (2.33 * 0.712),   # thermal diffusivity α = k/(ρCp)
}

JEDEC_LIMIT = 125.0   # °C

# ── AI Die power map (10×10 normalized, sum = 1.0) ───────────────────────────
# Models a typical AI accelerator: central compute cores, peripheral SRAM/IO

def _build_power_map(n: int = 10, hotspot_boost: float = 1.8) -> np.ndarray:
    """
    Build normalized power density map.
    Returns N×N array where sum = N² (normalized to average 1.0).
    hotspot_boost: peak-to-average power ratio at compute cores.
    """
    pmap = np.ones((n, n))

    # Compute core region (center 60% of die) — highest power density
    core_start = n // 5
    core_end   = 4 * n // 5
    pmap[core_start:core_end, core_start:core_end] = hotspot_boost

    # SRAM regions (edges of compute core) — ~50% of compute density
    sram_margin = max(1, n // 10)
    pmap[core_start:core_end, :core_start]  = hotspot_boost * 0.45
    pmap[core_start:core_end, core_end:]    = hotspot_boost * 0.45
    pmap[:core_start, core_start:core_end]  = hotspot_boost * 0.45
    pmap[core_end:,  core_start:core_end]   = hotspot_boost * 0.45

    # IO ring (perimeter) — ~15% of compute density
    pmap[0, :]  = hotspot_boost * 0.15
    pmap[-1, :] = hotspot_boost * 0.15
    pmap[:, 0]  = hotspot_boost * 0.15
    pmap[:, -1] = hotspot_boost * 0.15

    # Normalize to average = 1.0
    pmap = pmap / pmap.mean()
    return pmap


def _build_2d_laplacian(n: int) -> "scipy.sparse.csr_matrix":
    """
    Build N²×N² sparse Laplacian for 2D finite difference.
    Uses Kronecker product of 1D second-difference operator.
    """
    from scipy.sparse import diags as sp_diags, kron as sp_kron, eye as sp_eye

    # 1D second difference operator
    d1 = sp_diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format="csr", dtype=float)

    # 2D Laplacian = D⊗I + I⊗D
    I  = sp_eye(n, format="csr", dtype=float)
    L  = sp_kron(d1, I) + sp_kron(I, d1)
    return L


def run_thermal_grid_2d(
    power_w: float,
    die_area_cm2: float = 0.83,   # ~830 mm² (H100-class)
    die_thickness_um: float = 780,
    t_ambient_c: float = 25.0,
    r_heatsink_c_per_w: float = 0.10,
    thermal_params: Optional[dict] = None,
    grid_n: int = 10,
    hotspot_boost: float = 1.8,
    t_sim_ms: float = 100.0,
) -> Dict:
    """
    Run 2D transient thermal simulation on N×N die grid.

    Returns dict with:
      - t_hotspot_ss_c:    steady-state hotspot temperature
      - t_average_ss_c:    steady-state average die temperature
      - hotspot_excess_c:  hotspot - average (the hidden danger)
      - hotspot_location:  (row, col) index of peak temperature
      - hotspot_map:       N×N steady-state temperature map
      - thermal_gradient_c_per_mm: max lateral gradient
      - runaway_risk_local: True if any cell exceeds JEDEC
      - time_ms, t_hotspot_t: transient at hotspot cell
    """
    mat = SILICON
    thickness_cm = die_thickness_um * 1e-4   # µm → cm
    dx = np.sqrt(die_area_cm2) / grid_n      # cell size [cm]
    cell_volume = dx * dx * thickness_cm     # cm³

    # Power density per cell [W/cm³]
    power_map = _build_power_map(grid_n, hotspot_boost)
    q_avg_w_cm3 = power_w / (die_area_cm2 * thickness_cm)
    Q_map = power_map * q_avg_w_cm3           # W/cm³ per cell

    # Thermal conductance between adjacent cells [W/K]
    # G = k * A / dx  where A = dx * thickness
    G_lateral = mat["k_w_cm_k"] * dx * thickness_cm / dx   # = k * thickness
    # Bottom boundary convection to heatsink
    h_bottom = 1.0 / (r_heatsink_c_per_w * die_area_cm2)  # W/(cm²·K)
    G_bottom  = h_bottom * dx * dx   # W/K per cell

    # Thermal capacitance per cell [J/K]
    C_cell = mat["rho_g_cm3"] * mat["cp_j_g_k"] * cell_volume

    N2 = grid_n * grid_n

    # ── Sparse Laplacian ──────────────────────────────────────────────────────
    L = _build_2d_laplacian(grid_n)   # N²×N²

    # ODE: C * dT/dt = G_lat * L * T - G_bottom * (T - T_amb) + Q
    # Rearranged: dT/dt = (G_lat/C) * L * T - (G_bottom/C) * (T - T_amb) + Q/C

    alpha_lat   = G_lateral / C_cell
    alpha_bot   = G_bottom  / C_cell
    q_per_cell  = Q_map.flatten() * cell_volume  # W per cell
    q_rate      = q_per_cell / C_cell            # K/s per cell

    # Convert Laplacian units: L is dimensionless (1/dx² implicit)
    # We need to scale: G_lateral * L / dx² → but L already has the 1/dx structure
    # Corrected: the finite difference gives k/dx² factor
    k_over_dx2  = mat["k_w_cm_k"] * thickness_cm / (dx * dx * C_cell)

    def ode_rhs(t, T_flat):
        return (k_over_dx2 * L.dot(T_flat)
                - alpha_bot * (T_flat - t_ambient_c)
                + q_rate)

    # Solve transient ODE
    T0 = np.ones(N2) * t_ambient_c
    t_end = t_sim_ms * 1e-3   # ms → s

    sol = solve_ivp(
        ode_rhs,
        (0, t_end),
        T0,
        method="RK23",          # less stiff than 1D model (lateral diffusion moderates it)
        max_step=t_end / 200,
        rtol=1e-3,
        atol=1e-4,
    )

    T_final = sol.y[:, -1].reshape(grid_n, grid_n)
    T_hotspot_ss = float(T_final.max())
    T_avg_ss     = float(T_final.mean())

    # Hotspot location
    hot_idx = np.unravel_index(T_final.argmax(), T_final.shape)

    # Thermal gradient: max |ΔT| between adjacent cells / dx [°C/mm]
    dx_mm = dx * 10  # cm → mm
    grad_x = np.abs(np.diff(T_final, axis=1)) / dx_mm
    grad_y = np.abs(np.diff(T_final, axis=0)) / dx_mm
    max_gradient = float(max(grad_x.max(), grad_y.max()))

    # Transient at hotspot
    hot_idx_flat = hot_idx[0] * grid_n + hot_idx[1]
    t_hotspot_ms = [round(float(x)*1000, 3) for x in sol.t[::max(1, len(sol.t)//200)]]
    T_hot_transient = [round(float(x), 2) for x in sol.y[hot_idx_flat][::max(1, len(sol.t)//200)]]

    return {
        "t_hotspot_ss_c":         round(T_hotspot_ss, 2),
        "t_average_ss_c":         round(T_avg_ss, 2),
        "hotspot_excess_c":       round(T_hotspot_ss - T_avg_ss, 2),
        "hotspot_location":       (int(hot_idx[0]), int(hot_idx[1])),
        "hotspot_map":            T_final.tolist(),
        "thermal_gradient_c_per_mm": round(max_gradient, 2),
        "runaway_risk_local":     bool(T_hotspot_ss > JEDEC_LIMIT),
        "jedec_margin_hotspot_pct": round((JEDEC_LIMIT - T_hotspot_ss) / JEDEC_LIMIT * 100, 1),
        "time_ms":                t_hotspot_ms,
        "t_hotspot_transient":    T_hot_transient,
        "grid_n":                 grid_n,
        "hotspot_boost_factor":   hotspot_boost,
    }
