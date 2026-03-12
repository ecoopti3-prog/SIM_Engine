# Physics Simulation Engine — v1.0

**Companion to RD Engine v3. Takes Diamond ideas. Runs real simulations. Returns numbers.**

---

## What This Does

The RD Engine's Physics Gate answers: *"Does this idea violate a known physical law?"*  
This engine answers: *"Given the extracted parameters — what actually happens, numerically?"*

| RD Engine Physics Gate | This Engine |
|------------------------|-------------|
| `T_j > JEDEC limit? → KILL` | `T_j(t) transient curve, rise time, runaway detection` |
| `IR drop > 5% VDD? → KILL` | `V(t) droop waveform, Z(f) impedance profile, resonance` |
| `E/op < Landauer? → KILL`  | `P_dynamic + P_leakage(T) breakdown, thermal overhead %` |
| `BW > 4×HBM3? → KILL`     | `Roofline per memory level, binding bottleneck, efficiency %` |

---

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run demo (no Supabase needed)
python main.py --demo

# Run tests
python tests/test_simulations.py

# Simulate all Diamond ideas from Supabase
cp .env.example .env   # fill in SUPABASE_URL + SUPABASE_KEY
python main.py --diamonds

# Add sim_results table to Supabase (run once)
python main.py --schema
```

---

## Architecture

```
Diamond Ideas (Supabase)
        ↓
  core/dispatcher.py        ← reads extracted params, routes to simulations
        ↓
  ┌──────────────────────────────────────────────────────┐
  │  simulations/                                        │
  │   thermal/rc_network.py      ← 5-node ODE (Radau)   │
  │   pdn/rlc_transient.py       ← RLC + impedance Z(f) │
  │   electrical/cmos_power.py   ← CMOS power breakdown  │
  │   data_movement/roofline.py  ← full memory hierarchy │
  └──────────────────────────────────────────────────────┘
        ↓
  core/schemas.py (SimResult)
        ↓
  reports/generator.py      ← markdown + matplotlib plots
        ↓
  db/reader.py              ← writes SimResult to sim_results table
```

---

## Simulations

### 🌡 Thermal — RC Network Transient ODE
- **Model**: 3-node RC ladder (die → IHS → heatsink)
- **Solver**: `scipy.integrate.solve_ivp` with Radau (stiff ODE)
- **Outputs**: T_junction(t), steady-state T, thermal margin, 90% rise time, runaway risk
- **Key insight**: `dP_leakage/dT × R_theta ≥ 1` → thermal runaway (no stable operating point)

### ⚡ PDN — Transient Droop + Impedance
- **Model**: DC IR drop + L×di/dt inductive droop + Z(f) ladder
- **Outputs**: V(t) droop waveform, Z(f) bode plot, resonant frequency, electromigration check
- **Key insight**: Resonance in Z(f) above 10MHz at >50mΩ → voltage noise will cause timing violations

### 🔋 Electrical — CMOS Power Breakdown
- **Model**: P_dynamic + P_leakage(T) = P_total, with leakage exponent calibrated to modern chips
- **Outputs**: P_dynamic vs P_leakage split, Landauer gap ratio, power density vs limits
- **Key insight**: At Tj=125°C, leakage typically reaches 30-40% of total — thermal overhead dominates

### 📡 Data Movement — Full Roofline
- **Model**: L1→L2→L3→HBM3→DRAM hierarchy with per-level BW and latency
- **Outputs**: Arithmetic intensity, ridge point, achievable TFLOPS, binding memory level, efficiency %
- **Key insight**: Most AI workloads are HBM-bound (556 GB/s), not compute-bound

---

## Sim Score

```
0-10 per domain → weighted composite

Critical failure (thermal runaway, EM violation) → 0
Critical (>15% droop, runaway risk)              → 2
Warning (tight margins, >5% droop)               → 5
Pass with margin                                  → 8-10
```

---

## Connecting to Supabase (RD Engine)

Uses the same Supabase database as RD Engine. Same keys, no migration except:

```sql
-- Run once in Supabase SQL editor
-- (python main.py --schema prints this)
CREATE TABLE IF NOT EXISTS sim_results (
    sim_id UUID PRIMARY KEY,
    idea_id UUID REFERENCES ideas(id),
    overall_status TEXT,
    sim_score FLOAT,
    thermal_result JSONB,
    pdn_result JSONB,
    electrical_result JSONB,
    dm_result JSONB,
    ...
);
```

---

## File Structure

```
sim_engine/
├── main.py                              # entry: --demo | --diamonds | --idea | --schema
├── config/settings.py
├── core/
│   ├── schemas.py                       # SimResult, ThermalSimResult, PDNSimResult...
│   └── dispatcher.py                    # routes idea → simulations → composite verdict
├── simulations/
│   ├── thermal/rc_network.py            # ODE: dT/dt = (P - ΔT/R) / C
│   ├── pdn/rlc_transient.py             # V(t) droop + Z(f) impedance
│   ├── electrical/cmos_power.py         # CMOS P breakdown + Landauer check
│   └── data_movement/roofline.py        # Full roofline with memory hierarchy
├── db/reader.py                         # read Diamond ideas + write SimResult
├── reports/generator.py                 # markdown + matplotlib plots
└── tests/test_simulations.py            # 28 unit tests
```
