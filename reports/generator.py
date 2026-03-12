"""
generator.py — Generates per-idea simulation report with matplotlib plots.

Output: markdown report + 4 PNG plots (one per domain).
Saved to: reports/output/<idea_id>/
"""
from __future__ import annotations
import os, math
from pathlib import Path
from typing import Optional
from core.schemas import SimResult

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


STATUS_COLOR = {
    "pass":     "#2ecc71",
    "warning":  "#f39c12",
    "critical": "#e74c3c",
    "fail":     "#8e44ad",
    "skipped":  "#95a5a6",
}
STATUS_EMOJI = {"pass": "✅", "warning": "⚠️", "critical": "🔴", "fail": "💀", "skipped": "⏭"}


def _plot_thermal(result, out_dir: Path) -> Optional[str]:
    if not HAS_MPL or not result.thermal or not result.thermal.time_s:
        return None
    r = result.thermal
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Thermal Simulation — {result.idea_title[:50]}", fontsize=11, fontweight="bold")

    # Plot 1: T(t) transient
    ax = axes[0]
    ax.plot(r.time_s, r.t_junction_transient, color="#e74c3c", linewidth=2, label="T_junction")
    ax.axhline(125, color="#e74c3c", linestyle="--", alpha=0.5, label="JEDEC 125°C")
    ax.axhline(r.t_junction_ss_c, color="#3498db", linestyle=":", alpha=0.7, label=f"SS={r.t_junction_ss_c:.1f}°C")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [°C]")
    ax.set_title("Junction Temperature Transient")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 2: Thermal budget bar
    ax2 = axes[1]
    categories = ["T_ambient", "ΔT (dynamic+leakage)", "JEDEC headroom"]
    t_amb = 25.0
    delta_t = (r.t_junction_ss_c or 25) - t_amb
    headroom = max(0, 125 - (r.t_junction_ss_c or 125))
    vals = [t_amb, delta_t, headroom]
    colors = ["#3498db", "#e67e22", "#2ecc71"]
    bars = ax2.barh(categories, vals, color=colors, edgecolor="white")
    ax2.set_xlabel("Temperature [°C]")
    ax2.set_title("Thermal Budget Breakdown")
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}°C", va="center", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = out_dir / "thermal.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return str(path)


def _plot_pdn(result, out_dir: Path) -> Optional[str]:
    if not HAS_MPL or not result.pdn:
        return None
    r = result.pdn
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"PDN Simulation — {result.idea_title[:50]}", fontsize=11, fontweight="bold")

    # Plot 1: Voltage transient
    ax = axes[0]
    if r.time_ns and r.voltage_transient:
        ax.plot(r.time_ns, r.voltage_transient, color="#3498db", linewidth=2)
        ax.axhline(r.v_nominal, color="green", linestyle="--", alpha=0.6, label=f"VDD={r.v_nominal}V")
        if r.v_min_transient:
            ax.axhline(r.v_min_transient, color="red", linestyle=":", alpha=0.7,
                       label=f"V_min={r.v_min_transient:.4f}V")
        if r.v_nominal:
            ax.axhline(r.v_nominal * 0.95, color="orange", linestyle="--", alpha=0.4, label="95% VDD")
    ax.set_xlabel("Time [ns]"); ax.set_ylabel("Voltage [V]")
    ax.set_title("PDN Transient Droop"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 2: Impedance profile
    ax2 = axes[1]
    if r.freq_hz and r.impedance_mohm:
        ax2.semilogx(r.freq_hz, r.impedance_mohm, color="#9b59b6", linewidth=2)
        ax2.axhline(10, color="red", linestyle="--", alpha=0.6, label="10mΩ target limit")
        if r.resonant_freq_mhz:
            ax2.axvline(r.resonant_freq_mhz * 1e6, color="orange", linestyle=":", alpha=0.7,
                        label=f"Resonance {r.resonant_freq_mhz}MHz")
    ax2.set_xlabel("Frequency [Hz]"); ax2.set_ylabel("Impedance [mΩ]")
    ax2.set_title("PDN Impedance Z(f)"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "pdn.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return str(path)


def _plot_roofline(result, out_dir: Path) -> Optional[str]:
    if not HAS_MPL or not result.data_movement:
        return None
    r = result.data_movement
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Roofline — {result.idea_title[:50]}", fontsize=11, fontweight="bold")

    # Draw roofline for each memory level
    ai_range = np.logspace(-2, 4, 500)
    peak_compute = r.peak_compute_tflops or 100.0
    bw_levels = {
        "L1 (20 TB/s)":  20000,
        "HBM3 (1.2 TB/s)": 1200,
        "DRAM (100 GB/s)": 100,
    }
    colors_levels = ["#2ecc71", "#3498db", "#e74c3c"]
    for (label, bw), color in zip(bw_levels.items(), colors_levels):
        mem_roof  = bw * ai_range / 1000  # TFLOPS = GB/s * FLOP/byte / 1000
        roof_line = np.minimum(peak_compute, mem_roof)
        ax.loglog(ai_range, roof_line, color=color, linewidth=1.5, alpha=0.7, label=label)

    # Mark compute roof
    ax.axhline(peak_compute, color="black", linestyle="--", alpha=0.5, label=f"Peak compute ({peak_compute} TFLOPS)")

    # Mark the idea's operating point
    ai = r.arithmetic_intensity_flop_per_byte or 1.0
    achiev = r.achievable_tflops or peak_compute * 0.5
    ax.scatter([ai], [achiev], s=200, color="#e74c3c", zorder=5,
               label=f"This idea (AI={ai:.1f}, {r.efficiency_pct:.0f}% util)")
    ax.axvline(ai, color="#e74c3c", linestyle=":", alpha=0.4)

    ax.set_xlabel("Arithmetic Intensity [FLOP/byte]")
    ax.set_ylabel("Achievable Throughput [TFLOPS]")
    ax.set_title("Roofline Model — Memory Hierarchy Analysis")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = out_dir / "roofline.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return str(path)


def generate_report(result: SimResult, output_base: str = "reports/output") -> str:
    """
    Generate markdown report + plots for one SimResult.
    Returns path to the markdown file.
    """
    out_dir = Path(output_base) / result.idea_id[:8]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    thermal_plot = _plot_thermal(result, out_dir)
    pdn_plot     = _plot_pdn(result, out_dir)
    _plot_roofline(result, out_dir)

    overall_emoji = STATUS_EMOJI.get(result.overall_status, "?")
    overall_color = STATUS_COLOR.get(result.overall_status, "#95a5a6")

    md_lines = [
        f"# Simulation Report: {result.idea_title}",
        f"",
        f"**Idea ID:** `{result.idea_id}`  ",
        f"**Domain:** `{result.idea_domain}`  ",
        f"**Sim Score:** `{result.sim_score:.1f}/10`  ",
        f"**Overall Status:** {overall_emoji} `{result.overall_status.upper()}`  ",
        f"**Timestamp:** {result.timestamp}  ",
        f"**Duration:** {result.duration_ms}ms",
        f"",
        f"---",
        f"",
    ]

    # Critical failures upfront
    if result.critical_failures:
        md_lines += ["## ⛔ Critical Failures", ""]
        for f in result.critical_failures:
            md_lines.append(f"- {f}")
        md_lines.append("")

    # Per-domain sections
    for domain_name, domain_res, plot_file in [
        ("🌡 Thermal",       result.thermal,       thermal_plot),
        ("⚡ PDN",            result.pdn,           pdn_plot),
        ("🔋 Electrical",    result.electrical,    None),
        ("📡 Data Movement", result.data_movement, None),
    ]:
        if not domain_res:
            continue
        emoji = STATUS_EMOJI.get(domain_res.status, "?")
        md_lines += [f"## {domain_name}  {emoji} `{domain_res.status.upper()}`", ""]

        # Key metrics
        if hasattr(domain_res, "t_junction_ss_c") and domain_res.t_junction_ss_c:
            md_lines.append(f"| T_junction SS | **{domain_res.t_junction_ss_c:.1f}°C** |")
            md_lines.append(f"| JEDEC margin | **{domain_res.thermal_margin_pct:.1f}%** |")
            md_lines.append(f"| Thermal runaway risk | **{'YES ⚠️' if domain_res.thermal_runaway_risk else 'no'}** |")
            md_lines.append(f"| Rise time (90%) | **{domain_res.t_rise_90pct_ms}ms** |")
        if hasattr(domain_res, "droop_mv") and domain_res.droop_mv is not None:
            md_lines.append(f"| IR drop | **{domain_res.ir_drop_mv:.1f}mV** |")
            md_lines.append(f"| Voltage droop | **{domain_res.droop_mv:.1f}mV ({domain_res.droop_pct:.1f}%)** |")
            md_lines.append(f"| PDN resonance | **{domain_res.resonant_freq_mhz}MHz** |")
            md_lines.append(f"| Recover time | **{domain_res.recovery_time_ns}ns** |")
        if hasattr(domain_res, "p_total_w") and domain_res.p_total_w:
            md_lines.append(f"| P_dynamic | **{domain_res.p_dynamic_w:.1f}W** |")
            md_lines.append(f"| P_leakage | **{domain_res.p_leakage_w:.1f}W** |")
            md_lines.append(f"| T_equilibrium | **{domain_res.t_equilibrium_c:.1f}°C** |")
            md_lines.append(f"| Converged | **{'yes' if domain_res.converged else 'NO - RUNAWAY'}** |")
        if hasattr(domain_res, "efficiency_pct") and domain_res.efficiency_pct is not None:
            md_lines.append(f"| Achievable TFLOPS | **{domain_res.achievable_tflops:.1f}** |")
            md_lines.append(f"| Efficiency | **{domain_res.efficiency_pct:.1f}%** |")
            md_lines.append(f"| Bottleneck | **{domain_res.bottleneck}** |")
            md_lines.append(f"| Binding memory | **{domain_res.binding_memory_level}** |")

        md_lines.append("")

        # Notes
        if domain_res.notes:
            md_lines.append("**Simulation notes:**")
            for note in domain_res.notes:
                md_lines.append(f"> {note}")
            md_lines.append("")

        # Plot
        if plot_file and Path(plot_file).exists():
            rel_path = Path(plot_file).name
            md_lines.append(f"![{domain_name} plot]({rel_path})")
            md_lines.append("")

    # Key insights
    if result.key_insights:
        md_lines += ["---", "## Key Simulation Insights", ""]
        for insight in result.key_insights[:8]:
            md_lines.append(f"- {insight}")
        md_lines.append("")

    md_content = "\n".join(md_lines)
    md_path = out_dir / "report.md"
    md_path.write_text(md_content, encoding="utf-8")
    result.report_path = str(md_path)
    return str(md_path)