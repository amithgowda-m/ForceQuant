"""Stage 7: figures (300 DPI PNG).

fig1_mas_radar        - MAS radar chart per company
fig2_lerner_trends    - Lerner proxy 3-year trend per company
fig3_cloudmetrics_breakeven - CloudMetrics break-even chart
fig4_cloudmetrics_npv       - CloudMetrics NPV sensitivity chart

Usage:  python -m src.figures
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.utils import load_config, root_path
from src import models

FORCES_SHORT = {
    "Competitive_Rivalry": "Rivalry",
    "Threat_of_New_Entry": "New Entry",
    "Threat_of_Substitutes": "Substitutes",
    "Buyer_Power": "Buyer Pwr",
    "Supplier_Power": "Supplier Pwr",
    "Complementors": "Complementors",
}

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


def save(fig, path: Path, dpi: int = 300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Fig 1: MAS radar chart
# ---------------------------------------------------------------------------
def fig1_mas_radar(forces_summary: pd.DataFrame, cfg: dict, out_dir: Path, dpi: int):
    forces = cfg["forces"]
    tickers = sorted(forces_summary["Ticker"].unique())
    labels = [FORCES_SHORT.get(f, f) for f in forces]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], size=8)
    ax.set_title("Porter's Forces S_i by Company (1=weak, 10=strong threat)", size=12, pad=20)

    for i, t in enumerate(tickers):
        sub = forces_summary[forces_summary["Ticker"] == t].set_index("Force")
        vals = [float(sub.loc[f, "S_i"]) if f in sub.index else 5.0 for f in forces]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, color=COLORS[i % len(COLORS)], label=t)
        ax.fill(angles, vals, alpha=0.08, color=COLORS[i % len(COLORS)])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    save(fig, out_dir / "fig1_mas_radar.png", dpi)


# ---------------------------------------------------------------------------
# Fig 2: Lerner proxy trends
# ---------------------------------------------------------------------------
def fig2_lerner_trends(fm: pd.DataFrame, cfg: dict, out_dir: Path, dpi: int):
    tickers = sorted(fm["Ticker"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, t in enumerate(tickers):
        sub = fm[fm["Ticker"] == t].copy()
        sub["lp"] = pd.to_numeric(sub["Lerner_proxy"], errors="coerce")
        sub = sub.dropna(subset=["lp"]).sort_values("FiscalYear")
        if sub.empty:
            continue
        ax.plot(sub["FiscalYear"].astype(str), sub["lp"], marker="o",
                label=t, color=COLORS[i % len(COLORS)], linewidth=2)
    ax.axhline(y=0.72, linestyle="--", color="gray", linewidth=1,
               label="Damodaran SW benchmark 71.7%")
    ax.set_title("Lerner Proxy (Gross Margin) Trend by Company", fontsize=13)
    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Gross Margin (Lerner proxy)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    save(fig, out_dir / "fig2_lerner_trends.png", dpi)


# ---------------------------------------------------------------------------
# Fig 3: CloudMetrics break-even chart
# ---------------------------------------------------------------------------
def fig3_cloudmetrics_breakeven(out_dir: Path, dpi: int):
    P, VC = 40_000, 8_000
    FC = models.fixed_costs(fc_ops=1_200_000, i_rd=6_000_000, n_life=5)
    Q_BE = models.breakeven_quantity(FC, P, VC)

    Q = np.linspace(0, 150, 300)
    TR = models.total_revenue(P, Q)
    TC = FC + VC * Q

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(Q, TR / 1e6, label="Total Revenue", color="#2196F3", linewidth=2)
    ax.plot(Q, TC / 1e6, label="Total Cost", color="#FF5722", linewidth=2)
    ax.axvline(Q_BE, linestyle="--", color="gray", linewidth=1.5, label=f"Break-even Q={Q_BE:.0f}")
    ax.fill_between(Q, TC / 1e6, TR / 1e6,
                    where=(Q >= Q_BE), alpha=0.12, color="green", label="Profit zone")
    ax.fill_between(Q, TC / 1e6, TR / 1e6,
                    where=(Q < Q_BE), alpha=0.12, color="red", label="Loss zone")
    ax.set_title("CloudMetrics (Hypothetical): Break-even Analysis\n"
                 f"P=${P:,}  VC=${VC:,}  FC=${FC:,.0f}  Q_BE={Q_BE:.0f}", fontsize=11)
    ax.set_xlabel("Quantity (customers)")
    ax.set_ylabel("Amount ($M)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out_dir / "fig3_cloudmetrics_breakeven.png", dpi)


# ---------------------------------------------------------------------------
# Fig 4: CloudMetrics NPV sensitivity
# ---------------------------------------------------------------------------
def fig4_cloudmetrics_npv(out_dir: Path, dpi: int):
    SC = models.switching_cost(c_mig=25_000, n_emp=50, h_train=20, w_emp=35,
                               c_int=15_000, rho=0.05, c_down=300_000)
    P_new = 40_000
    R, T = 0.10, 4
    # Sensitivity: vary P_inc from 40k to 70k
    p_inc_range = np.linspace(40_000, 70_000, 100)
    base_npvs = [models.npv(p_inc=pi, p_new=P_new, d_e=10_000, r=R, t=T, sc=SC)
                 for pi in p_inc_range]
    # Sensitivity on d_e: 0, 5k, 10k, 20k
    d_e_vals = [0, 5_000, 10_000, 20_000]
    colors_de = ["#9E9E9E", "#FF9800", "#2196F3", "#4CAF50"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: NPV vs P_inc (at base d_e=10k)
    ax = axes[0]
    ax.plot(p_inc_range / 1000, [n / 1000 for n in base_npvs], color="#2196F3", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(P_new / 1000, color="gray", linewidth=1, linestyle=":", label=f"P_new=${P_new/1e3:.0f}k")
    ax.set_title("CloudMetrics NPV vs Incumbent Price\n(d_e=$10k, SC=$90k, r=10%, T=4)", fontsize=10)
    ax.set_xlabel("Incumbent Price P_inc ($k)")
    ax.set_ylabel("NPV ($k)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: NPV vs P_inc for different d_e
    ax2 = axes[1]
    for d_e, col in zip(d_e_vals, colors_de):
        npvs = [models.npv(p_inc=pi, p_new=P_new, d_e=d_e, r=R, t=T, sc=SC)
                for pi in p_inc_range]
        ax2.plot(p_inc_range / 1000, [n / 1000 for n in npvs],
                 color=col, linewidth=2, label=f"d_e=${d_e/1000:.0f}k")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("CloudMetrics NPV Sensitivity to Efficiency Gains (d_e)", fontsize=10)
    ax2.set_xlabel("Incumbent Price P_inc ($k)")
    ax2.set_ylabel("NPV ($k)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("CloudMetrics: Switching Cost NPV Analysis (Hypothetical)", fontsize=12, y=1.02)
    save(fig, out_dir / "fig4_cloudmetrics_npv.png", dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    out_dir = root_path(cfg["paths"]["figures"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = cfg["paths"].get("figure_dpi", 300)

    print("Generating figures...")
    fm = pd.read_csv(root_path(cfg["paths"]["outputs"]) / "financial_metrics.csv",
                     keep_default_na=False)
    forces_summary = pd.read_csv(root_path(cfg["paths"]["outputs"]) / "forces_summary.csv",
                                 keep_default_na=False)

    fig1_mas_radar(forces_summary, cfg, out_dir, dpi)
    fig2_lerner_trends(fm, cfg, out_dir, dpi)
    fig3_cloudmetrics_breakeven(out_dir, dpi)
    fig4_cloudmetrics_npv(out_dir, dpi)

    print("All figures written to outputs/figures/")


if __name__ == "__main__":
    main()
