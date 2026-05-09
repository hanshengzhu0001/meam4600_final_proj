#!/usr/bin/env python3
"""Build publication-style figures for the final project report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )
    plt.style.use("seaborn-v0_8-whitegrid")


def best_u_row(path: Path) -> dict:
    rows = json.loads(path.read_text())
    return min(rows, key=lambda r: r["u_error"])


def build_nonlinear_guidance_trend() -> None:
    guidances = [2.0, 4.0, 8.0, 10.0, 14.0, 16.0]
    u_vals: list[float] = []
    v_vals: list[float] = []
    obs_vals: list[float] = []
    for g in guidances:
        row = best_u_row(ROOT / f"outputs/guidance_big_g{g:.1f}/eval_full.json")
        u_vals.append(row["u_error"])
        v_vals.append(row["v_error"])
        obs_vals.append(row["obs_error"])

    baseline = best_u_row(ROOT / "outputs/posterior_projection_baseline/eval_full.json")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(guidances, u_vals, color="#1f77b4", marker="o", lw=2.4, label="u_error (inverse target)")
    ax.plot(guidances, v_vals, color="#ff7f0e", marker="s", lw=2.2, label="v_error (full-state recon)")
    ax.plot(guidances, obs_vals, color="#2ca02c", marker="^", lw=2.2, label="obs_error (observed fit)")
    ax.axhline(baseline["u_error"], color="#1f77b4", lw=1.4, ls=":", alpha=0.8)
    ax.text(guidances[0] - 0.3, baseline["u_error"] + 0.01, "baseline best u_error", color="#1f77b4", fontsize=8)
    ax.set_title("Nonlinear Elliptic: Stronger Guidance Improves Inverse Reconstruction", pad=10)
    ax.set_xlabel("Observation guidance strength g")
    ax.set_ylabel("Relative error (lower is better)")
    ax.set_xticks(guidances)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "nonlinear_guidance_trend.png", bbox_inches="tight")
    plt.close(fig)


def build_tradeoff_scatter() -> None:
    rows = json.loads((ROOT / "outputs/guidance_big_g16.0/eval_full.json").read_text())

    color_by_order = {"first_order": "#2166ac", "gauss_newton": "#b2182b", "none": "#4d4d4d"}
    marker_by_schedule = {
        "none": "o",
        "final_only": "s",
        "every_step": "^",
        "every_2": "D",
        "every_5": "P",
        "late_only": "X",
        "adaptive_residual": "v",
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for row in rows:
        order = row["order"]
        schedule = row["schedule"]
        ax.scatter(
            row["runtime"],
            row["u_error"],
            s=95,
            c=color_by_order.get(order, "#4d4d4d"),
            marker=marker_by_schedule.get(schedule, "o"),
            alpha=0.92,
            edgecolor="white",
            linewidth=0.9,
        )

    best_u = min(rows, key=lambda r: r["u_error"])
    fastest = min(rows, key=lambda r: r["runtime"])
    best_phys = min(rows, key=lambda r: r["physical_consistency"])
    for row, text, xytext in [
        (best_u, "Best u_error", (8, -14)),
        (fastest, "Fastest", (8, 8)),
        (best_phys, "Best physics", (-60, -14)),
    ]:
        ax.scatter(row["runtime"], row["u_error"], s=180, facecolors="none", edgecolors="#111111", linewidth=1.4)
        ax.annotate(text, (row["runtime"], row["u_error"]), textcoords="offset points", xytext=xytext, fontsize=8)

    order_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_by_order["first_order"], markersize=8, label="order: first_order"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_by_order["gauss_newton"], markersize=8, label="order: gauss_newton"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_by_order["none"], markersize=8, label="order: none"),
    ]
    schedule_handles = [
        plt.Line2D([0], [0], marker=m, color="#666666", linestyle="None", markersize=7, label=f"schedule: {s}")
        for s, m in marker_by_schedule.items()
    ]
    legend1 = ax.legend(handles=order_handles, loc="upper right", frameon=True, framealpha=0.9)
    ax.add_artist(legend1)
    ax.legend(handles=schedule_handles, loc="lower right", frameon=True, framealpha=0.9, ncol=1)

    ax.set_title("Nonlinear Elliptic (g=16): Quality vs Runtime by Projection Policy", pad=10)
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("u_error (lower is better)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "nonlinear_tradeoff_scatter.png", bbox_inches="tight")
    plt.close(fig)


def build_workflow_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.5))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#fbfcfe", edgecolor="none", zorder=0))
    ax.plot([0.05, 0.95], [0.47, 0.47], color="#d7dee8", linewidth=1.8, zorder=1)

    stages = [
        (0.02, 0.20, 0.18, 0.54, "1", "Input", "Dataset + PDE Family", "Oracle paired data"),
        (0.215, 0.20, 0.18, 0.54, "2", "Model", "Train Flow Prior", "Joint state model [u; v]"),
        (0.41, 0.20, 0.18, 0.54, "3", "Inference", "Posterior Sampling", "Partial-v guidance"),
        (0.605, 0.20, 0.18, 0.54, "4", "Correction", "Projection Policy", "Schedule + order sweep"),
        (0.80, 0.20, 0.18, 0.54, "5", "Output", "Evaluation", "Quality, physics, runtime"),
    ]
    accent = "#2e5c89"
    for x, y, w, h, idx, lane, title, subtitle in stages:
        shadow = FancyBboxPatch(
            (x + 0.004, y - 0.008),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            linewidth=0,
            facecolor="#000000",
            alpha=0.08,
            zorder=2,
        )
        ax.add_patch(shadow)

        card = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            linewidth=1.1,
            edgecolor="#c7d3e0",
            facecolor="#ffffff",
            zorder=3,
        )
        ax.add_patch(card)
        ax.add_patch(Rectangle((x, y + h - 0.012), w, 0.012, facecolor=accent, edgecolor="none", zorder=4))

        tag = FancyBboxPatch(
            (x + 0.012, y + h - 0.085),
            0.072,
            0.040,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0,
            facecolor="#eaf1f8",
            zorder=4,
        )
        ax.add_patch(tag)
        ax.text(x + 0.048, y + h - 0.065, lane, ha="center", va="center", fontsize=8.2, color="#35556f", zorder=5)

        ax.add_patch(Circle((x + w - 0.026, y + h - 0.064), 0.0165, facecolor=accent, edgecolor="none", zorder=4))
        ax.text(x + w - 0.026, y + h - 0.064, idx, ha="center", va="center", fontsize=8.7, color="white", weight="bold", zorder=5)
        ax.text(x + w / 2, y + 0.30, title, ha="center", va="center", fontsize=10.0, color="#17344f", weight="bold", zorder=5)
        ax.text(x + w / 2, y + 0.21, subtitle, ha="center", va="center", fontsize=8.9, color="#4a6074", zorder=5)

    for i in range(len(stages) - 1):
        x, y, w, h, *_ = stages[i]
        x2, y2, w2, h2, *_ = stages[i + 1]
        arrow = FancyArrowPatch(
            (x + w + 0.004, y + h / 2),
            (x2 - 0.004, y2 + h2 / 2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.8,
            color="#5d748b",
            connectionstyle="arc3,rad=0.0",
            zorder=5,
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.93,
        "Workflow: From PDE Formulation to Objective-Aware Projection Selection",
        ha="center",
        va="center",
        fontsize=13.0,
        color="#1b3651",
        weight="bold",
    )
    ax.text(
        0.5,
        0.875,
        "I sweep projection schedule and order in Stage 4 under fixed training and sampling settings.",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#4d667e",
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "workflow_flowchart.png", bbox_inches="tight")
    plt.close(fig)


def build_cross_family_bar() -> None:
    families = [
        ("Linear elliptic", 18.73),
        ("Heat equation", 25.20),
        ("Reaction-diffusion", 29.84),
        ("Burgers IC", 26.12),
        ("Burgers BC", 9.03),
        ("Navier-Stokes\n1D surrogate", 24.11),
    ]
    labels = [k for k, _ in families]
    vals = [v for _, v in families]

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    colors = ["#28587b", "#2f6f8f", "#3a88a3", "#43a3b5", "#7aa6c2", "#2f8f8a"]
    bars = ax.barh(labels, vals, color=colors, edgecolor="none")
    ax.set_xlabel("u_error reduction (%) from g=0.25 to g=16")
    ax.set_title("Cross-Family Benefit of Stronger Observation Guidance", pad=10)
    ax.set_xlim(0, 35)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.35, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "cross_family_u_improvement.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    build_nonlinear_guidance_trend()
    build_tradeoff_scatter()
    build_workflow_flowchart()
    build_cross_family_bar()
    print(f"Wrote figures to {FIGDIR}")


if __name__ == "__main__":
    main()
