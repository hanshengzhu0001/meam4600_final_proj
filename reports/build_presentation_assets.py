from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


RUNS = {
    "baseline": ROOT / "outputs/posterior_projection_baseline/eval_full.json",
    "g8": ROOT / "outputs/guidance_big_g8.0/eval_full.json",
    "g10": ROOT / "outputs/guidance_big_g10.0/eval_full.json",
    "g14": ROOT / "outputs/guidance_big_g14.0/eval_full.json",
    "g16": ROOT / "outputs/guidance_big_g16.0/eval_full.json",
}

GUIDANCE_LEVEL = {
    "baseline": 0.0,
    "g8": 8.0,
    "g10": 10.0,
    "g14": 14.0,
    "g16": 16.0,
}


def load_rows(path: Path) -> list[dict[str, float | str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def best(rows: list[dict[str, float | str]], key: str) -> dict[str, float | str]:
    return min(rows, key=lambda row: float(row[key]))


def pct_change(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def short_case_label(row: dict[str, float | str]) -> str:
    return f"{row['schedule']}/{row['order']}"


def main() -> None:
    rows_by_run: dict[str, list[dict[str, float | str]]] = {
        name: load_rows(path) for name, path in RUNS.items()
    }

    best_u = {name: best(rows, "u_error") for name, rows in rows_by_run.items()}
    best_obj = {
        metric: min(
            (
                (run_name, best(rows, metric))
                for run_name, rows in rows_by_run.items()
            ),
            key=lambda item: float(item[1][metric]),
        )
        for metric in [
            "posterior_quality",
            "physical_consistency",
            "runtime",
            "trajectory_stability",
            "u_error",
            "v_error",
            "obs_error",
        ]
    }

    baseline_best_u = best_u["baseline"]
    g10_best_u = best_u["g10"]
    g16_best_u = best_u["g16"]

    summary = {
        "best_u_rows": best_u,
        "best_objectives": {
            metric: {
                "run": run_name,
                "schedule": row["schedule"],
                "order": row["order"],
                "value": float(row[metric]),
            }
            for metric, (run_name, row) in best_obj.items()
        },
        "deltas": {
            "g16_vs_baseline_best_u": {
                metric: pct_change(float(g16_best_u[metric]), float(baseline_best_u[metric]))
                for metric in ["posterior_quality", "u_error", "v_error", "obs_error", "runtime"]
            },
            "g16_vs_g10_best_u": {
                metric: pct_change(float(g16_best_u[metric]), float(g10_best_u[metric]))
                for metric in ["posterior_quality", "u_error", "v_error", "obs_error", "runtime"]
            },
        },
    }
    (REPORTS_DIR / "presentation_data.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Figure 1: Guidance trend on best-u rows
    run_order = ["baseline", "g8", "g10", "g14", "g16"]
    x = [GUIDANCE_LEVEL[r] for r in run_order]
    pq = [float(best_u[r]["posterior_quality"]) for r in run_order]
    ue = [float(best_u[r]["u_error"]) for r in run_order]
    ve = [float(best_u[r]["v_error"]) for r in run_order]
    oe = [float(best_u[r]["obs_error"]) for r in run_order]
    rt = [float(best_u[r]["runtime"]) for r in run_order]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)

    ax = axes[0]
    ax.plot(x, pq, marker="o", linewidth=2.0, label="posterior_quality")
    ax.plot(x, ue, marker="o", linewidth=2.0, label="u_error")
    ax.plot(x, ve, marker="o", linewidth=2.0, label="v_error")
    ax.plot(x, oe, marker="o", linewidth=2.0, label="obs_error")
    ax.set_xlabel("Guidance Strength")
    ax.set_ylabel("Metric Value (lower is better)")
    ax.set_title("Best-u Row Metrics vs Guidance")
    ax.legend(frameon=True, fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(x, rt, marker="o", color="#aa3377", linewidth=2.2)
    ax.set_xlabel("Guidance Strength")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Best-u Row Runtime vs Guidance")
    for xi, yi, rn in zip(x, rt, run_order):
        ax.annotate(rn, (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "guidance_trend_best_u.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Runtime-quality tradeoff across runs (best-u rows)
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=180)
    colors = {
        "baseline": "#555555",
        "g8": "#1f77b4",
        "g10": "#2ca02c",
        "g14": "#ff7f0e",
        "g16": "#d62728",
    }
    for rn in run_order:
        row = best_u[rn]
        ax.scatter(
            float(row["runtime"]),
            float(row["u_error"]),
            s=90,
            color=colors[rn],
            edgecolors="black",
            linewidth=0.6,
            label=rn,
            zorder=3,
        )
        ax.annotate(
            f"{rn}\n{short_case_label(row)}",
            (float(row["runtime"]), float(row["u_error"])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.set_xlabel("Runtime [s] (lower is better)")
    ax.set_ylabel("u_error (relative L2, lower is better)")
    ax.set_title("Best-u Tradeoff Across Guidance Levels")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "runtime_uerror_tradeoff.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: g16 schedule/order tradeoff
    rows_g16 = rows_by_run["g16"]
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=180)
    order_color = {"first_order": "#1f77b4", "gauss_newton": "#d62728", "none": "#555555"}
    for row in rows_g16:
        o = str(row["order"])
        c = order_color.get(o, "#777777")
        ax.scatter(
            float(row["runtime"]),
            float(row["u_error"]),
            color=c,
            s=70,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.annotate(
            f"{row['schedule']}/{row['order']}",
            (float(row["runtime"]), float(row["u_error"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.set_xlabel("Runtime [s]")
    ax.set_ylabel("u_error")
    ax.set_title("g16 Full Study: Schedule/Order Runtime-vs-u_error")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "g16_schedule_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
