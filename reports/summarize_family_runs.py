from __future__ import annotations

import argparse
import json
from pathlib import Path


OBJECTIVES = (
    "posterior_quality",
    "physical_consistency",
    "runtime",
    "trajectory_stability",
)


def _load_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pct_delta(current: float, reference: float) -> float:
    if reference == 0.0:
        return float("inf")
    return 100.0 * (current - reference) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", required=True, help="glob for eval_full.json files")
    parser.add_argument(
        "--run-prefix",
        default="",
        help="prefix to strip from parent folder name to form run labels",
    )
    parser.add_argument(
        "--baseline",
        default="g0.25",
        help="baseline run label for best-u delta calculations",
    )
    args = parser.parse_args()

    paths = sorted(Path().glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"no files matched {args.glob!r}")

    runs: list[tuple[str, list[dict[str, float | str]], dict[str, float | str]]] = []
    for path in paths:
        run_label = path.parent.name
        if args.run_prefix and run_label.startswith(args.run_prefix):
            run_label = run_label[len(args.run_prefix) :]
        rows = _load_rows(path)
        best_u = min(rows, key=lambda row: float(row["u_error"]))
        runs.append((run_label, rows, best_u))

    print("Best-u row per run:")
    for run_label, _, best_u in runs:
        print(
            f"- {run_label}: case={best_u['schedule']}/{best_u['order']} "
            f"u={float(best_u['u_error']):.6f} "
            f"v={float(best_u['v_error']):.6f} "
            f"obs={float(best_u['obs_error']):.6f} "
            f"pq={float(best_u['posterior_quality']):.6f} "
            f"rt={float(best_u['runtime']):.6f}"
        )

    print("\nObjective winners across runs:")
    for metric in OBJECTIVES:
        winner_run = ""
        winner_row: dict[str, float | str] | None = None
        for run_label, rows, _ in runs:
            row = min(rows, key=lambda candidate: float(candidate[metric]))
            if winner_row is None or float(row[metric]) < float(winner_row[metric]):
                winner_row = row
                winner_run = run_label
        assert winner_row is not None
        print(
            f"- {metric}: run={winner_run} "
            f"case={winner_row['schedule']}/{winner_row['order']} "
            f"value={float(winner_row[metric]):.6f}"
        )

    baseline_row: dict[str, float | str] | None = None
    best_run = ""
    best_row: dict[str, float | str] | None = None
    for run_label, _, best_u in runs:
        if run_label == args.baseline:
            baseline_row = best_u
        if best_row is None or float(best_u["u_error"]) < float(best_row["u_error"]):
            best_run = run_label
            best_row = best_u

    if baseline_row is None or best_row is None:
        return

    print(f"\nBest-u run vs baseline ({args.baseline}):")
    print(f"- best_u_run={best_run} case={best_row['schedule']}/{best_row['order']}")
    for metric in ("u_error", "v_error", "obs_error", "posterior_quality", "runtime"):
        delta = _pct_delta(float(best_row[metric]), float(baseline_row[metric]))
        print(f"- {metric}: {delta:+.2f}%")


if __name__ == "__main__":
    main()
