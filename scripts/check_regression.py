"""
Regression Checker — Session 2 Starter

Compares current eval scores against a saved baseline.
Flags any metric that drops more than the threshold.

Functions to implement:
  1. load_baseline() — load baseline_scores.json
  2. load_current() — load eval_results.json (note: scores are under "summary" key)
  3. check_regression() — compare metric by metric, return list of regressions
  4. display_results() — print a clear pass/fail table with deltas

Run: python scripts/check_regression.py
Run with options:
  python scripts/check_regression.py --threshold 3.0
  python scripts/check_regression.py --baseline scripts/baseline_scores.json
"""
import os
import sys
import json
import argparse

SCRIPT_DIR = os.path.dirname(__file__)

DEFAULT_BASELINE = os.path.join(SCRIPT_DIR, "baseline_scores.json")
DEFAULT_CURRENT = os.path.join(SCRIPT_DIR, "..", "eval_results.json")
DEFAULT_THRESHOLD = 5.0  # percentage points


# =========================================================================
# FUNCTIONS TO IMPLEMENT IN SESSION 2
# =========================================================================

def load_baseline(path: str) -> dict:
    """
    Load baseline scores from JSON file.
    Returns the parsed dict.

    TODO: Implement in Session 2.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data["summary"] if "summary" in data else data


def load_current(path: str) -> dict:
    """
    Load current eval results from JSON file.
    Note: eval_results.json wraps scores inside a "summary" key.
    If "summary" is present, return data["summary"]. Otherwise return data directly.

    TODO: Implement in Session 2.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Current eval file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data["summary"] if "summary" in data else data


def check_regression(current: dict, baseline: dict, threshold: float = DEFAULT_THRESHOLD) -> list:
    """
    Compare current scores against baseline for these metrics:
      - retrieval_hit_rate
      - avg_faithfulness
      - avg_correctness

    For each metric:
      - Calculate delta = current[metric] - baseline[metric]
      - Mark as regression if delta < -threshold

    Returns a list of dicts:
      [{"metric": "...", "baseline": N, "current": N, "delta": N, "is_regression": bool}]

    TODO: Implement in Session 2.
    """
    metrics = ["retrieval_hit_rate", "avg_faithfulness", "avg_correctness"]
    regressions = []

    for metric in metrics:
        if metric not in baseline or metric not in current:
            raise KeyError(f"Missing metric '{metric}' in baseline or current scores")
        delta = current[metric] - baseline[metric]
        regressions.append({
            "metric": metric,
            "baseline": baseline[metric],
            "current": current[metric],
            "delta": round(delta, 2),
            "is_regression": delta < -threshold,
        })

    return regressions


def display_results(regressions: list, threshold: float):
    """
    Print a clear comparison table showing baseline vs current for each metric.
    Show PASS (green) or REGRESSION (red) for each.
    Print a headline: ✅ NO REGRESSION or ❌ REGRESSION DETECTED.

    TODO: Implement in Session 2.
    """
    has_regression = any(result["is_regression"] for result in regressions)
    print("\nREGRESSION CHECK")
    print(f"Threshold: {threshold:.1f} percentage points")
    print("metric | baseline | current | delta | status")

    for result in regressions:
        status = "REGRESSION" if result["is_regression"] else "PASS"
        print(
            f"{result['metric']} | {result['baseline']:.2f} | "
            f"{result['current']:.2f} | {result['delta']:+.2f} | {status}"
        )

    if has_regression:
        print("\nREGRESSION DETECTED")
    else:
        print("\nNO REGRESSION")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Regression checker for RAG eval")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE)
    parser.add_argument("--current", type=str, default=DEFAULT_CURRENT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Regression threshold in percentage points (default: 5.0)")
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    current = load_current(args.current)
    regressions = check_regression(current, baseline, args.threshold)
    display_results(regressions, args.threshold)

    if any(result["is_regression"] for result in regressions):
        sys.exit(1)


if __name__ == "__main__":
    main()
