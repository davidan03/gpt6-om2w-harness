"""Compute the dashboard's "success rate w/o aborted tasks" metric for one eval
run directory, and aggregate mean/std across multiple runs (e.g. 3 judge runs).

Methodology (matches the dashboard's documented pseudocode):
    for task in run:                      # 300 tasks total
        if task.status == ABORTED:
            exclude_from_denom()          # sglang/env crash -- no output produced
        elif task.status == COMPLETED and reward >= 1.0:
            count_success()
        elif task.status == COMPLETED and reward < 1.0:
            if BLOCKED_RE matches the agent's final done() response text:
                exclude_from_denom()      # agent explicitly reports being blocked
            else:
                count_as_failure()
        else:  # FAILED / TRUNCATED
            count_as_failure()

Difference from the original methodology used for the Reproduction Results tab:
this script does NOT apply the per-task forensic env-block check on timed-out
tasks (browser-state snapshot inspection at timeout) -- that step required
manual/semi-manual inspection of saved screenshots in earlier runs and isn't
reproduced here. The dashboard's own numbers show this affects at most ~30 of
5,400 tasks across 18 runs (<1%), so the effect on comparability is negligible,
but it is a known, disclosed simplification.

Usage:
    python compute_eval_success_rate_woaborted.py <eval_dir_1> <eval_dir_2> <eval_dir_3>
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

BLOCKED_RE = re.compile(
    r"captcha|cloudflare|access.?denied|403|verification|blocked",
    re.IGNORECASE,
)


def _load_jsonl_record(path: Path) -> dict[str, Any] | None:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _reward_value(record: dict[str, Any]) -> float:
    reward = record.get("reward")
    if isinstance(reward, (int, float)):
        return float(reward)
    reward_meta = (record.get("metadata") or {}).get("reward") or {}
    combined = reward_meta.get("combined")
    if isinstance(combined, (int, float)):
        return float(combined)
    return 0.0


def _final_done_text(record: dict[str, Any]) -> str:
    """Best-effort extraction of the agent's final done() response text from
    the raw `response` field, across both JSON tool-call formats
    ({"name": "done", "arguments": {"response": "..."}}) and XML
    (<parameter=response>...) formats. Falls back to the whole response text
    if a done() call can't be isolated -- BLOCKED_RE is permissive enough that
    this rarely changes the match outcome."""
    response = record.get("response") or ""

    m = re.search(r'"name"\s*:\s*"done".*?"response"\s*:\s*"(.*?)(?<!\\)"', response, re.DOTALL)
    if m:
        return m.group(1)

    m = re.search(r"<parameter=response>(.*?)</parameter>", response, re.DOTALL)
    if m:
        return m.group(1)

    return response


def compute_stats(eval_dir: Path) -> dict[str, Any]:
    result_paths = sorted(eval_dir.glob("**/results_task_*.jsonl"))

    n_total = 0
    n_excluded_aborted = 0
    n_excluded_self_blocked = 0
    n_success = 0
    n_failure = 0

    for path in result_paths:
        record = _load_jsonl_record(path)
        if record is None:
            continue
        n_total += 1
        status = str(record.get("status", ""))

        if "ABORTED" in status:
            n_excluded_aborted += 1
            continue

        reward = _reward_value(record)
        if "COMPLETED" in status and reward >= 1.0:
            n_success += 1
        elif "COMPLETED" in status and reward < 1.0:
            if BLOCKED_RE.search(_final_done_text(record)):
                n_excluded_self_blocked += 1
            else:
                n_failure += 1
        else:  # FAILED / TRUNCATED
            n_failure += 1

    denom = n_success + n_failure
    success_rate = n_success / denom if denom else 0.0

    return {
        "eval_dir": str(eval_dir),
        "n_total_result_files": n_total,
        "n_excluded_aborted": n_excluded_aborted,
        "n_excluded_self_blocked": n_excluded_self_blocked,
        "n_success": n_success,
        "n_failure": n_failure,
        "denominator": denom,
        "success_rate": success_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dirs", nargs="+", help="One or more eval run directories to aggregate")
    parser.add_argument("--json", action="store_true", help="Print per-run + aggregate stats as JSON")
    args = parser.parse_args()

    per_run = [compute_stats(Path(d)) for d in args.eval_dirs]
    rates = [r["success_rate"] for r in per_run]

    mean = statistics.mean(rates) if rates else 0.0
    stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0

    if args.json:
        print(json.dumps({"per_run": per_run, "mean": mean, "stdev": stdev}, indent=2))
        return

    for r in per_run:
        print(
            f"{r['eval_dir']}: success_rate={r['success_rate']*100:.1f}% "
            f"(n_success={r['n_success']}, denom={r['denominator']}, "
            f"excluded_aborted={r['n_excluded_aborted']}, excluded_self_blocked={r['n_excluded_self_blocked']}, "
            f"total_files={r['n_total_result_files']})"
        )
    print()
    print(f"mean = {mean*100:.1f}%  std = {stdev*100:.1f}%  (n_runs={len(rates)})")


if __name__ == "__main__":
    main()
