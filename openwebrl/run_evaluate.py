"""
Evaluate browser agent success rate on a browser-env style JSONL task file.

Runs generate_turn_sample for each task, then uses the LLM judge
to determine success. Supports parallel evaluation with a bounded number
of concurrent environments.

astra_eval.py installs its generator as ``generate_turn_sample`` and then
calls ``main()``; see scripts/run_om2w.sh for the flags it is run with.
"""

import argparse
import asyncio
import json
import os
import sys
import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging

from openwebrl.reward_browser import reward_func
from openwebrl.sample import Sample

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# 禁用 httpx 库的 DEBUG 日志
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class EvalArgs:
    """Arguments for evaluation, read by the generator and by reward_browser."""
    max_steps: int = 16
    context_num_screenshots: int = 3
    path_to_save_generated_samples: str = ""
    task_timeout_secs: float = 900.0
    # reward config
    judge_max_attached_imgs: int = 3
    judge_timeout_secs: float = 120.0
    judge_api_model: str = "gpt-4.1" # o4-mini
    browser_response_format_mode: str = "browser_env"


async def generate_turn_sample(args, sample, sampling_params):
    """Turn-level generator; astra_eval.py replaces this module global before calling main()."""
    raise NotImplementedError("No generator installed: run through astra_eval.py.")


def _serialize_exception(exc: BaseException) -> dict[str, str]:
    """Convert an exception into a small JSON-serializable payload."""
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _to_jsonable(value: Any) -> Any:
    """Convert nested evaluation results into JSON-safe Python objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    value_type = type(value)
    if value_type.__name__ == "ndarray" and value_type.__module__.startswith("numpy"):
        return None
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def _build_abnormal_result(task: dict, reason: str, **extra: Any) -> dict[str, Any]:
    """Create a record for a task that did not finish cleanly."""
    result = {
        "task_id": task["task_id"],
        "index": task["index"],
        "start_url": task["start_url"],
        "intent": task["metadata"].get("task", task["prompt"]),
        "abnormal_reason": reason,
    }
    result.update(extra)
    return result


def _is_summary_eligible(result: dict[str, Any]) -> bool:
    """Only normally returned, non-pending task results should affect summary stats."""
    status = str(result.get("status", ""))
    return bool(status) and status != str(Sample.Status.PENDING)


def load_tasks_from_jsonl(task_file: str) -> list[dict]:
    """Load tasks from a browser-env style JSONL task file."""
    tasks = []
    with open(task_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = dict(row.get("metadata", {}) or {})
            task_id = metadata.get("task_id") or row.get("id") or f"task_{i}"
            prompt_text = metadata.get("intent") or row.get("ques", "")
            tasks.append({
                "index": i,
                "task_id": task_id,
                "start_url": metadata.get("start_url", row.get("web", "")),
                "prompt": prompt_text,
                "metadata": {
                    **metadata,
                    "task": metadata.get("intent", prompt_text),
                    "old_task_id": row.get("id", task_id),
                    "_browser_task_file": task_file,
                },
            })
    return tasks


async def evaluate_single_task(args: EvalArgs, task: dict) -> dict:
    """Run generation + judge reward for a single task. Returns a result dict."""
    async def _evaluate_single_task_impl() -> dict:
        task_id = task["task_id"]
        input_sample = Sample(
            index=task["index"],
            prompt=task["prompt"],
            metadata={
                "task_id": task_id,
                "start_url": task["start_url"],
                "intent": task["metadata"].get("task", task["prompt"]),
            },
        )

        print(f"Evaluating task {task_id} with turn-level evaluation...")
        turn_samples = await generate_turn_sample(args, input_sample, None)
        reward = await reward_func(args, turn_samples)
        result = turn_samples[-1]
        reward = reward[-1] if isinstance(reward, list) and reward else reward

        return {
            "task_id": task_id,
            "start_url": task["start_url"],
            "intent": task["metadata"].get("task"),
            "prompt": _to_jsonable(result.prompt),
            "response": _to_jsonable(result.response),
            "status": str(result.status),
            "reward": _to_jsonable(reward),
            "total_steps": result.metadata.get("total_steps", -1),
            "terminate_reason": result.metadata.get("terminate_reason", ""),
            "metadata": _to_jsonable(result.metadata),
        }

    task_id = task["task_id"]
    task_timeout_secs = args.task_timeout_secs
    try:
        return await asyncio.wait_for(_evaluate_single_task_impl(), timeout=task_timeout_secs)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Task {task_id} timed out after {task_timeout_secs}s") from exc


async def evaluate_all(
    args: EvalArgs,
    tasks: list[dict],
    n_parallel: int,
    output_path: str = "",
) -> tuple[list[dict], list[dict]]:
    """Evaluate all tasks with bounded parallelism using a semaphore."""
    sem = asyncio.Semaphore(n_parallel)
    completed_results: list[dict] = []
    abnormal_results: list[dict] = []

    async def run_with_sem(task):
        async with sem:
            print(f"[START] {task['task_id']}")
            try:
                result = await evaluate_single_task(args, task)
            except asyncio.CancelledError:
                abnormal = _build_abnormal_result(task, "cancelled")
                print(f"[ABNORMAL] {task['task_id']} -> cancelled")
                if output_path:
                    with open(os.path.join(output_path, f"abnormal_task_{task['task_id'].replace('/', '_')}.jsonl"), "w") as f:
                        f.write(json.dumps(_to_jsonable(abnormal)) + "\n")
                return ("abnormal", abnormal)
            except TimeoutError as exc:
                abnormal = _build_abnormal_result(task, "task_timeout", **_serialize_exception(exc))
                print(f"[ABNORMAL] {task['task_id']} -> timeout: {abnormal['error_message']}")
                if output_path:
                    with open(os.path.join(output_path, f"abnormal_task_{task['task_id'].replace('/', '_')}.jsonl"), "w") as f:
                        f.write(json.dumps(_to_jsonable(abnormal)) + "\n")
                return ("abnormal", abnormal)
            except Exception as exc:
                abnormal = _build_abnormal_result(task, "exception", **_serialize_exception(exc))
                print(f"[ABNORMAL] {task['task_id']} -> {abnormal['error_type']}: {abnormal['error_message']}")
                if output_path:
                    with open(os.path.join(output_path, f"abnormal_task_{task['task_id'].replace('/', '_')}.jsonl"), "w") as f:
                        f.write(json.dumps(_to_jsonable(abnormal)) + "\n")
                return ("abnormal", abnormal)

            if not _is_summary_eligible(result):
                abnormal = _build_abnormal_result(
                    task,
                    "pending_result",
                    status=result.get("status"),
                    total_steps=result.get("total_steps"),
                    terminate_reason=result.get("terminate_reason", ""),
                )
                print(
                    f"[ABNORMAL] {task['task_id']} -> "
                    f"[status={result.get('status')}, steps={result.get('total_steps')}]"
                )
                if output_path:
                    with open(os.path.join(output_path, f"abnormal_task_{task['task_id'].replace('/', '_')}.jsonl"), "w") as f:
                        f.write(json.dumps(_to_jsonable(abnormal)) + "\n")
                return ("abnormal", abnormal)

            print(f"[DONE]  {task['task_id']} -> [reward={result['reward']}, status={result['status']}, steps={result['total_steps']}]")
            if output_path:
                with open(os.path.join(output_path, f"results_task_{task['task_id'].replace('/', '_')}.jsonl"), "w") as f:
                    f.write(json.dumps(_to_jsonable(result)) + "\n")
            return ("completed", result)

    eval_tasks = [asyncio.create_task(run_with_sem(t)) for t in tasks]
    progress = tqdm(total=len(eval_tasks), desc="Evaluating", unit="task") if tqdm is not None else None
    finished = 0
    try:
        for eval_task in asyncio.as_completed(eval_tasks):
            result_type, payload = await eval_task
            if result_type == "completed":
                completed_results.append(payload)
            else:
                abnormal_results.append(payload)

            finished += 1
            if progress is not None:
                progress.set_postfix(
                    completed=len(completed_results),
                    abnormal=len(abnormal_results),
                    refresh=False,
                )
                progress.update(1)
            else:
                print(
                    f"[PROGRESS] {finished}/{len(eval_tasks)} tasks finished "
                    f"(completed={len(completed_results)}, abnormal={len(abnormal_results)})"
                )
    finally:
        if progress is not None:
            progress.close()

    return completed_results, abnormal_results


def get_summary(results: list[dict], abnormal_results: list[dict]) -> str:
    """Print evaluation summary."""
    total = len(results)

    stats = {"succeeded": []}
    for r in results:
        if r["reward"] == 1.0:
            stats["succeeded"].append(r)
        else:
            if r["status"] not in stats:
                stats[r["status"]] = [r]
            else:
                stats[r["status"]].append(r)

    success_rate = (len(stats["succeeded"]) / total * 100) if total else 0.0
    summary = (f"\n------------------------------\n"
               f"Total finished tasks: {total}\n"
               f"Success rate: {success_rate:.1f}%\n"
               f"------------------------------\n"
               f"Successes ({len(stats['succeeded'])}): {[r['task_id'] for r in stats['succeeded']]}\n")
    for status, stat_results in stats.items():
        if status == "succeeded":
            continue
        summary += f"{status.capitalize()} ({len(stat_results)}): {[r['task_id'] for r in stat_results]}\n"

    if stats["succeeded"]:
        summary += "\nDetails for succeeded tasks:\n"
        for r in stats["succeeded"]:
            summary += (
                f"- Task {r['task_id']}: reward={r['reward']}, status={r['status']}, "
                f"steps={r['total_steps']}, terminate_reason={r['terminate_reason']}\n"
            )

    for status, stat_results in stats.items():
        if status == "succeeded":
            continue
        summary += f"\nDetails for {status} tasks:\n"
        for r in stat_results:
            summary += f"- Task {r['task_id']}: reward={r['reward']}, steps={r['total_steps']}, terminate_reason={r['terminate_reason']}\n"

    summary += f"\nAbnormal / incomplete tasks ({len(abnormal_results)}): {[r['task_id'] for r in abnormal_results]}\n"
    if abnormal_results:
        summary += "\nDetails for abnormal / incomplete tasks:\n"
        for r in abnormal_results:
            detail = f"- Task {r['task_id']}: abnormal_reason={r['abnormal_reason']}"
            if r.get("status"):
                detail += f", status={r['status']}"
            if r.get("total_steps") is not None:
                detail += f", steps={r['total_steps']}"
            if r.get("terminate_reason"):
                detail += f", terminate_reason={r['terminate_reason']}"
            if r.get("error_type"):
                detail += f", error_type={r['error_type']}"
            if r.get("error_message"):
                detail += f", error_message={r['error_message']}"
            summary += detail + "\n"

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate browser agent success rate")
    parser.add_argument("--task-file", type=str, required=True,
                        help="Path to browser-env style JSONL task file.")
    parser.add_argument("--n-parallel", type=int, default=8,
                        help="Max number of concurrent browser environments")
    parser.add_argument("--hf-checkpoint", type=str, default=os.environ.get("MODEL_PATH") or os.environ.get("HF_CHECKPOINT", ""))
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--output", type=str, default=os.environ.get("OUTPUT_ROOT", "eval_outputs/browser_eval"),
                        help="Output JSONL path (default: auto-generated with timestamp)")
    parser.add_argument("--context-num-screenshots", type=int, default=3,
                        help="Number of latest screenshots kept in turn-level context.")
    parser.add_argument("--judge-max-attached-imgs", type=int, default=3,
                        help="Maximum number of screenshots attached to the judge prompt.")
    parser.add_argument(
        "--browser-response-format-mode",
        type=str,
        default="browser_env",
        choices=["browser_env"],
        help="Browser response format mode used by reward.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4.1")
    parser.add_argument(
        "--judge-api-mode",
        type=str,
        default="served",
        choices=["served"],
        help="Judge client mode. 'served' uses JUDGE_API_BASE.",
    )
    parser.add_argument("--judge-timeout-secs", type=float, default=30.0,
                        help="Timeout in seconds for a single judge API call.")
    parser.add_argument("--task-indices", type=str, default="",
                        help="Comma-separated task indices to evaluate (e.g. '0,5,10'). Empty = all.")
    parser.add_argument("--turn-level", action="store_true", required=True,
                        help="Run turn-level evaluation (the only supported mode).")
    parser.add_argument("--task-timeout-secs", type=float, default=600.0,
                        help="Timeout in seconds for a whole task evaluation.")
    args = parser.parse_args()

    checkpoint_tag = os.path.basename(os.path.normpath(args.hf_checkpoint))
    judge_tag = os.path.basename(os.path.normpath(args.judge_model))
    expt_name = f"eval_{checkpoint_tag}_{judge_tag}_turn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output = os.path.join(args.output, expt_name)
    os.makedirs(args.output, exist_ok=True)

    # Build eval args
    eval_args = EvalArgs(
        max_steps=args.max_steps,
        context_num_screenshots=args.context_num_screenshots,
        judge_max_attached_imgs=args.judge_max_attached_imgs,
        judge_api_model=args.judge_model,
        browser_response_format_mode=args.browser_response_format_mode,
        judge_timeout_secs=args.judge_timeout_secs,
        path_to_save_generated_samples=os.path.join(args.output, 'eval_samples'),
        task_timeout_secs=args.task_timeout_secs,
    )

    # Load tasks
    tasks = load_tasks_from_jsonl(args.task_file)
    print(f"Loaded {len(tasks)} tasks from {args.task_file}")

    # Filter tasks if specified
    if args.task_indices:
        indices = set(int(x) for x in args.task_indices.split(","))
        tasks = [t for t in tasks if t["index"] in indices]
        print(f"Filtered to {len(tasks)} tasks: {[t['task_id'] for t in tasks]}")

    # Run evaluation
    results, abnormal_results = asyncio.run(
        evaluate_all(eval_args, tasks, args.n_parallel, args.output)
    )

    summary = get_summary(results, abnormal_results)
    print(summary)

    summary_path = os.path.join(args.output, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")

    abnormal_path = os.path.join(args.output, "abnormal_tasks.jsonl")
    with open(abnormal_path, "w") as f:
        for result in abnormal_results:
            f.write(json.dumps(result) + "\n")
    print(f"Abnormal task list saved to {abnormal_path}")


if __name__ == "__main__":
    main()
