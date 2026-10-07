"""
Browser environment helpers used by astra_eval.py: create a local_process
browser env for a task, apply launch-script overrides to its config, and save
a generated sample to disk.

Adapted from slime's multi-turn rollout interface.
"""

import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from typing import Any

# Suppress noisy asyncio socket warnings that flood logs on connection failures.
# Set level to CRITICAL and install a filter as a safety net in case Ray or
# another library resets the level after import.
_asyncio_logger = logging.getLogger("asyncio")
_asyncio_logger.setLevel(logging.CRITICAL)
_asyncio_logger.addFilter(lambda record: "socket.send()" not in record.getMessage())

# Use fully-qualified imports so that importlib.import_module() in Ray workers
# can resolve them without sys.path hacks.
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from openwebrl.response_format import (
    get_browser_response_format_mode,
    rewrite_policy_thinking_tags,
)

from openwebrl.sample import Sample

# Resolve the openwebrl/ directory once at import time.
_BROWSER_DIR = os.path.dirname(os.path.abspath(__file__))
_TASK_JSONL_CACHE: dict[str, dict[str, Any]] = {}


def _build_fallback_task_data(task_id: str, task_metadata: dict[str, Any] | None) -> dict[str, Any]:
    task_metadata = task_metadata or {}
    start_url = task_metadata.get("start_url", "")
    intent = task_metadata.get("task") or task_metadata.get("intent", "")
    if not start_url or not intent:
        raise ValueError(
            f"❗  Task {task_id} not found locally and metadata fallback is incomplete: "
            f"start_url={bool(start_url)} intent={bool(intent)}"
        )

    numeric_task_id = task_id.split("/")[-1] if "/" in task_id else task_id
    return {
        "sites": task_metadata.get("sites", []),
        "task_id": numeric_task_id,
        "require_login": bool(task_metadata.get("require_login", False)),
        "storage_state": task_metadata.get("storage_state"),
        "start_url": start_url,
        "intent": intent,
        "require_reset": bool(task_metadata.get("require_reset", False)),
        "eval": task_metadata.get(
            "eval",
            {
                "eval_types": ["string_match"],
                "reference_answers": {"fuzzy_match": [""]},
                "reference_url": "",
                "program_html": [],
                "string_note": "",
                "reference_answer_raw_annotation": "",
            },
        ),
        "intent_template_id": task_metadata.get("intent_template_id", 0),
        "old_task_id": task_metadata.get("old_task_id", task_id),
    }


def _task_record_to_task_data(record: dict[str, Any], requested_task_id: str) -> dict[str, Any]:
    metadata = dict(record.get("metadata", {}) or {})
    resolved_task_id = (
        metadata.get("task_id")
        or record.get("id")
        or requested_task_id
    )
    start_url = metadata.get("start_url") or record.get("web", "")
    intent = metadata.get("intent") or record.get("ques", "")
    if not start_url or not intent:
        raise ValueError(
            f"❗  Task record for {requested_task_id} is missing required fields: "
            f"start_url={bool(start_url)} intent={bool(intent)}"
        )

    return {
        "sites": metadata.get("sites", [record.get("web_name")] if record.get("web_name") else []),
        "task_id": resolved_task_id,
        "require_login": bool(metadata.get("require_login", False)),
        "storage_state": metadata.get("storage_state"),
        "start_url": start_url,
        "intent": intent,
        "require_reset": bool(metadata.get("require_reset", False)),
        "eval": metadata.get(
            "eval",
            {
                "eval_types": ["string_match"],
                "reference_answers": {"fuzzy_match": [""]},
                "reference_url": "",
                "program_html": [],
                "string_note": "",
                "reference_answer_raw_annotation": "",
            },
        ),
        "intent_template_id": metadata.get("intent_template_id", 0),
        "old_task_id": record.get("id", resolved_task_id),
    }


def _load_task_jsonl_cache(path: str) -> dict[str, Any]:
    cached_by_id = _TASK_JSONL_CACHE.get(path)
    if cached_by_id is not None:
        return cached_by_id

    by_id: dict[str, Any] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {}) or {}
            candidate_ids = [
                record.get("id"),
                metadata.get("task_id"),
            ]
            for candidate in candidate_ids:
                if isinstance(candidate, str) and candidate and candidate not in by_id:
                    by_id[candidate] = record

    _TASK_JSONL_CACHE[path] = by_id
    return by_id


def _collect_task_lookup_candidates(task_id: str, task_metadata: dict[str, Any] | None) -> list[str]:
    metadata = task_metadata or {}
    candidates: list[str] = []
    for value in (
        task_id,
        metadata.get("old_task_id"),
        metadata.get("task_id"),
        metadata.get("id"),
    ):
        if isinstance(value, str):
            value = value.strip()
            if value and value not in candidates:
                candidates.append(value)
    return candidates


def _try_load_task_from_jsonl(task_id: str, task_metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    configured_task_file = None
    if task_metadata and isinstance(task_metadata.get("_browser_task_file"), str):
        configured_task_file = task_metadata["_browser_task_file"].strip() or None

    jsonl_paths: list[str] = []
    if configured_task_file:
        configured_path = configured_task_file
        if not os.path.isabs(configured_path):
            configured_path = os.path.join(_BROWSER_DIR, configured_path)
        if os.path.isfile(configured_path):
            jsonl_paths.append(configured_path)

    if not jsonl_paths:
        return None

    candidates = _collect_task_lookup_candidates(task_id, task_metadata)
    for path in jsonl_paths:
        by_id = _load_task_jsonl_cache(path)
        for candidate in candidates:
            record = by_id.get(candidate)
            if record is not None:
                logger.info(f"Loaded task {task_id} from JSONL {path} using key {candidate}")
                return _task_record_to_task_data(record, task_id)

    return None


def _load_local_resources(
    task_id: str,
    env_config: dict,
    task_metadata: dict[str, Any] | None = None,
    response_format_mode_name: str | None = None,
):
    """Load task_data, tool_list, and policy from the local filesystem."""
    tool_list = []
    _tool_list_path = os.path.join(_BROWSER_DIR, env_config["path_to_tool_list"])
    if os.path.exists(_tool_list_path):
        with open(_tool_list_path, "r") as f:
            tool_list = json.load(f)

    response_mode = get_browser_response_format_mode(response_format_mode_name)

    policy = ""
    _policy_path = os.path.join(_BROWSER_DIR, response_mode.policy_relpath)
    if os.path.exists(_policy_path):
        with open(_policy_path, "r") as f:
            policy = f.read()
    policy = rewrite_policy_thinking_tags(policy, response_mode)

    task_metadata = dict(task_metadata or {})
    configured_task_file = env_config.get("path_to_task_file")
    if configured_task_file and "_browser_task_file" not in task_metadata:
        task_metadata["_browser_task_file"] = configured_task_file

    task_data = _try_load_task_from_jsonl(task_id, task_metadata)
    if task_data is None:
        logger.warning(f"Task {task_id} not found locally, falling back to sample metadata.")
        task_data = _build_fallback_task_data(task_id, task_metadata)

    return task_data, tool_list, policy


def _apply_local_process_env_overrides(env_config: dict) -> dict:
    """Allow launch scripts to override local_process settings via environment variables."""
    local_cfg = dict(env_config.get("local_process", {}))
    override_map = {
        "max_processes": "SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES",
        "startup_timeout_secs": "SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS",
        "log_dir": "SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR",
        "port_lock_dir": "SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR",
        "python_bin": "SLIME_BROWSER_LOCAL_PROCESS_PYTHON",
    }
    int_keys = {"max_processes"}
    float_keys = {"startup_timeout_secs"}
    for key, env_name in override_map.items():
        value = os.environ.get(env_name)
        if value in (None, ""):
            continue
        if key in int_keys:
            local_cfg[key] = int(value)
        elif key in float_keys:
            local_cfg[key] = float(value)
        else:
            local_cfg[key] = value

    merged_config = dict(env_config)
    merged_config["local_process"] = local_cfg
    return merged_config


async def _create_env(
    task_id: str,
    env_config: dict,
    task_metadata: dict[str, Any] | None = None,
    response_format_mode_name: str | None = None,
):
    """
    Create a browser environment backed by a fresh local env_server
    subprocess, talked to over HTTP.

    All config/task/policy data is loaded from the local filesystem and
    sent to the env_server at initialization.

    Returns:
        (env, task_data) tuple
    """
    task_data, tool_list, policy = _load_local_resources(
        task_id,
        env_config,
        task_metadata,
        response_format_mode_name=response_format_mode_name,
    )

    # ---- Local-process mode: fresh local env_server subprocess per environment ----
    local_cfg = env_config.get("local_process", {})

    from openwebrl.env.local_process_env import create_local_process_env

    env = None
    try:
        env = await create_local_process_env(local_cfg)
        await env.initialize(
            task_id=task_id,
            task_data=task_data,
            env_config=env_config,
            tool_list=tool_list,
            policy=policy,
        )
    except BaseException:
        if env is not None:
            try:
                await env.exit()
            except Exception as cleanup_exc:
                logger.warning(
                    "Task %s: failed to clean up local_process env during initialization: %s",
                    task_id,
                    cleanup_exc,
                )
        raise

    return env, task_data


# ---------------------------------------------------------------------------
# Sample data saving utilities
# ---------------------------------------------------------------------------
def _save_sample(
    samples_dir: str,
    sample_id: str,
    mm_messages: list[dict],
    sample: Sample,
) -> None:
    os.makedirs(samples_dir, exist_ok=True)

    def _to_jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
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

    # Extract per-step images from user messages in mm_messages
    user_images: list[list[str]] = []
    for msg in mm_messages:
        if msg["role"] != "user" or not isinstance(msg.get("content"), list):
            continue
        imgs = [
            item["image_url"]
            for item in msg["content"]
            if isinstance(item, dict) and item.get("type") == "image_url"
        ]
        user_images.append(imgs)

    data = {
        "sample_id": f"{sample_id}",
        "total_steps": sample.metadata.get("total_steps", -1),
        "llm_input_texts": _to_jsonable(sample.prompt),
        "llm_response": _to_jsonable(sample.response),
        "images": _to_jsonable(user_images),
        "status": sample.status.value,
        "terminate_reason": _to_jsonable(sample.metadata.get("terminate_reason", "N/A")),
    }

    path_to_save = os.path.join(samples_dir, f"{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{data['status']}_{data['total_steps']}steps.json")
    with open(path_to_save, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
