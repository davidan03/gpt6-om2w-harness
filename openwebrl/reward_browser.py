"""
Reward computation for browser agent tasks.

Two reward components:
1. Format reward  – deterministic check that the agent response contains a
   closing </think> tag and a parseable <tool_call>.
2. LLM-as-a-Judge – async OpenAI-compatible call that evaluates whether the
   task was successfully completed, adapted from the original GPT-4V evaluator.

Usage:
    from reward_browser import reward_func

    scores = await reward_func(args, turn_samples)
"""

import asyncio
import json
import logging
import os
import re
from typing import Any

from openai import AsyncOpenAI

from openwebrl.tool_parser import ToolParser
from openwebrl.feedback_utils import (
    DEFAULT_BROWSER_HISTORY_ENV_FEEDBACK_MAX_CHARS,
    normalize_feedback_text,
    truncate_feedback_text,
)
from openwebrl.response_format import get_browser_response_format_mode
from openwebrl.sample import Sample

logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

# Concurrency control
_reward_semaphore = asyncio.Semaphore(8)


# ==============================================================================
# LLM Judge Prompts (adapted from WebVoyager)
# ==============================================================================

JUDGE_SYSTEM_PROMPT = (
    "As an evaluator, you will be presented with three primary components "
    "to assist you in your role:\n\n"
    "1. Web Task Instruction: This is a clear and specific directive provided "
    "in natural language, detailing the online activity to be carried out.\n\n"
    "2. Result Screenshots: This is a visual representation of the screen "
    "showing the result or intermediate state of performing a web task.\n\n"
    "3. Result Response: This is a textual response obtained after the "
    "execution of the web task.\n\n"
    "-- You DO NOT NEED to interact with web pages or perform actions.\n"
    "-- You SHOULD NOT make assumptions based on information not presented "
    "in the screenshot when comparing it to the instructions.\n"
    "-- Your primary responsibility is to conduct a thorough assessment of "
    "the web task instruction against the outcome depicted in the screenshot "
    "and in the response, evaluating whether the actions taken align with "
    "the given instructions.\n"
    "-- NOTE that the instruction may involve more than one task. Failing to "
    "complete either task should be considered unsuccessful.\n"
    "-- NOTE that the screenshot is authentic, but the response provided by "
    "LLM is generated at the end of web browsing; there may be discrepancies "
    "between the text and the screenshots.\n"
    "-- Note the difference: 1) Result response may contradict the screenshot, "
    "then the content of the screenshot prevails, 2) The content in the Result "
    "response is not mentioned on the screenshot, choose to believe the content.\n\n"
    "You should elaborate on how you arrived at your final evaluation and then "
    "provide a definitive verdict on whether the task has been successfully "
    "accomplished, either as 'SUCCESS' or 'NOT SUCCESS'."
)


JUDGE_USER_PROMPT = (
    "### TASK: {task}\n"
    "### Result Response: {answer}\n"
    "### {num} screenshots at the end: "
)


# ==============================================================================
# 1. Format Reward
# ==============================================================================

# Module-level ToolParser for format reward checks.
_tool_parser = ToolParser()


def _get_browser_response_mode(args: Any):
    return get_browser_response_format_mode(args.browser_response_format_mode)


def _build_thinking_block_pattern(thinking_open_tag: str, thinking_close_tag: str) -> str:
    return rf"{re.escape(thinking_open_tag)}.*?{re.escape(thinking_close_tag)}"


def _compute_turn_format_reward_browser_env(args: Any, response: str) -> float:
    response = response.strip()
    if response.endswith("<|im_end|>"):
        response = response[: -len("<|im_end|>")].rstrip()

    response_mode = _get_browser_response_mode(args)
    has_think_close = response_mode.thinking_close_tag in response
    parsed = _tool_parser.parse(response)
    has_tool_call = parsed.success and len(parsed.calls) > 0
    return 1.0 if has_think_close and has_tool_call else 0.0


def compute_format_reward(args: Any, response: str, turn_level: bool = False) -> float:
    """
    Compute format reward across (multiple) assistant turns in a
    sample response string.

    For trajectory-level response strings, each assistant turn is delimited by ``<|im_start|>assistant\\n`` …
    ``<|im_end|>``. All turns are extracted, scored individually, and
    averaged.

    For turn-level response strings, the full response is treated as a single turn.

    Args:
        response: Full trajectory response string.
        turn_level: Whether to treat the response as a single turn.

    Returns:
        1.0 only if every turn scores 1.0; otherwise 0.0.
    """
    if turn_level:
        return _compute_turn_format_reward_browser_env(args, response)
    
    # Extract every assistant turn: content between <|im_start|>assistant\n
    # and <|im_end|>.  This is the standard chat format used in the data files.
    turn_pattern = r"<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>"
    turns = re.findall(turn_pattern, response, re.DOTALL)
    turns = [t.strip() for t in turns if t.strip()]

    if not turns:
        return 0.0

    scores = [_compute_turn_format_reward_browser_env(args, turn) for turn in turns]
    if all(s == 1.0 for s in scores):
        return 1.0
    return 0.0


# ==============================================================================
# 2. LLM-as-a-Judge Reward
# ==============================================================================

def _extract_final_answer_browser_env(response: str) -> str:
    parsed = _tool_parser.parse(response)
    if parsed.success and parsed.calls:
        last_call = parsed.calls[-1]
        if last_call.name != "done":
            return ""
        params = (
            json.loads(last_call.parameters)
            if isinstance(last_call.parameters, str)
            else last_call.parameters
        )
        if isinstance(params, dict):
            return str(params.get("response", "")).strip()
    return ""


def _get_judge_screenshots(args: Any, sample: Sample) -> list[str]:
    """
    Return the screenshots to attach to the judge prompt.

    Turn-level reward may inject a ``judge_images`` override containing
    trajectory-level recent screenshots; otherwise we fall back to the sample's
    own multimodal inputs.
    """
    multimodal_inputs = sample.multimodal_inputs or {}
    screenshots = multimodal_inputs.get("judge_images") or multimodal_inputs.get("images") or []

    max_imgs = getattr(args, "judge_max_attached_imgs")
    return screenshots[-max_imgs:] if screenshots else []


def _collect_recent_trajectory_images(samples: list[Sample], max_imgs: int) -> list[str]:
    """
    Aggregate one representative screenshot per turn sample and keep the most recent ones.

    For turn-level samples, ``multimodal_inputs["images"]`` may contain a rolling
    context window. We take only the last image from each turn sample, which
    corresponds to that turn's latest observation, then keep the most recent
    ``max_imgs`` of those turn-level screenshots.
    """
    ordered_samples = sorted(samples, key=lambda s: s.metadata.get("turn_index", -1))
    all_images: list[str] = []

    for sample in ordered_samples:
        multimodal_inputs = sample.multimodal_inputs or {}
        sample_images = multimodal_inputs.get("images", []) or []
        if sample_images:
            all_images.append(sample_images[-1])

    return all_images[-max_imgs:] if all_images else []


def _clean_think_from_action_history(text: str, args: Any) -> str:
    response_mode = _get_browser_response_mode(args)
    pattern = _build_thinking_block_pattern(
        response_mode.thinking_open_tag,
        response_mode.thinking_close_tag,
    )
    return re.sub(pattern, "", text, flags=re.DOTALL)


def _clean_img_placeholders_from_action_history(text: str) -> str:
    screenshot_placeholder = "screenshot:\n<|vision_start|><|image_pad|><|vision_end|>\n"
    return text.replace(screenshot_placeholder, "")


def _format_tool_call_for_action_history(name: str, parameters: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        name = "unknown_tool"
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        return f"{name}({str(parameters)})"
    if not parameters:
        return f"{name}()"
    return f"{name}({json.dumps(parameters, ensure_ascii=False, sort_keys=True)})"


def _extract_browser_env_tool_calls(response: str) -> list[str]:
    parsed = _tool_parser.parse((response or "").strip())
    if parsed.success and parsed.calls:
        return [
            _format_tool_call_for_action_history(call.name, call.parameters)
            for call in parsed.calls
        ]

    tool_calls: list[str] = []
    for match in re.finditer(
        r'\{\s*"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"arguments"\s*:\s*(?P<args>\{.*?\})\s*\}',
        response or "",
        re.DOTALL,
    ):
        name = match.group("name")
        try:
            parameters = json.loads(match.group("args"))
        except json.JSONDecodeError:
            parameters = match.group("args")
        tool_calls.append(_format_tool_call_for_action_history(name, parameters))
    return tool_calls


def _normalize_action_history_text(text: str) -> str:
    return normalize_feedback_text(text)


def _extract_browser_env_action_summary(args: Any, response: str) -> str:
    response = (response or "").strip()
    if not response:
        return ""

    tool_calls = _extract_browser_env_tool_calls(response)

    # Browser-env responses often interleave visible reasoning, malformed
    # closing tags, and raw JSON tool calls in the same text span. For judge
    # action history, the executed tool calls are the most reliable signal.
    if tool_calls:
        return f"[{'; '.join(tool_calls)}]"

    cleaned = _clean_think_from_action_history(response, args)
    cleaned = _clean_img_placeholders_from_action_history(cleaned)
    cleaned = re.sub(r"</?tool_call>", "", cleaned)
    cleaned = re.sub(r"<\|im_end\|>", "", cleaned)
    cleaned = re.sub(r"</think>|<think>|</thinking>|<thinking>", "", cleaned)
    cleaned = re.sub(r'\{\s*"name"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', "", cleaned)
    return _normalize_action_history_text(cleaned)


def _extract_step_environment_feedback(sample: Sample) -> str:
    metadata = sample.metadata or {}
    feedback_parts: list[str] = []
    max_chars = DEFAULT_BROWSER_HISTORY_ENV_FEEDBACK_MAX_CHARS

    tool_responses = metadata.get("step_tool_responses")
    if isinstance(tool_responses, list):
        for entry in tool_responses:
            if not isinstance(entry, dict):
                continue
            tool_name = str(entry.get("tool_name", "")).strip()
            tool_response = truncate_feedback_text(entry.get("tool_response", ""), max_chars)
            if not tool_response:
                continue
            if tool_name:
                feedback_parts.append(f"{tool_name}: {tool_response}")
            else:
                feedback_parts.append(tool_response)

    return " | ".join(part for part in feedback_parts if part)


def _build_turn_level_action_history_for_browser_env(args: Any, samples: list[Sample]) -> str:
    ordered_samples = sorted(samples, key=lambda s: s.metadata.get("turn_index", -1))
    history_lines: list[str] = []

    for idx, sample in enumerate(ordered_samples, start=1):
        turn_index = sample.metadata.get("turn_index")
        display_step = turn_index + 1 if isinstance(turn_index, int) and turn_index >= 0 else idx
        action_summary = _extract_browser_env_action_summary(args, sample.response or "")
        env_feedback = _extract_step_environment_feedback(sample)

        parts = [f"Step {display_step}:"]
        if action_summary:
            parts.append(f"action={action_summary}")
        if env_feedback:
            parts.append(f"environment feedback={env_feedback}")

        if len(parts) > 1:
            history_lines.append(" ".join(parts))

    if not history_lines:
        return "No action history could be extracted from the agent response."
    return "\n".join(history_lines)


def _build_turn_level_action_history_response(samples: list[Sample]) -> str:
    """
    Reconstruct a multi-turn assistant transcript from turn-level samples.

    Each turn-level sample stores only its own assistant response in
    ``sample.response``. To record the action history across the full
    trajectory, we concatenate those responses in turn order using
    ``<|im_start|>assistant`` / ``<|im_end|>`` delimiters.
    """
    ordered_samples = sorted(samples, key=lambda s: s.metadata.get("turn_index", -1))
    assistant_turns: list[str] = []

    for sample in ordered_samples:
        response = (sample.response or "").strip()
        if not response:
            continue
        assistant_turns.append(f"<|im_start|>assistant\n{response}\n<|im_end|>")

    return "\n".join(assistant_turns)


def _normalize_served_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        raise ValueError("Judge served mode requires JUDGE_API_BASE.")
    if not re.match(r"^https?://", base_url):
        base_url = f"http://{base_url}"
    return base_url.rstrip("/")


def _get_openai_client() -> Any:
    """
    Create an AsyncOpenAI client for the served judge from JUDGE_API_BASE / JUDGE_API_KEY.
    """
    base_url = _normalize_served_base_url(os.environ.get("JUDGE_API_BASE"))
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def compute_judge_reward(
    args: Any,
    sample: Sample,
) -> tuple[float, str, bool]:
    """
    Compute LLM-as-a-Judge reward using an OpenAI-compatible API.

    Builds a multimodal evaluation prompt with the task instruction,
    the agent's final answer, and (optionally) screenshots, then
    parses SUCCESS / NOT SUCCESS from the judge's response.

    Args:
        args: Evaluation arguments (judge model, timeout, screenshot cap).
        sample: Sample with response and metadata.

    Returns:
        1.0 for SUCCESS, 0.0 for NOT SUCCESS or on error.
    """
    # --- Extract task and answer ---
    task = sample.metadata["intent"]

    answer = _extract_final_answer_browser_env(sample.response or "")
    if not answer:
        logger.warning("No answer extracted from response; judge score = 0.0")
        return 0.0, "No answer extracted.", False

    # --- Collect screenshots ---
    screenshots = _get_judge_screenshots(args, sample)
    num_imgs = len(screenshots)

    # --- Build messages ---
    user_text = JUDGE_USER_PROMPT.format(
        task=task,
        answer=answer,
        num=str(num_imgs) if num_imgs > 0 else "0",
    )

    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for img_b64 in screenshots:
        # Support both raw base64 and data-URL format
        if img_b64.startswith("data:"):
            url = img_b64
        else:
            url = f"data:image/png;base64,{img_b64}"
        user_content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })
    user_content.append({"type": "text", "text": "Your verdict:\n"})

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # --- Call LLM judge ---
    api_model = getattr(args, "judge_api_model")
    judge_timeout_secs = getattr(args, "judge_timeout_secs", 30.0)
    # max_tokens = getattr(args, "judge_max_tokens")
    # temperature = getattr(args, "judge_temperature")

    client = _get_openai_client()

    max_retries = 3
    saw_timeout = False
    # No `seed`: the served endpoint is an arbitrary OpenAI-compatible API that
    # can't be assumed to support it.
    for attempt in range(max_retries):
        try:
            judge_coro = client.chat.completions.create(
                model=api_model,
                messages=messages,
                # max_tokens=max_tokens,
                # temperature=temperature,
            )
            response = await asyncio.wait_for(judge_coro, timeout=judge_timeout_secs)
            judge_text = response.choices[0].message.content or ""
            logger.debug("Judge response: %s", judge_text)

            judge_score = 0.0
            if "NOT SUCCESS" in judge_text:
                judge_score = 0.0
            elif "SUCCESS" in judge_text:
                judge_score = 1.0
            else:
                logger.warning(f"Judge response does not contain SUCCESS/NOT SUCCESS: {judge_text}")
                judge_score = 0.0

            return judge_score, judge_text, False

        except asyncio.TimeoutError as e:
            saw_timeout = True
            logger.warning(
                "Judge API call timed out (attempt %d/%d, timeout=%ss): %s",
                attempt + 1,
                max_retries,
                judge_timeout_secs,
                e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(3 ** (attempt + 1))
            else:
                logger.error("All judge API retry attempts exhausted due to timeout.")
                return 0.0, f"Judge timeout exhausted after {max_retries} attempts.", True
        except Exception as e:
            logger.warning(
                "Judge API call failed (attempt %d/%d, timeout=%ss): %s",
                attempt + 1,
                max_retries,
                judge_timeout_secs,
                e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(3 ** (attempt + 1))
            else:
                logger.error("All judge API retry attempts exhausted.")
                return 0.0, "All judge API retry attempts exhausted.", saw_timeout

    return 0.0, "Unexpected error occurred.", saw_timeout  # Should not reach here


# ==============================================================================
# 3. Main Reward Function
# ==============================================================================

async def reward_func(args: Any, samples: list[Sample]) -> list[float]:
    """
    Main async reward function for browser agent tasks (batch mode).

    Called by run_evaluate with the list of samples one task's generator returned.

    Combines two reward signals: the combined reward is 1.0 only if the format
    score is nonzero and the judge says SUCCESS.

    Handles both turn-level and trajectory-level samples:
    - Turn-level: sample.response is a single assistant turn, and
      sample.metadata contains 'turn_index' and 'is_last_turn'.
      For turn-level samples, the reward is computed only on the last turn
      and then propagated to all turns in the trajectory.
    - Otherwise (the generator produced no turn and returned the input
      sample): each sample is scored on its own.

    The judge runs only on samples whose status is COMPLETED (agent called
    done()); any other status scores 0.

    Args:
        args: Evaluation arguments.
        samples: List of Sample objects with response and metadata.

    Returns:
        List of combined rewards (one float per sample).
    """
    # For turn-level samples (from generate_turn_sample), compute the reward
    # only on the last turn and share it across all turns in the trajectory.
    is_turn_level = any("turn_index" in s.metadata for s in samples)
    if is_turn_level:
        # Find the last turn sample
        last_turn_sample = None
        for s in samples:
            if s.metadata.get("is_last_turn", False):
                last_turn_sample = s
                break
        if last_turn_sample is None:
            raise ValueError("No turn sample with is_last_turn=True found in turn-level samples.")

        judge_max_attached_imgs = getattr(args, "judge_max_attached_imgs", 0) or 0 
        context_num_screenshots = getattr(args, "context_num_screenshots", 0) or 0 
        max_judge_imgs = max(judge_max_attached_imgs, context_num_screenshots)
        aggregated_judge_images = _collect_recent_trajectory_images(samples, max_judge_imgs)
        if aggregated_judge_images:
            if last_turn_sample.multimodal_inputs is None:
                last_turn_sample.multimodal_inputs = {}
            last_turn_sample.multimodal_inputs["judge_images"] = aggregated_judge_images

        trajectory_action_history_response = _build_turn_level_action_history_response(samples)
        if trajectory_action_history_response:
            last_turn_sample.metadata["trajectory_action_history_response"] = trajectory_action_history_response
        trajectory_action_history_text = _build_turn_level_action_history_for_browser_env(args, samples)
        if trajectory_action_history_text:
            last_turn_sample.metadata["trajectory_action_history_text"] = trajectory_action_history_text

        # Compute reward once on the last turn
        trajectory_reward = await _score_single_sample(args, last_turn_sample)
        # Propagate the same reward and reward metadata to all turns
        reward_meta = last_turn_sample.metadata.get("reward", {})
        for s in samples:
            s.metadata["reward"] = reward_meta
            if last_turn_sample.remove_sample:
                s.remove_sample = True
        return [trajectory_reward] * len(samples)

    tasks = [_score_single_sample(args, s) for s in samples]
    return await asyncio.gather(*tasks)


async def _score_single_sample(args: Any, sample: Sample) -> float:
    """Score a single sample (called in parallel by reward_func)."""
    if not isinstance(sample, Sample):
        raise TypeError(f"Expected Sample, got {type(sample)}")

    async with _reward_semaphore:
        # # Determine status from metadata (normalise to lowercase string)
        # raw_status = sample.metadata.get("status", "completed")
        # if isinstance(raw_status, str):
        #     # Handle "Status.ABORTED" → "aborted" format from data files
        #     status = raw_status.rsplit(".", 1)[-1].lower()
        # else:
        #     status = str(raw_status).lower()

        is_turn_level = True if "turn_index" in sample.metadata else False

        # --- Format reward (deterministic) ---
        format_score = compute_format_reward(args, sample.response, turn_level=is_turn_level)
    
        # --- LLM-as-a-Judge reward (async) ---
        # Only run judge on completed episodes (last turn with done() call)
        # or when is_last_turn is True.  For non-final turns, judge is 0.
        judge_timeout = False
        if sample.status != Sample.Status.COMPLETED:
            judge_score = 0.0
            judge_text = f"Judge not run for status={sample.status}"
        else:
            judge_score, judge_text, judge_timeout = await compute_judge_reward(args, sample)
            if judge_timeout:
                sample.remove_sample = True
                sample.metadata["judge_timeout"] = True

        # --- Combine ---
        if format_score == 0.0:
            combined = 0.0
        else:
            combined = 1.0 if judge_score == 1.0 else 0.0

        logger.info(
            "Browser reward: format=%.3f judge=%.3f combined=%.3f (status=%s)",
            format_score, judge_score, combined, sample.status
        )

        sample.metadata["reward"] = {
            "format": format_score,
            "judge": judge_score,
            "combined": combined,
            "judge_text": judge_text,
            "judge_timeout": judge_timeout,
            "judge_prompt_variant": "default",
        }
        return combined
