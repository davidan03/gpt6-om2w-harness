from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrowserResponseFormatMode:
    name: str
    policy_relpath: str
    thinking_open_tag: str
    thinking_close_tag: str


_BROWSER_RESPONSE_FORMAT_MODES: dict[str, BrowserResponseFormatMode] = {
    "browser_env": BrowserResponseFormatMode(
        name="browser_env",
        policy_relpath="env/prompts/system_prompt_browser_env.md",
        thinking_open_tag="<think>",
        thinking_close_tag="</think>",
    ),
}


def get_browser_response_format_mode(mode_name: str | None) -> BrowserResponseFormatMode:
    normalized = (mode_name or "").strip().lower().replace("-", "_")
    try:
        mode = _BROWSER_RESPONSE_FORMAT_MODES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_BROWSER_RESPONSE_FORMAT_MODES))
        raise ValueError(
            f"Unsupported browser_response_format_mode={mode_name!r}. Supported modes: {supported}."
        ) from exc
    return mode


def rewrite_policy_thinking_tags(
    policy: str,
    mode: BrowserResponseFormatMode,
) -> str:
    """
    Rewrite prompt text to use the currently configured thinking tags.
    """
    if not policy:
        return policy
    rewritten = policy
    rewritten = rewritten.replace("</thinking>", mode.thinking_close_tag)
    rewritten = rewritten.replace("</think>", mode.thinking_close_tag)
    rewritten = rewritten.replace("<thinking>", mode.thinking_open_tag)
    rewritten = rewritten.replace("<think>", mode.thinking_open_tag)
    return rewritten
