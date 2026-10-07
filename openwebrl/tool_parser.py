"""Extract `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks from a rendered agent response."""
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """Represents a parsed tool/function call."""
    name: str
    parameters: dict[str, Any]


@dataclass
class ParsedToolResult:
    """Result of parsing a tool call from LLM response."""
    success: bool
    normal_text: str
    calls: list[ToolCall]
    error: str | None = None


class ToolParser:
    """Parser for extracting tool calls from LLM responses."""

    def parse(self, response: str) -> ParsedToolResult:
        """Parse <tool_call>{...}</tool_call> blocks from a response."""
        # JSON tool calls: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = re.findall(tool_call_pattern, response, re.DOTALL)
        
        calls = []
        if matches:
            try:
                for match in matches:
                    tool_data = json.loads(match)
                    if "name" in tool_data:
                        calls.append(ToolCall(
                            name=tool_data["name"],
                            parameters=tool_data.get("arguments", tool_data.get("parameters", {})),
                        ))
                
                # Extract text before tool call
                text_before = re.split(tool_call_pattern, response)[0].strip()
                return ParsedToolResult(
                    success=True,
                    normal_text=text_before,
                    calls=calls,
                )
            except json.JSONDecodeError as e:
                return ParsedToolResult(
                    success=False,
                    normal_text=response,
                    calls=[],
                    error=f"JSON decode error: {e}",
                )
        
        # No tool call found, treat entire response as normal text
        return ParsedToolResult(
            success=True,
            normal_text=response.strip(),
            calls=[],
        )
