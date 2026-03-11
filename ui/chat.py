"""
QuantTrade chat backend.

Provides:
  - build_system_prompt()  — assembles page context into a system message
  - ask()                  — calls GPT-4.1 with Claude Sonnet 4.6 as fallback

Environment variables required (at least one pair):
  OPENAI_API_KEY    — for GPT-4.1 (primary)
  ANTHROPIC_API_KEY — for Claude Sonnet 4.6 (fallback)

Models used:
  Primary  : gpt-4.1           (OpenAI)
  Fallback : claude-sonnet-4-6 (Anthropic)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
OPENAI_MODEL = "gpt-4.1"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Maximum tokens to spend on the injected context block
_MAX_CONTEXT_CHARS = 12_000


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    daily_summary: Optional[Dict] = None,
    persistence_state: Optional[Dict] = None,
    daily_brief: str = "",
    selected_day: str = "",
) -> str:
    """
    Assemble a system prompt that gives the LLM all the page context it needs
    to answer user questions about today's signals.
    """
    lines: List[str] = [
        "You are QuantTrade Assistant, an expert quantitative trading analyst.",
        "You help the user understand satellite-data-driven trading signals.",
        "Answer in the same language the user writes in (Chinese or English).",
        "Be concise and precise. When referencing numbers, quote them exactly.",
        "Do NOT invent signals or data that are not in the context below.",
        "",
        "== CONTEXT START ==",
    ]

    if selected_day:
        lines.append(f"Today's date: {selected_day}")

    # Daily summary — signals section
    if daily_summary:
        signals = daily_summary.get("signals", {})
        if signals:
            lines.append("\n--- Active Signals ---")
            for region_id, sig in signals.items():
                action = sig.get("trading_action", "FLAT")
                raw = sig.get("raw_trading_action", action)
                confidence = sig.get("confidence", "Low")
                actionability = sig.get("actionability", "Ignore")
                signal_text = sig.get("signal", "")
                instruments = ", ".join(sig.get("instruments", []))
                score = sig.get("vote_score") or sig.get("ndvi_change", "n/a")
                lines.append(
                    f"  {region_id}: action={action} raw={raw} "
                    f"confidence={confidence} actionability={actionability} "
                    f"instruments=[{instruments}] signal='{signal_text}' score={score}"
                )

        meta = {
            "regions_processed": daily_summary.get("regions_processed"),
            "regions_successful": daily_summary.get("regions_successful"),
            "date": daily_summary.get("date"),
        }
        lines.append(f"\n--- Summary Meta ---\n  {json.dumps(meta)}")

    # Persistence / confirmation state
    if persistence_state:
        lines.append("\n--- Signal Persistence State ---")
        for key, val in persistence_state.items():
            lines.append(f"  {key}: {json.dumps(val)}")

    # Daily brief (trimmed to avoid blowing the context window)
    if daily_brief:
        trimmed = daily_brief[:_MAX_CONTEXT_CHARS]
        if len(daily_brief) > _MAX_CONTEXT_CHARS:
            trimmed += "\n[... truncated ...]"
        lines.append(f"\n--- Daily Chinese Brief ---\n{trimmed}")

    lines.append("\n== CONTEXT END ==")
    lines.append(
        "\nIf the user asks about how to trade on Fidelity, explain which ticker to search "
        "and whether to click Buy or use an inverse ETF for short signals. "
        "For beginners highlight that short positions require a margin account."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call — GPT-4.1 primary, Claude Sonnet 4.6 fallback
# ---------------------------------------------------------------------------

def ask(
    messages: List[Dict[str, str]],
    system_prompt: str,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Send a conversation to the LLM and return a dict:
      {"role": "assistant", "content": "<reply>", "model": "<model used>"}

    `messages` is the full history list:
      [{"role": "user"|"assistant", "content": "..."}, ...]

    Tries GPT-4.1 first; falls back to Claude Sonnet 4.6 on any error.
    """
    oai_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
    ant_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")

    last_error: Optional[Exception] = None

    # -- Primary: OpenAI GPT-4.1 --
    if oai_key:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=oai_key)
            oai_messages = [{"role": "system", "content": system_prompt}] + messages
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=oai_messages,  # type: ignore[arg-type]
                max_tokens=1024,
                temperature=0.3,
            )
            content = resp.choices[0].message.content or ""
            return {"role": "assistant", "content": content, "model": OPENAI_MODEL}
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    # -- Fallback: Anthropic Claude Sonnet 4.6 --
    if ant_key:
        try:
            import anthropic as _anthropic  # type: ignore
            client = _anthropic.Anthropic(api_key=ant_key)
            ant_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
            ]
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                system=system_prompt,
                messages=ant_messages,
                max_tokens=1024,
            )
            content = resp.content[0].text if resp.content else ""
            return {"role": "assistant", "content": content, "model": ANTHROPIC_MODEL}
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    # -- No keys configured --
    if last_error:
        return {
            "role": "assistant",
            "content": (
                f"LLM call failed: {last_error}\n\n"
                "Please set OPENAI_API_KEY or ANTHROPIC_API_KEY."
            ),
            "model": "error",
        }

    return {
        "role": "assistant",
        "content": (
            "No LLM API key configured. "
            "Set OPENAI_API_KEY (for GPT-4.1) or ANTHROPIC_API_KEY (for Claude Sonnet 4.6) "
            "in your environment or the sidebar."
        ),
        "model": "none",
    }
