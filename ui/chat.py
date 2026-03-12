"""
QuantTrade chat backend.

Provides:
  - build_system_prompt()  — assembles page context into a system message
  - ask()                  — calls ChatGPT Plus (web) → GPT-4.1 API → Claude Sonnet 4.6 fallback

Authentication options (at least one required):
  1. ChatGPT Plus access token (free, uses your subscription)
  2. OPENAI_API_KEY    — for GPT-4.1 API (paid)
  3. ANTHROPIC_API_KEY — for Claude Sonnet 4.6 (paid fallback)

Models used:
  Primary  : ChatGPT Plus (gpt-4o / gpt-4.1 via web backend)
  Fallback : gpt-4.1 (OpenAI API) → claude-sonnet-4-6 (Anthropic)
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
CHATGPT_WEB_MODEL = "gpt-4o"
OPENAI_MODEL = "gpt-4.1"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# ChatGPT web API endpoints
CHATGPT_SESSION_URL = "https://chat.openai.com/api/auth/session"
CHATGPT_CONVERSATION_URL = "https://chat.openai.com/backend-api/conversation"

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
# ChatGPT Plus (web) backend — uses your subscription, no API key needed
# ---------------------------------------------------------------------------

def get_chatgpt_access_token(session_token: str) -> Optional[str]:
    """
    Exchange a ChatGPT session token (__Secure-next-auth.session-token cookie)
    for an access token by calling the session endpoint.
    
    Returns the access token string, or None on failure.
    """
    try:
        cookies = {"__Secure-next-auth.session-token": session_token}
        resp = requests.get(CHATGPT_SESSION_URL, cookies=cookies, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("accessToken")
    except Exception:  # noqa: BLE001
        return None


def ask_chatgpt_web(
    messages: List[Dict[str, str]],
    system_prompt: str,
    access_token: str,
    model: str = CHATGPT_WEB_MODEL,
) -> Dict[str, str]:
    """
    Send a conversation to ChatGPT web backend using an OAuth access token.
    
    This uses your ChatGPT Plus subscription — no API costs.
    
    Args:
        messages: Conversation history [{"role": "user/assistant", "content": "..."}]
        system_prompt: System context injected at the start
        access_token: OAuth access token from ChatGPT session
        model: Model to use (gpt-4o, gpt-4, etc.)
    
    Returns:
        {"role": "assistant", "content": "<reply>", "model": "<model>"}
    """
    conversation_id = str(uuid.uuid4())
    parent_message_id = str(uuid.uuid4())
    
    # Build the message content: system prompt + conversation
    formatted_parts = [f"[System Instructions]\n{system_prompt}\n"]
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        formatted_parts.append(f"[{role}]\n{content}\n")
    
    user_content = "\n".join(formatted_parts)
    
    payload = {
        "action": "next",
        "messages": [
            {
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [user_content]},
                "metadata": {},
            }
        ],
        "conversation_id": None,
        "parent_message_id": parent_message_id,
        "model": model,
        "timezone_offset_min": -480,
        "suggestions": [],
        "history_and_training_disabled": False,
        "conversation_mode": {"kind": "primary_assistant"},
        "force_paragen": False,
        "force_rate_limit": False,
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    
    try:
        resp = requests.post(
            CHATGPT_CONVERSATION_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        
        # Parse SSE stream — look for the final [DONE] message with content
        content = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    message = data.get("message", {})
                    if message.get("author", {}).get("role") == "assistant":
                        content_parts = message.get("content", {}).get("parts", [])
                        if content_parts:
                            content = content_parts[0]
                except json.JSONDecodeError:
                    continue
        
        if not content:
            content = "(No response from ChatGPT)"
        
        return {"role": "assistant", "content": content, "model": f"chatgpt-web/{model}"}
    
    except Exception as exc:  # noqa: BLE001
        return {
            "role": "assistant",
            "content": f"ChatGPT web error: {exc}\n\nYour access token may have expired. Refresh it from the browser.",
            "model": "error",
        }


# ---------------------------------------------------------------------------
# LLM call — ChatGPT Plus → GPT-4.1 API → Claude Sonnet 4.6 fallback
# ---------------------------------------------------------------------------

def ask(
    messages: List[Dict[str, str]],
    system_prompt: str,
    chatgpt_access_token: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Send a conversation to the LLM and return a dict:
      {"role": "assistant", "content": "<reply>", "model": "<model used>"}

    `messages` is the full history list:
      [{"role": "user"|"assistant", "content": "..."}, ...]

    Priority:
      1. ChatGPT Plus (web) — uses your subscription, free
      2. GPT-4.1 API — requires OPENAI_API_KEY
      3. Claude Sonnet 4.6 — requires ANTHROPIC_API_KEY
    """
    chatgpt_token = chatgpt_access_token or os.getenv("CHATGPT_ACCESS_TOKEN", "")
    oai_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
    ant_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")

    last_error: Optional[Exception] = None

    # -- Primary: ChatGPT Plus (web) — free, uses subscription --
    if chatgpt_token:
        try:
            return ask_chatgpt_web(
                messages=messages,
                system_prompt=system_prompt,
                access_token=chatgpt_token,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    # -- Fallback 1: OpenAI GPT-4.1 API --
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

    # -- Fallback 2: Anthropic Claude Sonnet 4.6 --
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

    # -- No auth configured --
    if last_error:
        return {
            "role": "assistant",
            "content": (
                f"LLM call failed: {last_error}\n\n"
                "Please provide a ChatGPT access token, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
            ),
            "model": "error",
        }

    return {
        "role": "assistant",
        "content": (
            "No LLM authentication configured.\n\n"
            "**Free option:** Log into ChatGPT in your browser, then:\n"
            "1. Open DevTools (F12) → Application → Cookies → chat.openai.com\n"
            "2. Copy the value of `__Secure-next-auth.session-token`\n"
            "3. Paste it in the 'ChatGPT Access Token' field below\n\n"
            "Or use paid APIs: OPENAI_API_KEY or ANTHROPIC_API_KEY."
        ),
        "model": "none",
    }
