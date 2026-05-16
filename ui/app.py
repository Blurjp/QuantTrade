"""
Local Streamlit UI for browsing daily QuantTrade outputs.

Run with:
  streamlit run ui/app.py
"""

import html
import os
from pathlib import Path
import json
import sys
from typing import Dict, List, Optional, Tuple

_esc = html.escape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

from automation.status import load_region_status
from pipeline.instruments import get_primary_instrument, list_region_instruments
from pipeline.regions import list_regions, resolve_region_output_base
from pipeline.signals import build_monitor_snapshot, latest_region_signal
from pipeline.ui_data import list_available_days, load_day_bundle
from pipeline.price_feed import fetch_price_yahoo, get_prices_for_portfolio
from paper_trading.multi_asset_portfolio import MultiAssetPortfolio
from scripts.rebuild_asset_history import rebuild_asset_history
from ui.chat import ask, build_system_prompt


# ---------------------------------------------------------------------------
# Fidelity ticker mapping
# System instrument names → Fidelity-searchable ticker codes + beginner notes
# ---------------------------------------------------------------------------
FIDELITY_MAP = {
    # Commodities
    "Soybeans": {
        "fidelity_ticker": "SOYB",
        "type": "ETF",
        "long_action": "Buy SOYB",
        "short_action": "Buy SOYB inverse: use WEAT as proxy or avoid (no direct soy inverse ETF)",
        "beginner_note": "在 Fidelity 搜索 SOYB，点 Buy，选 Market Order。做空方向初学者建议暂时不参与。",
        "beginner_short_note": "做空大豆没有简单的反向 ETF。建议初学者在做空信号时跳过或只观察。",
    },
    "Corn": {
        "fidelity_ticker": "CORN",
        "type": "ETF",
        "long_action": "Buy CORN",
        "short_action": "Avoid (no liquid inverse; stand aside)",
        "beginner_note": "在 Fidelity 搜索 CORN，点 Buy，选 Market Order。",
        "beginner_short_note": "做空玉米没有简单反向 ETF，建议初学者在做空信号时观望。",
    },
    # Energy
    "WTI": {
        "fidelity_ticker": "USO",
        "type": "ETF",
        "long_action": "Buy USO (or XLE for sector exposure)",
        "short_action": "Buy SCO (2× inverse crude) — small size only",
        "beginner_note": "在 Fidelity 搜索 USO，点 Buy，选 Market Order。能源板块也可以用 XLE。",
        "beginner_short_note": "做空原油可以搜 SCO（反向2倍），但波动很大，只用极小仓位试。",
    },
    "Brent": {
        "fidelity_ticker": "BNO",
        "type": "ETF",
        "long_action": "Buy BNO",
        "short_action": "Buy SCO as crude proxy inverse — small size only",
        "beginner_note": "在 Fidelity 搜索 BNO，点 Buy，选 Market Order。",
        "beginner_short_note": "布伦特无直接反向 ETF，做空信号时可参考 SCO，仓位需极轻。",
    },
    # US Retail
    "US retail": {
        "fidelity_ticker": "XRT",
        "type": "ETF",
        "long_action": "Buy XRT",
        "short_action": "Buy SZK (inverse consumer discretionary) or avoid",
        "beginner_note": "在 Fidelity 搜索 XRT，点 Buy，选 Market Order。",
        "beginner_short_note": "做空零售可以搜 SZK，但流动性一般，建议初学者观望。",
    },
    # Individual stocks
    "WMT": {
        "fidelity_ticker": "WMT",
        "type": "Stock",
        "long_action": "Buy WMT",
        "short_action": "Avoid short for beginners (requires margin)",
        "beginner_note": "在 Fidelity 搜索 WMT（沃尔玛），点 Buy，选 Market Order。",
        "beginner_short_note": "股票做空需要保证金账户，初学者建议不参与。",
    },
    "COST": {
        "fidelity_ticker": "COST",
        "type": "Stock",
        "long_action": "Buy COST",
        "short_action": "Avoid short for beginners (requires margin)",
        "beginner_note": "在 Fidelity 搜索 COST（好市多），点 Buy，选 Market Order。",
        "beginner_short_note": "股票做空需要保证金账户，初学者建议不参与。",
    },
    "F": {
        "fidelity_ticker": "F",
        "type": "Stock",
        "long_action": "Buy F (Ford)",
        "short_action": "Avoid short for beginners (requires margin)",
        "beginner_note": "在 Fidelity 搜索 F（福特），点 Buy，选 Market Order。",
        "beginner_short_note": "股票做空需要保证金账户，初学者建议不参与。",
    },
    "GM": {
        "fidelity_ticker": "GM",
        "type": "Stock",
        "long_action": "Buy GM (General Motors)",
        "short_action": "Avoid short for beginners (requires margin)",
        "beginner_note": "在 Fidelity 搜索 GM（通用汽车），点 Buy，选 Market Order。",
        "beginner_short_note": "股票做空需要保证金账户，初学者建议不参与。",
    },
    "FDX": {
        "fidelity_ticker": "FDX",
        "type": "Stock",
        "long_action": "Buy FDX (FedEx)",
        "short_action": "Avoid short for beginners (requires margin)",
        "beginner_note": "在 Fidelity 搜索 FDX（联邦快递），点 Buy，选 Market Order。",
        "beginner_short_note": "股票做空需要保证金账户，初学者建议不参与。",
    },
    "UPS": {
        "fidelity_ticker": "UPS",
        "type": "Stock",
        "long_action": "Buy UPS",
        "short_action": "Avoid short for beginners (requires margin)",
        "beginner_note": "在 Fidelity 搜索 UPS，点 Buy，选 Market Order。",
        "beginner_short_note": "股票做空需要保证金账户，初学者建议不参与。",
    },
}


def _fidelity_info(instrument: str) -> Dict:
    """Return Fidelity mapping for a given instrument name, with safe fallback."""
    return FIDELITY_MAP.get(
        instrument,
        {
            "fidelity_ticker": instrument,
            "type": "Unknown",
            "long_action": f"Search '{instrument}' on Fidelity",
            "short_action": "Avoid short for beginners",
            "beginner_note": f"在 Fidelity 搜索 {instrument}。",
            "beginner_short_note": "做空建议初学者暂时不参与。",
        },
    )


def _format_pct(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def _format_num(value: Optional[float], digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _signal_style(signal: str) -> Tuple[str, str]:
    if signal == "Long disruption risk":
        return "error", "偏多原油风险溢价"
    if signal == "Short disruption risk":
        return "success", "偏空原油风险溢价"
    return "warning", "观望"


def _confidence_note(confidence: str) -> str:
    if confidence == "High":
        return "数据覆盖较好，今天的信号可以看。"
    if confidence == "Medium":
        return "数据可参考，但不要单独作为下单依据。"
    if confidence == "Low":
        return "数据覆盖太差，今天更适合不交易。"
    return "今天没有足够数据。"


def _coverage_confidence(coverage_score: Optional[float]) -> str:
    if coverage_score is None or pd.isna(coverage_score):
        return "Unknown"
    if coverage_score >= 0.75:
        return "High"
    if coverage_score >= 0.55:
        return "Medium"
    return "Low"


def _build_trade_ticket(trade_signal: Optional[Dict], region_instruments: List[Dict]) -> Optional[Dict]:
    if trade_signal is None:
        return None

    primary = region_instruments[0] if region_instruments else None
    ticker = primary["ticker"] if primary else "n/a"
    signal = trade_signal["signal"]
    confidence = trade_signal["confidence"]
    coverage = trade_signal.get("coverage_score") or 0.0
    baseline = trade_signal.get("baseline_value")
    throughput = trade_signal.get("throughput_index_corrected")
    signal_strength = trade_signal.get("signal_strength") or 0.0

    position_size = "No Trade"
    if trade_signal.get("actionability") == "Actionable":
        position_size = "Medium" if confidence == "High" else "Small"
    elif trade_signal.get("actionability") == "Watchlist":
        position_size = "Starter / Watchlist" if confidence != "Low" else "No Trade"

    entry = "Wait"
    exit_rule = "Stay flat until a new actionable signal appears."
    stop_loss = "No position."
    take_profit = "No position."

    if signal == "Long disruption risk":
        primary_trade = f"Long {ticker}"
        direction = "做多"
        entry = "Enter next market open if signal is still active."
        exit_rule = "Exit when signal downgrades to `No trade` or `Short disruption risk`."
        stop_loss = "Tight stop if confidence is Medium; wider stop only if confidence is High."
        take_profit = "Scale out when signal weakens or after 2-3 strong up days in the trade asset."
        invalidation = (
            "If tomorrow the signal drops to `No trade`, confidence falls to `Low`, "
            "or throughput mean-reverts back above the 7-day baseline."
        )
    elif signal == "Short disruption risk":
        primary_trade = f"Short {ticker}"
        direction = "做空"
        entry = "Enter next market open if signal is still active."
        exit_rule = "Exit when signal downgrades to `No trade` or `Long disruption risk`."
        stop_loss = "Tight stop if confidence is Medium; wider stop only if confidence is High."
        take_profit = "Scale out when signal weakens or after 2-3 strong down days in the trade asset."
        invalidation = (
            "If tomorrow the signal drops to `No trade`, confidence falls to `Low`, "
            "or throughput reverses back below the 7-day baseline."
        )
    else:
        primary_trade = "No Trade"
        direction = "观望"
        invalidation = "Only act if the signal upgrades to `Actionable` with at least `Medium` confidence."

    why = (
        f"{signal} | confidence={confidence} | coverage={_format_pct(coverage)} | "
        f"value={_format_num(throughput)} vs baseline={_format_num(baseline)}"
    )
    if trade_signal.get("reroute_flag"):
        why += " | reroute risk detected"

    if coverage is not None and coverage < 0.55:
        position_size = "No Trade"
        primary_trade = "No Trade"
        direction = "观望"
        entry = "Do not enter."
        exit_rule = "No position."
        stop_loss = "No position."
        take_profit = "No position."
        invalidation = "Coverage is too low; wait for a higher-quality day."

    if position_size == "Medium" and signal_strength >= 1.5:
        stop_loss = "Use a standard daily risk stop; do not allow a full thesis reversal."
        take_profit = "Take partial profits into a strong move; trail the rest while the signal remains active."
    elif position_size == "Small":
        stop_loss = "Use a tighter stop; this is a lower-conviction trade."
        take_profit = "Take profits quickly if the asset reacts before signal confidence improves."

    return {
        "primary_trade": primary_trade,
        "direction": direction,
        "position_size": position_size,
        "why": why,
        "entry": entry,
        "exit_rule": exit_rule,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "invalidation": invalidation,
        "ticker": ticker,
    }


def _load_remote_bundle(api_base: str, selected_day: str, output_base: str, region_id: str) -> dict:
    params = {"output_base": output_base, "region": region_id}
    bundle = requests.get(f"{api_base.rstrip('/')}/days/{selected_day}", params=params, timeout=30)
    bundle.raise_for_status()
    payload = bundle.json()

    calibration = requests.get(f"{api_base.rstrip('/')}/calibration", params=params, timeout=30)
    calibration.raise_for_status()
    calibration_payload = calibration.json()

    paths = payload["paths"]
    return {
        "paths": {
            "root": Path(paths["root"]),
            "manifest": Path(paths["manifest"]),
            "load_log": Path(paths["load_log"]),
            "detections_parquet": Path(paths["detections_parquet"]),
            "detections_geojson": Path(paths["detections_geojson"]),
            "metrics": Path(paths["metrics"]),
            "qa_html": Path(paths["qa_html"]),
            "run_report": Path(paths["run_report"]),
            "previews": [Path(path) for path in paths["previews"]],
        },
        "metrics": pd.DataFrame(payload["metrics"]),
        "detections": pd.DataFrame(payload["detections"]),
        "load_log": pd.DataFrame(payload["load_log"]),
        "manifest": pd.DataFrame(payload["manifest"]),
        "report": payload["report"],
        "calibration_metrics": pd.DataFrame(calibration_payload["metrics"]),
        "calibration_report": calibration_payload["report"],
    }
def _render_global_monitor(st, regions: List[Dict], output_base: str) -> None:
    # Try scheduler API first (for Railway where web has no local files)
    monitor_df = pd.DataFrame()
    api_signals = _fetch_from_scheduler("/api/all-signals")
    if api_signals and api_signals.get("signals"):
        rows = []
        for region_id, sig in api_signals["signals"].items():
            # Filter to only show actionable/watchlist signals to avoid clutter
            actionability = sig.get("actionability", "Ignore")
            rows.append({
                "region": region_id,
                "region_name": sig.get("region_name", region_id),
                "date": sig.get("date"),
                "signal": sig.get("signal", "No data"),
                "direction": sig.get("direction", "neutral"),
                "confidence": sig.get("confidence", "Unknown"),
                "confidence_score": sig.get("confidence_score", 0),
                "actionability": actionability,
                "coverage_score": sig.get("coverage_score"),
                "signal_strength": sig.get("signal_strength"),
                "primary_instrument": ",".join(sig.get("instruments", [])[:3]),
                "run_status": "ok",
                "reroute_flag": sig.get("reroute_flag", False),
                "rationale": sig.get("rationale", ""),
                "signal_type": sig.get("signal_type", ""),
            })
        monitor_df = pd.DataFrame(rows)
    
    if monitor_df.empty:
        monitor_df = build_monitor_snapshot(output_base=output_base)
    
    if monitor_df.empty:
        st.info("No regional runs available yet.")
        return

    # Add run_status and region_name from configured regions (for local data)
    if "run_status" not in monitor_df.columns or monitor_df["run_status"].isna().all():
        status_map = {region["id"]: load_region_status(output_base, region["id"]) for region in regions}
        monitor_df["run_status"] = monitor_df["region"].map(lambda region_id: status_map.get(region_id, {}).get("run_status", "unknown"))
    if "last_run_at" not in monitor_df.columns or monitor_df.get("last_run_at", pd.Series()).isna().all():
        status_map2 = {region["id"]: load_region_status(output_base, region["id"]) for region in regions}
        monitor_df["last_run_at"] = monitor_df["region"].map(lambda region_id: status_map2.get(region_id, {}).get("last_run_at"))
    if "region_name" not in monitor_df.columns:
        monitor_df["region_name"] = monitor_df["region"].map({region["id"]: region["name"] for region in regions})
    if "primary_instrument" not in monitor_df.columns:
        monitor_df["primary_instrument"] = monitor_df["region"].apply(
            lambda region_id: (get_primary_instrument(region_id) or {}).get("ticker", "n/a")
        )
    action_rank = {"Actionable": 0, "Watchlist": 1, "Ignore": 2}
    confidence_rank = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
    monitor_df["action_rank"] = monitor_df["actionability"].map(action_rank).fillna(9)
    monitor_df["confidence_rank"] = monitor_df["confidence"].map(confidence_rank).fillna(9)
    monitor_df = monitor_df.sort_values(["action_rank", "confidence_rank", "region"]).drop(columns=["action_rank", "confidence_rank"])

    summary = {
        "actionable": int((monitor_df["actionability"] == "Actionable").sum()),
        "watchlist": int((monitor_df["actionability"] == "Watchlist").sum()),
        "ignore": int((monitor_df["actionability"] == "Ignore").sum()),
    }

    cols = st.columns(3)
    cols[0].metric("Actionable", summary["actionable"])
    cols[1].metric("Watchlist", summary["watchlist"])
    cols[2].metric("Ignore", summary["ignore"])

    st.markdown("**Global Monitor**")
    st.caption("Latest satellite signals ranked by actionability and confidence.")
    
    # Add filter
    show_filter = st.selectbox("Filter", ["All", "Actionable", "Watchlist", "Actionable + Watchlist"], index=3)
    filtered_df = monitor_df.copy()
    if show_filter == "Actionable":
        filtered_df = filtered_df[filtered_df["actionability"] == "Actionable"]
    elif show_filter == "Watchlist":
        filtered_df = filtered_df[filtered_df["actionability"] == "Watchlist"]
    elif show_filter == "Actionable + Watchlist":
        filtered_df = filtered_df[filtered_df["actionability"].isin(["Actionable", "Watchlist"])]
    
    display_cols = {
        "region": filtered_df["region_name"],
        "type": filtered_df.get("signal_type", pd.Series([""]*len(filtered_df))),
        "date": filtered_df["date"],
        "signal": filtered_df["signal"],
        "direction": filtered_df.get("direction", pd.Series([""]*len(filtered_df))),
        "confidence": filtered_df["confidence"],
        "actionability": filtered_df["actionability"],
        "instruments": filtered_df.get("primary_instrument", pd.Series([""]*len(filtered_df))),
    }
    display_df = pd.DataFrame(display_cols)
    st.dataframe(display_df, use_container_width=True)


def _load_latest_backtests(output_base: str, region_id: str) -> List[Dict]:
    backtest_root = Path(output_base) / "regions" / region_id / "backtests"
    if not backtest_root.exists():
        return []

    summaries = []
    for summary_path in sorted(backtest_root.glob("*/*_summary.json")):
        payload = json.loads(summary_path.read_text())
        payload["summary_path"] = str(summary_path)
        summaries.append(payload)
    return summaries


def _render_summary_header(
    st,
    selected_day: str,
    trade_signal: Optional[Dict],
    metrics_row: Dict,
    summary_day: Optional[str] = None,
) -> None:
    primary_day = summary_day or selected_day
    st.subheader(f"交易结论 | {primary_day}")
    if summary_day and summary_day != selected_day:
        st.caption(f"当前区域详情数据日期: {selected_day} | 首页主日期使用全局最新汇总 {summary_day}")

    if trade_signal is None:
        st.warning("当前没有可用的校准信号，暂时不能给出交易结论。")
        return

    level, short_bias = _signal_style(trade_signal["signal"])
    message = (
        f"今日建议: **{trade_signal['signal']}** | "
        f"方向: **{short_bias}** | "
        f"置信度: **{trade_signal['confidence']}**"
    )
    if level == "success":
        st.success(message)
    elif level == "error":
        st.error(message)
    else:
        st.warning(message)

    signal_label = "校准后船流强度" if trade_signal.get("signal_source") == "throughput_index_corrected" else "原始船流强度"
    explain_cols = st.columns(3)
    explain_cols[0].markdown(
        f"**先看这个**\n\n"
        f"- 信号: `{trade_signal['signal']}`\n"
        f"- 偏向: {short_bias}\n"
        f"- 是否值得做: {_confidence_note(trade_signal['confidence'])}"
    )
    explain_cols[1].markdown(
        f"**为什么**\n\n"
        f"- {signal_label}: `{_format_num(trade_signal['throughput_index_corrected'])}`\n"
        f"- 日变化: `{_format_pct(trade_signal['dod_change_pct'])}`\n"
        f"- 7日基准: `{_format_num(trade_signal['rolling_mean_7'])}`"
    )
    explain_cols[2].markdown(
        f"**数据质量**\n\n"
        f"- 覆盖率: `{_format_pct(trade_signal['coverage_score'])}`\n"
        f"- 当天场景数: `{metrics_row.get('num_scenes', 0)}`\n"
        f"- 成功加载: `{metrics_row.get('loaded_scenes', 0)}`"
    )


def _render_trade_ticket(st, trade_ticket: Optional[Dict]) -> None:
    st.markdown("**交易指令页**")
    if trade_ticket is None:
        st.info("今天没有足够信号生成交易指令。")
        return

    ticket_cols = st.columns(4)
    ticket_cols[0].metric("Primary Trade", trade_ticket["primary_trade"])
    ticket_cols[1].metric("Direction", trade_ticket["direction"])
    ticket_cols[2].metric("Position Size", trade_ticket["position_size"])
    ticket_cols[3].metric("Ticker", trade_ticket["ticker"])

    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.markdown("**Why**")
        st.write(trade_ticket["why"])
    with detail_cols[1]:
        st.markdown("**Entry / Exit**")
        st.write(f"Entry: {trade_ticket['entry']}")
        st.write(f"Exit: {trade_ticket['exit_rule']}")
    with detail_cols[2]:
        st.markdown("**Risk Rules**")
        st.write(f"Stop-loss: {trade_ticket['stop_loss']}")
        st.write(f"Take-profit: {trade_ticket['take_profit']}")

    st.markdown("**Invalidation**")
    st.write(trade_ticket["invalidation"])


def _render_how_to_use(st) -> None:
    with st.expander("这个页面该怎么用", expanded=True):
        st.markdown(
            """
1. 先看顶部的 `今日建议`，只需要理解它是 `做多风险`、`做空风险` 还是 `观望`。
2. 再看 `置信度`，如果是 `Low`，默认当成不交易日。
3. 看 `为什么`，重点只看三个数: 校准后船流强度、日变化、7日基准。
4. 最后去 `Trading` 页看历史信号表，确认这不是单日噪音。

这个工具不是自动下单器，而是把“霍尔木兹船流变化”翻译成一个更容易交易的风险信号。
            """.strip()
        )


def _load_daily_brief(output_base: str, selected_day: str) -> str:
    brief_path = Path(output_base) / selected_day / "daily_brief_zh.md"
    if not brief_path.exists():
        return ""
    return brief_path.read_text()


# Scheduler API URL for fetching data from Railway
SCHEDULER_API_URL = os.environ.get("SCHEDULER_API_URL", "https://scheduler-production-b60f.up.railway.app")


def _fetch_from_scheduler(endpoint: str, timeout: int = 10) -> Optional[Dict]:
    """Fetch data from scheduler API (used when running on Railway)."""
    try:
        url = f"{SCHEDULER_API_URL}{endpoint}"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _load_persistence_state(output_base: str) -> dict:
    # Try scheduler API first
    state_path = Path(output_base) / "signal_persistence_state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text())


def _load_daily_summary(output_base: str, selected_day: str) -> Dict:
    # Try scheduler API first (for Railway deployment)
    api_data = _fetch_from_scheduler("/api/summary")
    if api_data and not api_data.get("error"):
        return api_data

    # Fallback to local files
    summary_path = Path(output_base) / selected_day / "daily_summary.json"
    if not summary_path.exists():
        output_root = Path(output_base)
        candidates = sorted(
            path for path in output_root.glob("*/daily_summary.json")
            if path.parent.name[:4].isdigit()
        )
        if candidates:
            return json.loads(candidates[-1].read_text())
        return {}
    return json.loads(summary_path.read_text())


def _resolve_summary_day(output_base: str, selected_day: str) -> str:
    # Try scheduler API first (for Railway deployment)
    api_data = _fetch_from_scheduler("/api/summary")
    if api_data and api_data.get("date"):
        return api_data["date"]

    # Fallback to local files
    summary_path = Path(output_base) / selected_day / "daily_summary.json"
    if summary_path.exists():
        return selected_day

    output_root = Path(output_base)
    candidates = sorted(
        path.parent.name for path in output_root.glob("*/daily_summary.json")
        if path.parent.name[:4].isdigit()
    )
    return candidates[-1] if candidates else selected_day


def _parse_brief_sections(brief_text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = ""
    for raw_line in brief_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current = line[3:]
            sections[current] = []
            continue
        if current and line:
            sections[current].append(line)
    return sections


def _action_theme(action: str) -> Tuple[str, str]:
    if action == "LONG":
        return "#dff6e8", "#0d6b3c"
    if action == "SHORT":
        return "#fde7e7", "#9f1d1d"
    return "#f3efe3", "#6b5a2a"


def _render_signal_card(st, region_id: str, signal: Dict) -> None:
    bg, accent = _action_theme(signal.get("trading_action", "FLAT"))
    score = signal.get("vote_score")
    if score is None:
        score = signal.get("ndvi_change", "")
    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
    instruments = ", ".join(signal.get("instruments", [])) or "n/a"
    st.markdown(
        f"""
        <div style="background:{bg}; border-left:6px solid {accent}; padding:16px 18px; border-radius:14px; min-height:180px;">
          <div style="font-size:12px; letter-spacing:.08em; color:{accent}; font-weight:700;">{_esc(region_id)}</div>
          <div style="font-size:30px; font-weight:800; color:#1f2a2e; margin-top:6px;">{_esc(str(signal.get('trading_action', 'FLAT')))}</div>
          <div style="font-size:14px; color:#44525a; margin-top:4px;">{_esc(str(signal.get('confidence', 'Low')))} / {_esc(str(signal.get('actionability', 'Ignore')))}</div>
          <div style="font-size:16px; color:#1f2a2e; margin-top:10px; font-weight:600;">{_esc(str(signal.get('signal', 'No data')))}</div>
          <div style="font-size:13px; color:#44525a; margin-top:8px;">标的: {_esc(instruments)}</div>
          <div style="font-size:13px; color:#44525a; margin-top:4px;">score: {score_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _signal_rank(signal: Dict) -> Tuple[int, int, float]:
    actionability_rank = {"Actionable": 0, "Watchlist": 1, "Ignore": 2}
    confidence_rank = {"High": 0, "Medium": 1, "Low": 2}
    score = signal.get("vote_score")
    if score is None:
        score = abs(float(signal.get("ndvi_change", 0.0) or 0.0))
    return (
        actionability_rank.get(signal.get("actionability", "Ignore"), 9),
        confidence_rank.get(signal.get("confidence", "Low"), 9),
        -float(score),
    )


def _build_primary_trade_plan(region_id: str, signal: Dict) -> Dict:
    instruments = signal.get("instruments", [])
    ticker = instruments[0] if instruments else "n/a"
    action = signal.get("trading_action", "FLAT")
    confidence = signal.get("confidence", "Low")
    signal_text = signal.get("signal", "No data")

    if action == "LONG":
        return {
            "region": region_id,
            "ticker": ticker,
            "action": action,
            "entry": "下个交易时段优先找回踩后做多，不追第一根冲高。",
            "risk": "若信号回落到 FLAT 或次日原始方向消失，缩仓或退出。",
            "invalidation": "若系统确认状态丢失，或元信号继续维持观望，则这笔交易失效。",
            "summary": f"{ticker} 偏多，{confidence}，理由: {signal_text}",
        }
    if action == "SHORT":
        return {
            "region": region_id,
            "ticker": ticker,
            "action": action,
            "entry": "下个交易时段优先找反弹衰竭后做空，不追开盘直接跳空。",
            "risk": "若信号回落到 FLAT 或次日原始方向消失，缩仓或退出。",
            "invalidation": "若系统确认状态丢失，或元信号继续维持观望，则这笔交易失效。",
            "summary": f"{ticker} 偏空，{confidence}，理由: {signal_text}",
        }
    return {
        "region": region_id,
        "ticker": ticker,
        "action": action,
        "entry": "今天不主动开仓。",
        "risk": "保持轻仓或空仓，等待更一致的确认。",
        "invalidation": "只有当系统升级到 Actionable 且方向明确时才重新考虑。",
        "summary": f"{ticker} 暂无主交易，理由: {signal_text}",
    }


def _build_instrument_rankings(ranked: List[Tuple[str, Dict]]) -> List[Dict]:
    grouped: Dict[str, Dict] = {}
    confidence_weight = {"High": 1.0, "Medium": 0.6, "Low": 0.25}

    for region_id, signal in ranked:
        instruments = signal.get("instruments", [])
        if not instruments:
            continue
        action = signal.get("trading_action", "FLAT")
        direction = 1.0 if action == "LONG" else -1.0 if action == "SHORT" else 0.0
        weight = confidence_weight.get(signal.get("confidence", "Low"), 0.25)
        for instrument in instruments:
            bucket = grouped.setdefault(
                instrument,
                {"instrument": instrument, "score": 0.0, "regions": [], "longs": 0, "shorts": 0, "flats": 0},
            )
            bucket["score"] += direction * weight
            bucket["regions"].append(f"{region_id}:{action}")
            if action == "LONG":
                bucket["longs"] += 1
            elif action == "SHORT":
                bucket["shorts"] += 1
            else:
                bucket["flats"] += 1

    rows = []
    for instrument, bucket in grouped.items():
        score = bucket["score"]
        if score > 0.35:
            stance = "LONG"
        elif score < -0.35:
            stance = "SHORT"
        else:
            stance = "MIXED"
        rows.append(
            {
                "instrument": instrument,
                "stance": stance,
                "score": round(score, 3),
                "support": ", ".join(bucket["regions"][:4]),
                "longs": bucket["longs"],
                "shorts": bucket["shorts"],
                "flats": bucket["flats"],
            }
        )

    return sorted(rows, key=lambda item: (0 if item["stance"] != "MIXED" else 1, -abs(item["score"])))


def _overall_daily_verdict(actionable: List[Tuple[str, Dict]], instrument_rows: List[Dict]) -> Dict:
    if actionable:
        top_region, top_signal = actionable[0]
        top_instrument = instrument_rows[0]["instrument"] if instrument_rows else ", ".join(top_signal.get("instruments", []))
        return {
            "label": "TRADE",
            "reason": f"今天有明确可执行信号，优先关注 {top_region} / {top_instrument}。",
        }

    if any(row["stance"] != "MIXED" for row in instrument_rows):
        return {
            "label": "WATCH",
            "reason": "今天有一些方向性线索，但还没有形成足够清晰的一致性交易。",
        }

    return {
        "label": "NO TRADE",
        "reason": "今天全局没有形成清晰的一致性优势，优先观望。",
    }


def _beginner_verdict_guide(verdict: Dict, actionable: List[Tuple[str, Dict]], instrument_rows: List[Dict]) -> Dict:
    label = verdict["label"]
    if label == "TRADE":
        top_region, top_signal = actionable[0]
        instrument = ", ".join(top_signal.get("instruments", [])) or top_region
        return {
            "title": "如果你是新手，今天怎么做",
            "steps": [
                f"第一步：只看 `{instrument}`，不要同时做很多标的。",
                f"第二步：方向只按系统给的 `{top_signal.get('trading_action')}` 做，不要自己反着猜。",
                "第三步：等开盘后 15-30 分钟，确认不是一开盘的假突破，再考虑进场。",
                "第四步：第一次仓位只用你原计划仓位的 25%-33%，不要满仓。",
                "第五步：如果当天信号转成 FLAT，或者第二天首页统一结论不再支持这笔交易，就退出。",
            ],
            "warning": "今天虽然可以交易，但这不是满仓信号。先小仓位，先活下来，比赚快钱更重要。",
        }
    if label == "WATCH":
        focused = instrument_rows[0]["instrument"] if instrument_rows else "当前候选标的"
        return {
            "title": "如果你是新手，今天怎么做",
            "steps": [
                f"第一步：把 `{focused}` 放进观察名单，但今天先不要急着下单。",
                "第二步：等下一次日更，看它是否从 WATCH 升级成 TRADE。",
                "第三步：今天最多做笔记，记录你本来想怎么做，但不要真实重仓。",
                "第四步：如果你一定要试，只允许极小试仓，并且要接受它更像练习单，不是高把握单。",
                "第五步：如果你看不懂为什么是 WATCH，那就等，不做就是最好的动作。",
            ],
            "warning": "WATCH 的意思不是马上交易，而是『方向有味道，但还不够成熟』。",
        }
    return {
        "title": "如果你是新手，今天怎么做",
        "steps": [
            "第一步：今天不要主动开新仓。",
            "第二步：如果你手里已经有仓位，优先检查它是不是还被系统支持。",
            "第三步：把今天当成复盘日，看看哪些区域互相打架、为什么系统不给统一方向。",
            "第四步：等下一次日更，不要为了交易而交易。",
            "第五步：如果你今天特别想下单，那通常就是最不该下单的时候。",
        ],
        "warning": "NO TRADE 不是错过机会，而是系统在帮你避免低胜率、低一致性的交易。",
    }


def _professional_verdict_guide(verdict: Dict, actionable: List[Tuple[str, Dict]], instrument_rows: List[Dict]) -> Dict:
    label = verdict["label"]
    if label == "TRADE":
        top_region, top_signal = actionable[0]
        instrument = ", ".join(top_signal.get("instruments", [])) or top_region
        return {
            "title": "专业版说明",
            "points": [
                f"Primary setup: focus on `{instrument}` sourced from `{top_region}`.",
                f"Execution bias: keep to `{top_signal.get('trading_action')}` only; do not fade the system signal.",
                "Entry protocol: wait for opening volatility to settle before initiating risk.",
                "Risk sizing: start with reduced size and scale only if follow-through confirms the thesis.",
                "Invalidation: exit if the signal degrades to FLAT or the next daily update removes support.",
            ],
        }
    if label == "WATCH":
        focused = instrument_rows[0]["instrument"] if instrument_rows else "candidate instrument"
        return {
            "title": "专业版说明",
            "points": [
                f"Current state: `{focused}` shows directional bias but lacks full confirmation.",
                "Execution protocol: observation only unless a later update upgrades the setup.",
                "Sizing rule: if traded at all, treat as exploratory risk, not core risk.",
                "Confirmation trigger: wait for stronger cross-region or cross-signal alignment.",
                "Main mistake to avoid: forcing an early trade before the edge is mature.",
            ],
        }
    return {
        "title": "专业版说明",
        "points": [
            "Portfolio stance: remain flat on new risk.",
            "Use the day for observation, review, and preparation rather than execution.",
            "Lack of alignment means expected edge is not strong enough.",
            "Capital preservation takes priority over activity.",
            "Reassess only after the next daily signal update.",
        ],
    }


def _instrument_playbook(row: Dict) -> Dict:
    recommendation = "Trade" if row["stance"] in {"LONG", "SHORT"} and abs(row["score"]) >= 0.35 else "Watch" if row["stance"] != "MIXED" else "No Trade"
    if recommendation == "Trade":
        direction_cn = "做多" if row["stance"] == "LONG" else "做空"
        return {
            "today": recommendation,
            "how": f"今天只考虑 {direction_cn} `{row['instrument']}`，并用小仓位试单。",
            "entry": "等开盘后波动稍微稳定，再按方向进场，不追第一根大波动。",
            "risk": "如果日内走势明显反着走，或者明天统一结论不再支持它，就退出。",
        }
    if recommendation == "Watch":
        return {
            "today": recommendation,
            "how": f"`{row['instrument']}` 今天有方向感，但还不够强，先观察。",
            "entry": "先不进场，等它升级到更清晰的 Trade。",
            "risk": "最大风险不是错过，而是太早进场。",
        }
    return {
        "today": recommendation,
        "how": f"`{row['instrument']}` 今天没有足够优势，不建议开仓。",
        "entry": "不进场。",
        "risk": "把注意力放到更清晰的标的上。",
    }


def _instrument_playbook_pro(row: Dict) -> Dict:
    recommendation = "Trade" if row["stance"] in {"LONG", "SHORT"} and abs(row["score"]) >= 0.35 else "Watch" if row["stance"] != "MIXED" else "No Trade"
    if recommendation == "Trade":
        direction = "long" if row["stance"] == "LONG" else "short"
        return {
            "today": recommendation,
            "how": f"Bias: {direction} `{row['instrument']}` with controlled initial size.",
            "entry": "Avoid first impulse entries; wait for price acceptance after the open.",
            "risk": "Cut the position if the signal loses support or price action rejects the thesis.",
        }
    if recommendation == "Watch":
        return {
            "today": recommendation,
            "how": f"Directional lean exists in `{row['instrument']}`, but confirmation is incomplete.",
            "entry": "No immediate execution; monitor for upgrade.",
            "risk": "Premature entries are the primary risk.",
        }
    return {
        "today": recommendation,
        "how": f"No sufficient edge in `{row['instrument']}` right now.",
        "entry": "Stand aside.",
        "risk": "Opportunity cost is acceptable; forcing trades is not.",
    }


def _render_ranked_today_board(st, selected_day: str, summary: Dict, output_base: str) -> None:
    st.markdown("## 今日该交易什么")
    if not summary:
        st.info("这个日期没有 daily_summary.json，无法生成全局排序。")
        return

    summary_day = _resolve_summary_day(output_base, selected_day)
    # Don't show the confusing message - just use the latest summary silently

    signals = summary.get("signals", {})
    if not signals:
        st.info("当天没有可用信号。")
        return

    # Separate META signals from sub-regions
    meta_signals = [(rid, sig) for rid, sig in signals.items() if rid.endswith("_meta")]
    sub_signals = [(rid, sig) for rid, sig in signals.items() if not rid.endswith("_meta")]
    
    ranked = sorted(signals.items(), key=lambda item: _signal_rank(item[1]))
    actionable = [(region_id, signal) for region_id, signal in ranked if signal.get("actionability") == "Actionable"]
    pending = [
        (region_id, signal)
        for region_id, signal in ranked
        if signal.get("raw_trading_action") not in {None, "FLAT"} and signal.get("trading_action") == "FLAT"
    ]
    blocked = [
        (region_id, signal)
        for region_id, signal in ranked
        if region_id.endswith("_meta") and signal.get("trading_action") == "FLAT"
    ]

    # =========================================================================
    # ⭐ META SIGNALS - 这是你真正该看的交易结论
    # =========================================================================
    st.markdown(f"### ⭐ META 汇总信号 ({summary_day}) — 这是你真正该看的交易结论")
    st.caption("系统已经把所有子区域信号汇总成这几个 META 信号。**只看这个部分就够了。**")
    
    if meta_signals:
        meta_cols = st.columns(len(meta_signals))
        for idx, (meta_id, meta_sig) in enumerate(sorted(meta_signals, key=lambda x: x[0])):
            with meta_cols[idx]:
                action = meta_sig.get("trading_action", "FLAT")
                confidence = meta_sig.get("confidence", "Low")
                signal_text = meta_sig.get("signal", "No signal")
                instruments = ", ".join(meta_sig.get("instruments", [])) or "n/a"
                raw_action = meta_sig.get("raw_trading_action", action)
                pending_count = meta_sig.get("pending_count", 0)
                
                # 关键：置信度 Low = 暂不交易，只观察
                is_actionable = confidence in {"High", "Medium"}
                
                if not is_actionable:
                    # Low confidence = 观察中，不交易
                    action_color = "#6b5a2a"
                    action_bg = "#f3efe3"
                    if raw_action and raw_action != "FLAT":
                        # 有方向倾向但不确定
                        action_display = "观察中"
                        action_cn = f"倾向{raw_action}，但不确定"
                        action_hint = f"⏳ 等待确认 ({pending_count}/2)"
                    else:
                        action_display = "观望"
                        action_cn = "暂无明确方向"
                        action_hint = "不需要操作"
                elif action == "LONG":
                    action_color = "#0d6b3c"
                    action_bg = "#dff6e8"
                    action_display = "做多"
                    action_cn = "可以交易"
                    action_hint = "✅ 确认完成，可以下单"
                elif action == "SHORT":
                    action_color = "#9f1d1d"
                    action_bg = "#fde7e7"
                    action_display = "做空"
                    action_cn = "可以交易"
                    action_hint = "✅ 确认完成，可以下单"
                else:
                    action_color = "#6b5a2a"
                    action_bg = "#f3efe3"
                    action_display = "观望"
                    action_cn = "暂无明确方向"
                    action_hint = "不需要操作"
                
                st.markdown(
                    f"""
                    <div style="background:{action_bg}; border:3px solid {action_color}; border-radius:12px; padding:16px; text-align:center;">
                        <div style="font-size:14px; color:#666; font-weight:600;">{_esc(str(meta_id))}</div>
                        <div style="font-size:42px; font-weight:900; color:{action_color}; margin:8px 0;">{_esc(str(action_display))}</div>
                        <div style="font-size:16px; color:#333;">{_esc(str(action_cn))}</div>
                        <div style="font-size:13px; color:#666; margin-top:8px;">{_esc(str(action_hint))}</div>
                        <div style="font-size:12px; color:#888; margin-top:4px;">置信度: {_esc(str(confidence))} | 标的: {_esc(str(instruments))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(signal_text)
                
                # 只有 actionable 时才显示 Fidelity 操作指引
                if is_actionable and instruments and action != "FLAT":
                    first_instrument = instruments.split(",")[0].strip()
                    fi = _fidelity_info(first_instrument)
                    is_short = action == "SHORT"
                    fidelity_action = fi["short_action"] if is_short else fi["long_action"]
                    beginner_note = fi["beginner_short_note"] if is_short else fi["beginner_note"]
                    st.info(f"**Fidelity 操作:** {fidelity_action}\n\n{beginner_note}")
                elif not is_actionable and raw_action and raw_action != "FLAT":
                    st.warning(f"⚠️ **暂不交易** — 系统正在观察这个信号，等连续确认 2 天后才会建议交易。")
    else:
        st.warning("没有 META 汇总信号")
    
    st.markdown("---")
    st.markdown("### 📊 以下是研究参考（你不需要看这个）")
    st.caption("下面是各个子区域的原始信号，仅供研究参考。**交易时只看上面的 META 信号。**")
    
    # Continue with the rest of the original logic
    instrument_rows = _build_instrument_rankings(ranked)
    verdict = _overall_daily_verdict(actionable, instrument_rows)
    beginner_guide = _beginner_verdict_guide(verdict, actionable, instrument_rows)
    professional_guide = _professional_verdict_guide(verdict, actionable, instrument_rows)

    brief_sections = _parse_brief_sections(_load_daily_brief(output_base, summary_day))

    verdict_bg = {"TRADE": "#e6f6ea", "WATCH": "#fff3d9", "NO TRADE": "#f4f1ea"}.get(verdict["label"], "#f4f1ea")
    verdict_accent = {"TRADE": "#176b3a", "WATCH": "#9a6500", "NO TRADE": "#6f6555"}.get(verdict["label"], "#6f6555")

    st.markdown(
        f"""
        <div style="background:{verdict_bg}; border:1px solid #d8cfbf; border-radius:18px; padding:18px 22px; margin:8px 0 18px;">
          <div style="font-size:12px; letter-spacing:.12em; color:{verdict_accent}; font-weight:700;">UNIFIED DAILY VERDICT</div>
          <div style="font-size:36px; font-weight:800; color:#1f2a2e; margin-top:8px;">{_esc(str(verdict['label']))}</div>
          <div style="font-size:16px; color:#46545b; margin-top:8px;">{_esc(str(verdict['reason']))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_cols = st.columns(4)
    top_cols[0].metric("今日可交易", len(actionable))
    top_cols[1].metric("确认中", len(pending))
    top_cols[2].metric("系统观望", len(blocked))
    top_cols[3].metric("运行成功", f"{summary.get('regions_successful', 0)}/{summary.get('regions_processed', 0)}")

    if actionable:
        top_region, top_signal = actionable[0]
        instruments = ", ".join(top_signal.get("instruments", [])) or "n/a"
        bg, accent = _action_theme(top_signal.get("trading_action", "FLAT"))
        primary_plan = _build_primary_trade_plan(top_region, top_signal)
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg, {bg}, #fffdf7); border:1px solid #d9cfbb; border-radius:18px; padding:20px 24px; margin:8px 0 18px;">
              <div style="font-size:12px; letter-spacing:.12em; color:{accent}; font-weight:700;">TODAY'S TOP TRADE</div>
              <div style="font-size:34px; font-weight:800; color:#1f2a2e; margin-top:8px;">{_esc(str(top_region))} → {_esc(str(top_signal.get('trading_action')))}</div>
              <div style="font-size:18px; color:#334047; margin-top:6px;">{_esc(str(top_signal.get('signal')))}</div>
              <div style="font-size:14px; color:#5d6970; margin-top:10px;">标的: {_esc(str(instruments))} | 置信度: {_esc(str(top_signal.get('confidence')))} | actionability: {_esc(str(top_signal.get('actionability')))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        detail_cols = st.columns(3)
        detail_cols[0].markdown(f"**主交易摘要**\n\n{primary_plan['summary']}")
        detail_cols[1].markdown(f"**Entry**\n\n{primary_plan['entry']}")
        detail_cols[2].markdown(f"**Risk / Invalidation**\n\n{primary_plan['risk']}\n\n{primary_plan['invalidation']}")

        # Fidelity step-by-step for top trade
        top_instrument_name = top_signal.get("instruments", [""])[0]
        if top_instrument_name:
            fi_top = _fidelity_info(top_instrument_name)
            is_short_top = top_signal.get("trading_action") == "SHORT"
            fidelity_ticker_top = fi_top["fidelity_ticker"]
            fidelity_action_top = fi_top["short_action"] if is_short_top else fi_top["long_action"]
            beginner_cn_top = fi_top["beginner_short_note"] if is_short_top else fi_top["beginner_note"]
            with st.expander(f"如何在 Fidelity 下单 — {top_instrument_name} ({fidelity_ticker_top})", expanded=True):
                st.markdown(
                    f"**Fidelity 代码:** `{fidelity_ticker_top}` ({fi_top['type']})\n\n"
                    f"**操作:** {fidelity_action_top}\n\n"
                    f"**初学者中文步骤：**\n\n"
                    f"1. 登录 Fidelity 账户\n"
                    f"2. 顶部搜索框输入 `{fidelity_ticker_top}`\n"
                    f"3. 点击 **Trade / Buy**\n"
                    f"4. 选 **Market Order**（市价单），输入股数（建议先只买 1 股试手感）\n"
                    f"5. 确认方向后点 **Preview Order → Place Order**\n\n"
                    f"{beginner_cn_top}"
                )
    else:
        st.warning("今天没有明确的一致性系统交易，优先观望。")

    with st.expander("今天怎么交易（专业版 + 直白中文版）", expanded=True):
        guide_cols = st.columns(2)
        with guide_cols[0]:
            st.markdown(f"**{professional_guide['title']}**")
            for idx, point in enumerate(professional_guide["points"], start=1):
                st.write(f"{idx}. {point}")
        with guide_cols[1]:
            st.markdown(f"**{beginner_guide['title']}**")
            for idx, step in enumerate(beginner_guide["steps"], start=1):
                st.write(f"{idx}. {step}")
            st.info(beginner_guide["warning"])

    hero_cols = st.columns(3)
    featured = actionable[:2] + pending[:1]
    if not featured:
        featured = ranked[:3]
    for idx, (region_id, signal) in enumerate(featured[:3]):
        with hero_cols[idx]:
            _render_signal_card(st, region_id, signal)

    if brief_sections:
        st.markdown("**中文简报摘录**")
        brief_cols = st.columns(3)
        section_names = ["今日可交易", "今日不可交易", "今日观察名单"]
        for idx, section_name in enumerate(section_names):
            with brief_cols[idx]:
                st.markdown(f"**{section_name}**")
                lines = brief_sections.get(section_name, ["- 无"])
                for line in lines[:5]:
                    st.write(line)

    if instrument_rows:
        st.markdown("**按标的的统一判断 + Fidelity 操作指引**")
        instrument_display = []
        for row in instrument_rows:
            recommendation = "Trade" if row["stance"] in {"LONG", "SHORT"} and abs(row["score"]) >= 0.35 else "Watch" if row["stance"] != "MIXED" else "No Trade"
            fi = _fidelity_info(row["instrument"])
            if row["stance"] == "SHORT":
                fidelity_action = fi["short_action"]
            elif row["stance"] == "LONG":
                fidelity_action = fi["long_action"]
            else:
                fidelity_action = "观望 / Stand aside"
            instrument_display.append(
                {
                    "instrument": row["instrument"],
                    "Fidelity代码": fi["fidelity_ticker"],
                    "类型": fi["type"],
                    "今日建议": recommendation,
                    "方向": row["stance"],
                    "score": row["score"],
                    "Fidelity操作": fidelity_action,
                    "support": row["support"],
                }
            )
        st.dataframe(pd.DataFrame(instrument_display), width="stretch")

        st.markdown("**怎么在 Fidelity 下单（专业版 + 直白中文版）**")
        playbook_cols = st.columns(min(3, len(instrument_rows)))
        for idx, row in enumerate(instrument_rows[:3]):
            guide = _instrument_playbook(row)
            pro_guide = _instrument_playbook_pro(row)
            fi = _fidelity_info(row["instrument"])
            is_short = row["stance"] == "SHORT"
            beginner_cn = fi["beginner_short_note"] if is_short else fi["beginner_note"]
            fidelity_ticker = fi["fidelity_ticker"]
            fidelity_action = fi["short_action"] if is_short else fi["long_action"]
            with playbook_cols[idx]:
                st.markdown(
                    f"**{row['instrument']} ({fidelity_ticker}) | {guide['today']}**\n\n"
                    f"**专业版**\n"
                    f"- Bias: {row['stance']}\n"
                    f"- Fidelity操作: {fidelity_action}\n"
                    f"- Execution: {pro_guide['how']}\n"
                    f"- Entry: {pro_guide['entry']}\n"
                    f"- Risk: {pro_guide['risk']}\n\n"
                    f"**直白版（初学者）**\n"
                    f"- {beginner_cn}\n"
                    f"- 怎么做: {guide['how']}\n"
                    f"- 入场: {guide['entry']}\n"
                    f"- 风险: {guide['risk']}"
                )

    st.markdown("**区域明细（解释原因，不是首页最终结论）**")
    display_rows = []
    for region_id, signal in ranked:
        score = signal.get("vote_score")
        if score is None:
            score = signal.get("ndvi_change")
        # Ensure score is always float for Arrow compatibility
        if isinstance(score, (int, float)):
            score = float(score)
        else:
            score = 0.0
        display_rows.append(
            {
                "rank_region": region_id,
                "action": signal.get("trading_action", "FLAT"),
                "raw": signal.get("raw_trading_action", signal.get("trading_action", "FLAT")),
                "confidence": signal.get("confidence", "Low"),
                "actionability": signal.get("actionability", "Ignore"),
                "instrument": ", ".join(signal.get("instruments", [])),
                "signal": signal.get("signal", "No data"),
                "score": score,
            }
        )
    st.dataframe(pd.DataFrame(display_rows), width="stretch")

    if pending:
        st.markdown("**正在确认，不要急着下单**")
        for region_id, signal in pending[:5]:
            st.write(
                f"- {region_id}: 当前 `{signal.get('trading_action')}`，原始方向 `{signal.get('raw_trading_action')}`，"
                f"{signal.get('signal')}"
            )

    if blocked:
        st.markdown("**系统级别暂不交易**")
        for region_id, signal in blocked[:5]:
            st.write(f"- {region_id}: {signal.get('signal')} | {signal.get('confidence')} | action={signal.get('trading_action')}")


def _render_sidebar_chat(
    st,
    daily_summary: Dict,
    persistence_state: Dict,
    daily_brief: str,
    selected_day: str,
) -> None:
    """Render the persistent chat panel in the Streamlit sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 Ask QuantTrade")
    st.sidebar.caption(
        "Ask anything about today's signals, how to trade on Fidelity, "
        "or what a term means. Powered by ChatGPT Plus → GPT-4.1 → Claude Sonnet 4.6."
    )

    # Check if token already set
    has_token = bool(st.session_state.get("chat_chatgpt_token", ""))
    expand_auth = not has_token

    with st.sidebar.expander("🔑 Authentication (stored in session only)", expanded=expand_auth):
        st.markdown("### 🆓 Free: ChatGPT Plus (uses your subscription)")
        
        st.markdown(
            """
            <a href="https://chatgpt.com" target="_blank" style="
                display: inline-block;
                background: #10a37f;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                text-decoration: none;
                font-weight: 600;
                margin-bottom: 12px;
            ">Open ChatGPT →</a>
            """,
            unsafe_allow_html=True,
        )
        
        with st.expander("📖 How to get your session token (click to expand)", expanded=False):
            st.markdown(
                """
                **Step 1:** Click the green "Open ChatGPT" button above and log in with your ChatGPT Plus account.
                
                **Step 2:** Open Developer Tools
                - **Mac:** Press `Cmd + Option + I`
                - **Windows/Linux:** Press `F12`
                - Or right-click anywhere → "Inspect"
                
                **Step 3:** Go to the **Application** tab (or **Storage** in some browsers)
                
                **Step 4:** In the left sidebar, expand **Cookies** → click on `https://chatgpt.com`
                
                **Step 5:** Find the cookie named:
                ```
                __Secure-next-auth.session-token
                ```
                (or just `next-auth.session-token` on some browsers)
                
                **Step 6:** Double-click the **Value** column to select it, then copy (Cmd+C / Ctrl+C)
                
                **Step 7:** Paste it in the box below
                
                ---
                
                ⚠️ **Notes:**
                - This token expires every few hours/days. When chat stops working, just re-copy it.
                - The token is stored only in your browser session, never saved to disk.
                - If you don't have ChatGPT Plus, you can use the paid API keys below as fallback.
                """
            )
        
        chatgpt_token = st.text_input(
            "ChatGPT Session Token",
            type="password",
            key="chat_chatgpt_token",
            help="Free — uses your ChatGPT Plus subscription. Token expires periodically; re-copy when needed.",
            placeholder="Paste your __Secure-next-auth.session-token here...",
        )
        
        if chatgpt_token:
            st.success("✅ ChatGPT token set! You can now chat for free using your subscription.")
        
        st.markdown("---")
        st.markdown("### 💰 Paid fallbacks (optional)")
        st.caption("Only used if ChatGPT token fails or is not set.")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            key="chat_openai_key",
            help="Required for GPT-4.1 API fallback. Leave blank to use env OPENAI_API_KEY.",
            placeholder="sk-...",
        )
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            key="chat_anthropic_key",
            help="Required for Claude Sonnet 4.6 fallback. Leave blank to use env ANTHROPIC_API_KEY.",
            placeholder="sk-ant-...",
        )

    # Initialise session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_system_prompt" not in st.session_state:
        st.session_state.chat_system_prompt = ""

    # Rebuild system prompt whenever the day or summary changes
    new_prompt = build_system_prompt(
        daily_summary=daily_summary,
        persistence_state=persistence_state,
        daily_brief=daily_brief,
        selected_day=selected_day,
        wiki_query=selected_day or "",
    )
    if new_prompt != st.session_state.chat_system_prompt:
        st.session_state.chat_system_prompt = new_prompt
        # Don't wipe history — just update context silently

    # Render conversation history (inside a scrollable container)
    with st.sidebar.container():
        for msg in st.session_state.chat_messages:
            role_label = "**You**" if msg["role"] == "user" else f"**Assistant** _{msg.get('model', '')}_ "
            if msg["role"] == "user":
                st.sidebar.markdown(f"🧑 {role_label}: {msg['content']}")
            else:
                st.sidebar.markdown(f"🤖 {role_label}:\n\n{msg['content']}")

    # Input form
    with st.sidebar.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your question",
            placeholder="e.g. 今天该怎么交易大豆? / What does FLAT mean?",
            height=80,
            key="chat_input",
        )
        col1, col2 = st.columns([3, 1])
        submitted = col1.form_submit_button("Send")
        col2.form_submit_button("Clear", on_click=lambda: st.session_state.update(chat_messages=[]))

    if submitted and user_input.strip():
        user_msg: Dict = {"role": "user", "content": user_input.strip()}
        st.session_state.chat_messages.append(user_msg)

        with st.sidebar:
            with st.spinner("Thinking..."):
                reply = ask(
                    messages=st.session_state.chat_messages,
                    system_prompt=st.session_state.chat_system_prompt,
                    chatgpt_access_token=chatgpt_token or None,
                    openai_api_key=openai_key or None,
                    anthropic_api_key=anthropic_key or None,
                )
        st.session_state.chat_messages.append(reply)
        st.rerun()


def _render_execution_dashboard(st):
    """Render execution ledger, fills, risk decisions, and reconciler status."""
    db_path = Path("outputs/execution/orders.sqlite")
    if not db_path.exists():
        st.info("No execution ledger found. Run the pipeline in shadow mode first.")
        return

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    col1, col2, col3, col4 = st.columns(4)
    total = conn.execute("SELECT COUNT(*) as c FROM orders").fetchone()["c"]
    filled = conn.execute("SELECT COUNT(*) as c FROM orders WHERE status='filled'").fetchone()["c"]
    rejected = conn.execute("SELECT COUNT(*) as c FROM orders WHERE status='rejected'").fetchone()["c"]
    pending = conn.execute("SELECT COUNT(*) as c FROM orders WHERE status IN ('pending','accepted')").fetchone()["c"]
    col1.metric("Total Orders", total)
    col2.metric("Filled", filled)
    col3.metric("Rejected", rejected)
    col4.metric("Pending/Open", pending)

    st.markdown("---")
    st.markdown("### Recent Orders")
    rows = conn.execute(
        "SELECT client_order_id, symbol, side, status, quantity, notional, "
        "limit_price as filled_avg_price, broker, created_at, filled_at "
        "FROM orders ORDER BY created_at DESC LIMIT 20"
    ).fetchall()
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        display_cols = [c for c in ["symbol", "side", "status", "quantity", "notional", "filled_avg_price", "broker", "created_at"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No orders yet.")

    st.markdown("---")
    rec_col, alert_col = st.columns(2)

    with rec_col:
        st.markdown("### Reconciliation Runs")
        rec_rows = conn.execute(
            "SELECT run_at, status, orders_drift, fills_missing, alert "
            "FROM reconciliation_runs ORDER BY run_at DESC LIMIT 5"
        ).fetchall()
        if rec_rows:
            for r in rec_rows:
                icon = "OK" if r["status"] == "ok" else "ALERT"
                st.markdown(f"- **{r['run_at'][:19]}** {icon} drift={r['orders_drift']} missing={r['fills_missing']}")
        else:
            st.info("No reconciliation runs yet.")

    with alert_col:
        st.markdown("### Alert History")
        alert_path = Path("outputs/execution/alert_history.json")
        if alert_path.exists():
            try:
                alerts = json.loads(alert_path.read_text())
                for a in alerts[-5:]:
                    color = {"ORDER_FILLED": "green", "ORDER_REJECTED": "red", "RECONCILER_ALERT": "orange"}.get(a.get("type", ""), "blue")
                    st.markdown(f"- :{color}[**{a.get('type', '')}**] {a.get('timestamp', '')[:19]} — {a.get('symbol', a.get('stranded', ''))}")
            except Exception:
                st.info("Could not load alerts.")
        else:
            st.info("No alerts recorded.")

    st.markdown("---")
    st.markdown("### Risk Gate Decisions")
    risk_rows = conn.execute(
        "SELECT client_order_id, approved, reason, details, created_at "
        "FROM risk_decisions ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    if risk_rows:
        risk_df = pd.DataFrame([dict(r) for r in risk_rows])
        risk_df["approved"] = risk_df["approved"].map({1: "YES", 0: "NO"})
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

    conn.close()


def _render_portfolio_monitor(st):
    """Render real-time portfolio monitoring section"""
    from datetime import datetime
    
    portfolio_path = PROJECT_ROOT / "outputs" / "paper_trading" / "multi_asset_portfolio.json"
    
    # Try loading from scheduler API first, then fallback to local file
    portfolio = None
    
    # Try scheduler API
    api_data = _fetch_from_scheduler("/api/portfolio")
    if api_data and "positions" in api_data:
        portfolio = api_data
    
    # Fallback to local file
    if portfolio is None and portfolio_path.exists():
        raw_content = portfolio_path.read_text()
        
        # Handle Git LFS pointer
        if raw_content.startswith("version https://git-lfs.github.com/spec/v1"):
            st.warning(
                "Portfolio file is a Git LFS pointer. Run `git lfs pull` to fetch the actual data, "
                "or create a new portfolio file."
            )
            return
        
        try:
            portfolio = json.loads(raw_content)
        except json.JSONDecodeError as e:
            st.warning(f"Could not parse portfolio file: {e}")
            return
    
    if portfolio is None:
        return
    
    positions = portfolio.get("positions", {})
    cash = float(portfolio.get("cash", 0))

    portfolio_model = None
    try:
        portfolio_model = MultiAssetPortfolio(output_base="outputs")
    except Exception:
        pass
    if portfolio_model:
        current_prices = get_prices_for_portfolio(portfolio_model)
    else:
        # Web container: fetch prices from scheduler API
        api_prices = _fetch_from_scheduler("/api/prices", timeout=30)
        current_prices = api_prices if isinstance(api_prices, dict) else {}
    
    # Calculate total value and P&L
    total_position_value = 0
    total_pnl = 0
    max_risk = 0
    
    for ticker, pos in positions.items():
        entry_price = float(pos.get("entry_price", 0))
        current_price = float(current_prices.get(ticker) or entry_price)
        quantity = float(pos.get("quantity", 0))
        direction = pos.get("direction", "long")
        stop_loss = float(pos.get("stop_loss", 0))
        initial_position_value = float(pos.get("position_value", entry_price * quantity))
        
        # Mark-to-market position value
        if direction == "long":
            position_value = current_price * quantity
        else:
            position_value = initial_position_value + ((entry_price - current_price) * quantity)
        total_position_value += position_value
        
        # P&L calculation
        if direction == "long":
            pnl = (current_price - entry_price) * quantity
            risk = abs((current_price - stop_loss) * quantity)
        else:  # short
            pnl = (entry_price - current_price) * quantity
            risk = abs((stop_loss - current_price) * quantity)
        
        total_pnl += pnl
        max_risk += risk
    
    total_assets = cash + total_position_value
    
    # Display portfolio metrics
    st.markdown("---")
    st.markdown("## 💰 Portfolio Monitor")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Assets",
            f"${total_assets:,.2f}",
            f"${total_pnl:+,.2f}"
        )
    
    with col2:
        st.metric(
            "Cash",
            f"${cash:,.2f}",
            f"{(cash/total_assets*100):.1f}%"
        )
    
    with col3:
        st.metric(
            "Positions",
            f"${total_position_value:,.2f}",
            f"{(total_position_value/total_assets*100):.1f}%"
        )
    
    with col4:
        st.metric(
            "Total P&L",
            f"${total_pnl:+,.2f}",
            f"{(total_pnl/total_assets*100):+.2f}%"
        )
    
    with col5:
        st.metric(
            "Max Risk",
            f"${max_risk:,.2f}",
            f"{(max_risk/total_assets*100):.2f}%"
        )
    
    # Display positions if any
    if positions:
        st.markdown("### Current Positions")
        
        for ticker, pos in positions.items():
            entry_price = float(pos.get("entry_price", 0))
            current_price = float(current_prices.get(ticker) or entry_price)
            quantity = float(pos.get("quantity", 0))
            direction = pos.get("direction", "long")
            stop_loss = float(pos.get("stop_loss", 0))
            take_profit = float(pos.get("take_profit", 0))
            grade = pos.get("signal_grade", "N/A")
            accuracy = pos.get("signal_accuracy", 0)
            initial_position_value = float(pos.get("position_value", entry_price * quantity))
            
            if direction == "long":
                pnl = (current_price - entry_price) * quantity
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                position_value = current_price * quantity
            else:
                pnl = (entry_price - current_price) * quantity
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                position_value = initial_position_value + pnl
            
            # Color based on P&L
            pnl_color = "green" if pnl >= 0 else "red"
            direction_emoji = "🔴" if direction == "short" else "🟢"

            st.markdown(f"**{direction_emoji} {_esc(str(ticker))} {_esc(str(direction.upper()))}**")
            st.markdown(
                f"Entry: `${entry_price:.2f}` | Current: `${current_price:.2f}` | "
                f"Position: `${position_value:,.2f}` | "
                f"P&L: <span style=\"color:{pnl_color}\">${pnl:+,.2f} ({pnl_pct:+.2f}%)</span> | "
                f"Stop: `${stop_loss:.2f}` | Target: `${take_profit:.2f}` | "
                f"Grade: `{_esc(str(grade))}` | Accuracy: `{accuracy:.0f}%`",
                unsafe_allow_html=True,
            )
            st.markdown("---")
    
        st.markdown("")
        
        # Add equity curve section
        st.subheader("📈 资产曲线")
        action_col1, action_col2 = st.columns([1, 5])
        with action_col1:
            if st.button("重建曲线", key="rebuild_asset_history"):
                rebuilt = rebuild_asset_history(output_base="outputs", initial_capital=100000.0)
                st.success(f"已重建 {len(rebuilt.get('daily_assets', []))} 个资产点")
        with action_col2:
            st.caption("如果组合文件或历史价格修正了，可以点这里重新生成资产曲线。")
        
        # Load asset history - try scheduler API first, then local file
        history = None
        
        # Try scheduler API
        api_data = _fetch_from_scheduler("/api/outputs/asset_history.json")
        if api_data and isinstance(api_data, dict) and api_data.get("daily_assets"):
            history = api_data
        elif api_data and isinstance(api_data, list) and len(api_data) > 0:
            history = api_data
        
        # Fallback to local file
        if history is None:
            tracker_path = PROJECT_ROOT / "outputs" / "asset_history.json"
            if tracker_path.exists():
                try:
                    history = json.loads(tracker_path.read_text())
                except json.JSONDecodeError as e:
                    st.warning(f"无法解析资产历史文件: {e}")
                    return
        
        if history is None:
            st.info("暂无资产历史数据。运行 pipeline 后会自动记录。")
            return

        # Handle both list and dict formats
        if isinstance(history, list):
            records = history
        else:
            records = history.get("daily_assets", [])
        
        if not records:
            st.info("暂无资产历史数据")
            return

        today_str = datetime.now().strftime("%Y-%m-%d")
        updated_today = False
        for item in records:
            if item.get("date") == today_str:
                item["total_value"] = round(total_assets, 2)
                item["timestamp"] = datetime.now().isoformat()
                updated_today = True
                break
        if not updated_today:
            records.append({
                "date": today_str,
                "total_value": round(total_assets, 2),
                "timestamp": datetime.now().isoformat(),
            })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
        df = df.dropna(subset=["total_value"])

        if df.empty:
            st.info("资产历史为空，暂时无法绘制曲线。")
            return
        
        # Calculate daily returns
        df["daily_return"] = df["total_value"].pct_change() * 100
        df["equity_return_pct"] = ((df["total_value"] / df.iloc[0]["total_value"]) - 1) * 100
        
        # Metrics
        start_value = df.iloc[0]["total_value"]
        end_value = df.iloc[-1]["total_value"]
        total_return = ((end_value - start_value) / start_value * 100) if start_value > 0 else 0
        running_peak = df["total_value"].cummax()
        drawdown_pct = ((df["total_value"] / running_peak) - 1) * 100
        max_drawdown = drawdown_pct.min() if not drawdown_pct.empty else 0
        latest_change = df.iloc[-1]["daily_return"] if len(df) > 1 else 0
        
        # Display metrics
        st.caption("按 cash + 持仓实时市值 mark-to-market 计算；空头仓位会按当前价格重估，不再固定按开仓价记账。")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("起始资产", f"${start_value:,.2f}")
        col2.metric("当前资产", f"${end_value:,.1f}")
        col3.metric("总收益率", f"{total_return:.1f}%")
        col4.metric("最大回撤", f"{max_drawdown:.1f}%", f"{latest_change:+.2f}% 今日变动")
        
        # Equity curve chart
        st.line_chart(
            df.set_index("date")["total_value"],
            use_container_width=True,
        )

        st.caption("这条曲线反映组合总资产，不只是现金；如果当天重复跑 pipeline，同一天资产值会被覆盖更新。")
        
        # Daily returns chart
        st.subheader("📊 每日收益率")
        st.bar_chart(
            df.set_index("date")["daily_return"],
            use_container_width=True,
        )
        
        # Show data table
        with st.expander("查看详细数据"):
            display_df = df[["date", "total_value", "daily_return", "equity_return_pct"]].copy()
            display_df.columns = ["date", "total_value", "daily_return_pct", "equity_return_pct"]
            st.dataframe(display_df, use_container_width=True)

def main():
    import streamlit as st

    st.set_page_config(
        page_title="QuantTrade", 
        layout="wide",
        page_icon="📊"
    )
    
    st.title("QuantTrade")
    st.caption("Satellite-based trading signals → paper trading portfolio")

    output_base = os.environ.get("OUTPUT_BASE", "outputs")
    api_base = os.environ.get("SCHEDULER_API_URL", "http://127.0.0.1:8000")
    try:
        regions = list_regions()
    except Exception as e:
        st.error(f"Failed to load regions: {e}")
        st.info("Make sure the scheduler API is accessible.")
        return
    region_options = {region["name"]: region["id"] for region in regions}

    # Minimal sidebar: just region selector
    selected_region_name = st.sidebar.selectbox("Region", options=list(region_options.keys()))
    selected_region = region_options[selected_region_name]

    # Resolve latest day from signals API
    api_signals = _fetch_from_scheduler("/api/all-signals")
    days = []
    if api_signals and api_signals.get("signals"):
        # Extract unique dates from signals
        signal_dates = set()
        for sig in api_signals["signals"].values():
            d = sig.get("date")
            if d:
                signal_dates.add(d)
        days = sorted(signal_dates)
    if not days:
        days = list_available_days(output_base, selected_region)
    if not days:
        st.warning("No data available yet. Run the pipeline first.")
        return
    selected_day = days[-1]
    summary_day = _resolve_summary_day(output_base, selected_day)

    # --- 4 Tabs ---
    tab_portfolio, tab_signals, tab_trade, tab_exec, tab_backtests = st.tabs(
        ["Portfolio", "Signals", "Trading", "Execution", "Backtests"]
    )

    with tab_portfolio:
        _render_portfolio_monitor(st)

    with tab_signals:
        _render_global_monitor(st, regions, output_base)
        st.markdown("---")
        st.markdown("### 每日简报")
        brief_text = _load_daily_brief(output_base, summary_day)
        if brief_text:
            st.markdown(brief_text)
            st.download_button(
                "Download Brief",
                brief_text.encode("utf-8"),
                file_name=f"{summary_day}_daily_brief_zh.md",
                mime="text/markdown",
            )
        else:
            st.info("暂无简报。")

    with tab_trade:
        use_backend = bool(_fetch_from_scheduler("/health"))
        # Try remote bundle first, then local
        bundle = None
        try:
            bundle = _load_remote_bundle(
                api_base=api_base,
                selected_day=selected_day,
                output_base=output_base,
                region_id=selected_region,
            )
        except Exception:
            pass
        if not bundle:
            day_dir = os.path.join(output_base, selected_region, selected_day)
            if os.path.isdir(day_dir):
                bundle = load_day_bundle(day_dir)
        trade_signal = None
        api_signals = _fetch_from_scheduler("/api/all-signals")
        if api_signals and api_signals.get("signals") and selected_region in api_signals["signals"]:
            sig = api_signals["signals"][selected_region]
            trade_signal = {
                "date": selected_day,
                "source": "api",
                "signal": sig.get("signal", "No data"),
                "bias": _bias_for_signal(sig.get("signal", "No data")),
                "confidence": sig.get("confidence", "Unknown"),
                "signal_strength": sig.get("signal_strength"),
                "coverage_score": sig.get("coverage_score"),
                "throughput_index_corrected": sig.get("throughput_change"),
                "baseline_value": sig.get("baseline_value"),
                "dod_change": sig.get("throughput_change"),
                "dod_change_pct": sig.get("throughput_change_pct"),
                "confirmation_days": 0,
                "zscore": None,
                "rolling_mean_7": None,
                "rationale": sig.get("rationale", "Signal from scheduler API."),
                "series": pd.DataFrame(),
                "reroute_flag": sig.get("reroute_flag", False),
                "actionability": sig.get("actionability", "Ignore"),
                "signal_source": sig.get("type", "unknown"),
            }
        if trade_signal is None:
            trade_signal = latest_region_signal(selected_region, output_base=output_base, selected_day=selected_day, version="v2")
        region_instruments = list_region_instruments(selected_region)
        trade_ticket = _build_trade_ticket(trade_signal, region_instruments)

        _render_trade_ticket(st, trade_ticket)

        # Show all actionable high-confidence signals across regions
        if api_signals and api_signals.get("signals"):
            actionable = []
            for rid, s in api_signals["signals"].items():
                cs = s.get("confidence_score", 0)
                if s.get("actionability") in ("Actionable",) and cs and cs >= 75:
                    actionable.append({
                        "Region": s.get("region_name", rid),
                        "Signal": s.get("signal", ""),
                        "Confidence": f"{cs:.0f}%",
                        "Instruments": ", ".join(s.get("instruments", [])),
                        "Rationale": s.get("rationale", "")[:80],
                    })
            if actionable:
                st.markdown("---")
                st.markdown(f"**🔥 {len(actionable)} Actionable Signals (confidence ≥ 75%)**")
                for a in actionable:
                    st.markdown(f'- **{a["Region"]}** → {a["Signal"]} ({a["Confidence"]}) — {a["Instruments"]}')
                    st.caption(a["Rationale"])

        if trade_signal and trade_signal.get("series") is not None and not trade_signal["series"].empty:
            signal_series = trade_signal["series"].copy()
            plot_columns = [
                c for c in ["throughput_index_corrected", "rolling_mean_7", "coverage_score"]
                if c in signal_series.columns
            ]
            if plot_columns:
                st.markdown("**Signal History**")
                st.line_chart(signal_series.set_index("date")[plot_columns], use_container_width=True)

    with tab_exec:
        _render_execution_dashboard(st)

    with tab_backtests:
        summaries = _load_latest_backtests(output_base, selected_region)
        if not summaries:
            st.info("No backtest results yet.")
        else:
            summary_df = pd.DataFrame(summaries)
            st.markdown("**Backtest Results**")
            summary_columns = [
                c for c in ["symbol", "strategy_name", "total_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "trade_count"]
                if c in summary_df.columns
            ]
            st.dataframe(summary_df[summary_columns] if summary_columns else summary_df, use_container_width=True)
            selected_summary = summaries[0]
            equity_path_value = selected_summary.get("equity_path")
            if equity_path_value and Path(equity_path_value).exists():
                equity_df = pd.read_parquet(Path(equity_path_value))
                if "date" in equity_df.columns:
                    equity_df["date"] = pd.to_datetime(equity_df["date"])
                    st.markdown("**Equity Curve**")
                    st.line_chart(equity_df.set_index("date")[["equity_curve"]], use_container_width=True)
                    st.markdown("**Drawdown**")
                    st.line_chart(equity_df.set_index("date")[["drawdown"]], use_container_width=True)
            else:
                st.caption("No equity curve data for this backtest.")


if __name__ == "__main__":
    main()
