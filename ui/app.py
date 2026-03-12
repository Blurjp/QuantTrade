"""
Local Streamlit UI for browsing daily QuantTrade outputs.

Run with:
  streamlit run ui/app.py
"""

from pathlib import Path
import json
import sys
from typing import Dict, List, Optional, Tuple

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

    if coverage < 0.55:
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
    monitor_df = build_monitor_snapshot(output_base=output_base)
    if monitor_df.empty:
        st.info("No regional runs available yet.")
        return

    status_map = {region["id"]: load_region_status(output_base, region["id"]) for region in regions}
    monitor_df["run_status"] = monitor_df["region"].map(lambda region_id: status_map.get(region_id, {}).get("run_status", "unknown"))
    monitor_df["last_run_at"] = monitor_df["region"].map(lambda region_id: status_map.get(region_id, {}).get("last_run_at"))
    monitor_df["region_name"] = monitor_df["region"].map({region["id"]: region["name"] for region in regions})
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
    st.caption("Latest regional state ranked by actionability and confidence.")
    display_df = pd.DataFrame(
        {
            "region": monitor_df["region_name"],
            "latest_day": monitor_df["date"],
            "run_status": monitor_df["run_status"],
            "last_run_at": monitor_df["last_run_at"],
            "actionability": monitor_df["actionability"],
            "signal": monitor_df["signal"],
            "confidence": monitor_df["confidence"],
            "coverage": monitor_df["coverage_score"].apply(_format_pct),
            "strength": monitor_df["signal_strength"],
            "primary_ticker": monitor_df["primary_instrument"],
            "reroute_flag": monitor_df["reroute_flag"],
        }
    )
    st.dataframe(display_df, width="stretch")


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


def _load_persistence_state(output_base: str) -> dict:
    state_path = Path(output_base) / "signal_persistence_state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text())


def _load_daily_summary(output_base: str, selected_day: str) -> Dict:
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
          <div style="font-size:12px; letter-spacing:.08em; color:{accent}; font-weight:700;">{region_id}</div>
          <div style="font-size:30px; font-weight:800; color:#1f2a2e; margin-top:6px;">{signal.get('trading_action', 'FLAT')}</div>
          <div style="font-size:14px; color:#44525a; margin-top:4px;">{signal.get('confidence', 'Low')} / {signal.get('actionability', 'Ignore')}</div>
          <div style="font-size:16px; color:#1f2a2e; margin-top:10px; font-weight:600;">{signal.get('signal', 'No data')}</div>
          <div style="font-size:13px; color:#44525a; margin-top:8px;">标的: {instruments}</div>
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
    if summary_day != selected_day:
        st.caption(f"当前区域日期 `{selected_day}` 没有全局汇总，首页排序改为使用最近一次全局汇总 `{summary_day}`。")

    signals = summary.get("signals", {})
    if not signals:
        st.info("当天没有可用信号。")
        return

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
          <div style="font-size:36px; font-weight:800; color:#1f2a2e; margin-top:8px;">{verdict['label']}</div>
          <div style="font-size:16px; color:#46545b; margin-top:8px;">{verdict['reason']}</div>
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
              <div style="font-size:34px; font-weight:800; color:#1f2a2e; margin-top:8px;">{top_region} → {top_signal.get('trading_action')}</div>
              <div style="font-size:18px; color:#334047; margin-top:6px;">{top_signal.get('signal')}</div>
              <div style="font-size:14px; color:#5d6970; margin-top:10px;">标的: {instruments} | 置信度: {top_signal.get('confidence')} | actionability: {top_signal.get('actionability')}</div>
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
            score = signal.get("ndvi_change", "")
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

    with st.sidebar.expander("🔑 Authentication (stored in session only)", expanded=False):
        st.markdown("**Free: ChatGPT Plus (uses your subscription)**")
        st.markdown(
            """
            **How to get your access token:**
            1. Open [chat.openai.com](https://chat.openai.com) in your browser
            2. Log in with your ChatGPT Plus account
            3. Open DevTools (F12 or Cmd+Option+I)
            4. Go to **Application** → **Cookies** → `chat.openai.com`
            5. Copy the value of `__Secure-next-auth.session-token`
            6. Paste it below
            """
        )
        chatgpt_token = st.text_input(
            "ChatGPT Session Token",
            type="password",
            key="chat_chatgpt_token",
            help="Free — uses your ChatGPT Plus subscription. Token expires periodically; re-copy when needed.",
        )
        
        st.markdown("---")
        st.markdown("**Paid fallbacks (optional):**")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            key="chat_openai_key",
            help="Required for GPT-4.1 API fallback. Leave blank to use env OPENAI_API_KEY.",
        )
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            key="chat_anthropic_key",
            help="Required for Claude Sonnet 4.6 fallback. Leave blank to use env ANTHROPIC_API_KEY.",
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


def main():
    import streamlit as st

    st.set_page_config(page_title="QuantTrade", layout="wide")
    st.title("QuantTrade Trading Console")
    st.caption("把关键海运通道的船流数据翻译成更容易理解的交易信号。")

    output_base = st.sidebar.text_input("Outputs Directory", "outputs")
    api_base = st.sidebar.text_input("Backend API", "http://127.0.0.1:8000")
    use_backend = st.sidebar.checkbox("Use backend service", value=False)
    regions = list_regions()
    region_options = {region["name"]: region["id"] for region in regions}
    selected_region_name = st.sidebar.selectbox("Region", options=list(region_options.keys()))
    selected_region = region_options[selected_region_name]
    region_meta = next(region for region in regions if region["id"] == selected_region)
    st.sidebar.caption(region_meta["description"])

    days = []
    if use_backend:
        try:
            response = requests.get(
                f"{api_base.rstrip('/')}/days",
                params={"output_base": output_base, "region": selected_region},
                timeout=15,
            )
            response.raise_for_status()
            days = response.json().get("days", [])
            st.sidebar.success("Backend connected")
        except requests.RequestException as exc:
            st.sidebar.warning(f"Backend unavailable, using local files: {exc}")
            use_backend = False

    if not use_backend:
        days = list_available_days(output_base, selected_region)

    if not days:
        st.warning("No completed day runs found. Run `python -m pipeline.run --date YYYY-MM-DD` first.")
        return

    selected_day = st.sidebar.selectbox("Day", options=list(reversed(days)))
    if use_backend:
        bundle = _load_remote_bundle(api_base, selected_day, output_base, selected_region)
    else:
        region_output_base = resolve_region_output_base(output_base, selected_region)
        bundle = load_day_bundle(str(Path(region_output_base) / selected_day))
        bundle["calibration_metrics"] = pd.DataFrame()
        bundle["calibration_report"] = {}

    paths = bundle["paths"]
    metrics_df = bundle["metrics"]
    detections_df = bundle["detections"]
    load_log_df = bundle["load_log"]
    manifest_df = bundle["manifest"]
    report = bundle["report"]
    calibration_df = bundle["calibration_metrics"]
    calibration_report = bundle["calibration_report"]
    metrics_row = metrics_df.iloc[0].to_dict() if len(metrics_df) > 0 else {}
    trade_signal = latest_region_signal(selected_region, output_base=output_base, selected_day=selected_day, version="v2")
    region_instruments = list_region_instruments(selected_region)
    trade_ticket = _build_trade_ticket(trade_signal, region_instruments)
    summary_day = _resolve_summary_day(output_base, selected_day)
    daily_summary = _load_daily_summary(output_base, selected_day)

    _render_ranked_today_board(st, selected_day, daily_summary, output_base)
    with st.expander("当前区域详情", expanded=False):
        _render_summary_header(st, selected_day, trade_signal, metrics_row, summary_day=summary_day)
    _render_trade_ticket(st, trade_ticket)
    _render_how_to_use(st)

    # Sidebar chat — rendered after all data is loaded so context is complete
    persistence_state = _load_persistence_state(output_base)
    brief_text_for_chat = _load_daily_brief(output_base, summary_day)
    _render_sidebar_chat(
        st,
        daily_summary=daily_summary,
        persistence_state=persistence_state,
        daily_brief=brief_text_for_chat,
        selected_day=summary_day,
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric("数据覆盖率", _format_pct(metrics_row.get("coverage_score", 0)))
    metric_columns[1].metric("发现的场景数", int(metrics_row.get("num_scenes", 0)))
    metric_columns[2].metric("成功加载场景", int(metrics_row.get("loaded_scenes", 0)))
    metric_columns[3].metric("检测到的船只", int(len(detections_df)))
    metric_columns[4].metric("最大时间缺口(小时)", metrics_row.get("max_scene_gap_hours", "n/a"))

    tab_monitor, tab_brief, tab_today, tab_trade, tab_backtests, tab_signal, tab_detections, tab_scenes, tab_files = st.tabs(
        ["Monitor", "Brief", "Today", "Trading", "Backtests", "Signal", "Detections", "Previews", "Files"]
    )

    with tab_monitor:
        _render_global_monitor(st, regions, output_base)

    with tab_brief:
        st.markdown("**每日中文简报**")
        brief_text = _load_daily_brief(output_base, summary_day)
        if brief_text:
            st.markdown(brief_text)
            st.download_button(
                "Download Chinese Brief",
                brief_text.encode("utf-8"),
                file_name=f"{summary_day}_daily_brief_zh.md",
                mime="text/markdown",
            )
        else:
            st.info("这个日期还没有生成中文简报。先运行 `scripts/china_daily_brief.py` 或 Railway 日任务。")

        dashboard_path = Path(output_base) / summary_day / "signals_dashboard.html"
        if dashboard_path.exists():
            st.markdown("**Dashboard 文件**")
            st.code(str(dashboard_path))

        persistence_state = _load_persistence_state(output_base)
        if persistence_state:
            st.markdown("**信号确认状态**")
            st.json(persistence_state)

    with tab_today:
        st.markdown("**今天到底能不能用**")
        if metrics_row.get("num_scenes", 0) == 0:
            st.error("今天没有可用卫星场景。这一天不能拿来做交易判断。")
        elif _coverage_confidence(metrics_row.get("coverage_score")) == "Low":
            st.warning("今天虽然有数据，但覆盖太差。更适合观望。")
        else:
            st.success("今天有可参考数据，可以继续看 Trading 页。")

        quick_cols = st.columns(2)
        with quick_cols[0]:
            st.markdown("**给交易看的结论**")
            if trade_ticket is None:
                st.write("没有校准信号。")
            else:
                st.write(f"- 主交易: {trade_ticket['primary_trade']}")
                st.write(f"- 仓位: {trade_ticket['position_size']}")
                st.write(f"- 失效条件: {trade_ticket['invalidation']}")
        with quick_cols[1]:
            st.markdown("**给研究看的背景**")
            st.write("- `coverage_score`: 今天数据完整不完整")
            st.write("- `throughput_index_corrected`: 校准后的船流强度")
            st.write("- `dod_change_pct`: 比昨天变化了多少")
            if region_instruments:
                st.write(f"- 主要交易标的: {', '.join(item['ticker'] for item in region_instruments)}")

        with st.expander("查看原始日报数据"):
            st.markdown("**Daily Metrics**")
            st.dataframe(metrics_df, width="stretch")
            st.markdown("**Scene Load Log**")
            st.dataframe(load_log_df, width="stretch")
            st.markdown("**Manifest**")
            st.dataframe(manifest_df, width="stretch")

    with tab_trade:
        if trade_signal is None:
            st.info("No calibrated signal is available for trade classification.")
        else:
            if trade_signal["source"] == "latest_available":
                st.caption(
                    f"No calibrated row exists for {selected_day}; showing latest available signal from {trade_signal['date']}."
                )

            trade_cols = st.columns(4)
            trade_cols[0].metric("今日建议", trade_signal["signal"])
            trade_cols[1].metric("置信度", trade_signal["confidence"])
            signal_label = "校准后船流强度" if trade_signal.get("signal_source") == "throughput_index_corrected" else "原始船流强度"
            trade_cols[2].metric(signal_label, _format_num(trade_signal["throughput_index_corrected"]))
            trade_cols[3].metric("日变化", _format_pct(trade_signal["dod_change_pct"]))

            explain_cols = st.columns(2)
            with explain_cols[0]:
                st.markdown("**交易含义**")
                instrument_text = ", ".join(
                    f"{item['ticker']} ({item['trade_direction_on_low_throughput']}/{item['trade_direction_on_high_throughput']})"
                    for item in region_instruments
                ) or "n/a"
                st.write(f"推荐标的: {instrument_text}")
                st.markdown("**触发原因**")
                st.write(trade_signal["rationale"])
            with explain_cols[1]:
                st.markdown("**规则输入**")
                st.json(
                    {
                        "date": trade_signal["date"],
                        "coverage_score": trade_signal["coverage_score"],
                        "throughput_index_corrected": trade_signal["throughput_index_corrected"],
                        "rolling_mean_7": trade_signal["rolling_mean_7"],
                        "dod_change": trade_signal["dod_change"],
                        "dod_change_pct": trade_signal["dod_change_pct"],
                    }
                )

            signal_series = trade_signal["series"].copy()
            plot_columns = [
                column
                for column in ["throughput_index_corrected", "rolling_mean_7", "coverage_score"]
                if column in signal_series.columns
            ]
            st.markdown("**历史上下文**")
            st.caption("看当前信号是不是只是单日噪音。")
            st.line_chart(signal_series.set_index("date")[plot_columns], width="stretch")

            export_columns = [
                column
                for column in [
                    "date",
                    "signal",
                    "confidence",
                    "bias",
                    "throughput_index_corrected",
                    "coverage_score",
                    "dod_change",
                    "dod_change_pct",
                    "rolling_mean_7",
                    "rationale",
                ]
                if column in signal_series.columns
            ]
            export_df = signal_series[export_columns].copy()
            export_df["date"] = export_df["date"].dt.date.astype(str)
            export_df = export_df.rename(
                columns={
                    "throughput_index_corrected": "corrected_throughput",
                    "coverage_score": "coverage",
                }
            )

            st.markdown("**每日交易信号表**")
            st.caption("你真正可以导出去做后续研究或交易流程的表。")
            st.dataframe(export_df.sort_values("date", ascending=False), width="stretch")
            st.download_button(
                "Download signal CSV",
                export_df.to_csv(index=False).encode("utf-8"),
                file_name="quanttrade_daily_signals.csv",
                mime="text/csv",
            )

    with tab_backtests:
        summaries = _load_latest_backtests(output_base, selected_region)
        if not summaries:
            st.info("No backtest artifacts found for this region yet.")
        else:
            summary_df = pd.DataFrame(summaries)
            st.markdown("**Latest Backtest Summaries**")
            summary_columns = [
                column
                for column in [
                    "symbol",
                    "strategy_name",
                    "total_return",
                    "sharpe",
                    "max_drawdown",
                    "win_rate",
                    "profit_factor",
                    "trade_count",
                ]
                if column in summary_df.columns
            ]
            st.dataframe(summary_df[summary_columns] if summary_columns else summary_df, width="stretch")
            selected_summary = summaries[0]
            equity_path_value = selected_summary.get("equity_path")
            if equity_path_value and Path(equity_path_value).exists():
                equity_path = Path(equity_path_value)
                equity_df = pd.read_parquet(equity_path)
                if "date" in equity_df.columns:
                    equity_df["date"] = pd.to_datetime(equity_df["date"])
                    st.markdown("**Equity Curve**")
                    st.line_chart(equity_df.set_index("date")[["equity_curve"]], width="stretch")
                    st.markdown("**Drawdown**")
                    st.line_chart(equity_df.set_index("date")[["drawdown"]], width="stretch")
            else:
                st.caption("This backtest summary does not include an equity curve artifact.")

    with tab_signal:
        if len(calibration_df) == 0:
            st.info("No calibrated signal series found in outputs/calibration.")
        else:
            plot_df = calibration_df.copy()
            if "date" in plot_df.columns:
                plot_df["date"] = pd.to_datetime(plot_df["date"])
                plot_df = plot_df.sort_values("date")

            signal_columns = [
                column
                for column in [
                    "throughput_index_corrected",
                    "throughput_index_total",
                    "coverage_score",
                    "bias_factor",
                ]
                if column in plot_df.columns
            ]
            if signal_columns:
                st.line_chart(plot_df.set_index("date")[signal_columns], width="stretch")
            st.caption("这页是底层信号，不是结论页。主要用来看趋势，不要直接拿表格下单。")
            with st.expander("Calibration Report"):
                st.json(calibration_report or {"status": "missing"})
            with st.expander("Calibrated Metrics"):
                st.dataframe(plot_df, width="stretch")

    with tab_detections:
        if len(detections_df) == 0:
            st.info("今天没有检测到船只，或者当天没有可用场景。")
        else:
            st.dataframe(detections_df, width="stretch")
            st.download_button(
                "Download detections CSV",
                detections_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_day}_detections.csv",
                mime="text/csv",
            )

    with tab_scenes:
        previews = paths["previews"]
        if not previews:
            st.info("今天没有场景预览图。")
        else:
            for preview in previews:
                st.image(str(preview), caption=preview.stem, width="stretch")

    with tab_files:
        info_columns = st.columns(2)
        with info_columns[0]:
            st.markdown("**Run Report**")
            st.json(report or {"status": "missing"})
        with info_columns[1]:
            st.markdown("**Artifact Paths**")
            st.code(
                "\n".join(
                    f"{name}: {value}"
                    for name, value in paths.items()
                    if name != "previews"
                )
            )
        for preview in paths["previews"]:
            st.write(str(preview))


if __name__ == "__main__":
    main()
