"""
Local Streamlit UI for browsing daily QuantTrade outputs.

Run with:
  streamlit run ui/app.py
"""

from pathlib import Path
import json

import pandas as pd
import requests

from automation.status import load_region_status
from pipeline.instruments import get_primary_instrument, list_region_instruments
from pipeline.regions import list_regions, resolve_region_output_base
from pipeline.signals import build_monitor_snapshot, latest_region_signal
from pipeline.ui_data import list_available_days, load_day_bundle


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def _format_num(value: float | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _signal_style(signal: str) -> tuple[str, str]:
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


def _coverage_confidence(coverage_score: float | None) -> str:
    if coverage_score is None or pd.isna(coverage_score):
        return "Unknown"
    if coverage_score >= 0.75:
        return "High"
    if coverage_score >= 0.55:
        return "Medium"
    return "Low"


def _build_trade_ticket(trade_signal: dict | None, region_instruments: list[dict]) -> dict | None:
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
def _render_global_monitor(st, regions: list[dict], output_base: str) -> None:
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


def _load_latest_backtests(output_base: str, region_id: str) -> list[dict]:
    backtest_root = Path(output_base) / "regions" / region_id / "backtests"
    if not backtest_root.exists():
        return []

    summaries = []
    for summary_path in sorted(backtest_root.glob("*/*_summary.json")):
        payload = json.loads(summary_path.read_text())
        payload["summary_path"] = str(summary_path)
        summaries.append(payload)
    return summaries


def _render_summary_header(st, selected_day: str, trade_signal: dict | None, metrics_row: dict) -> None:
    st.subheader(f"交易结论 | {selected_day}")

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


def _render_trade_ticket(st, trade_ticket: dict | None) -> None:
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


def main():
    import streamlit as st

    st.set_page_config(page_title="QuantTrade", layout="wide")
    st.title("QuantTrade Trading Console")
    st.caption("把关键海运通道的船流数据翻译成更容易理解的交易信号。")

    output_base = st.sidebar.text_input("Outputs Directory", "outputs")
    api_base = st.sidebar.text_input("Backend API", "http://127.0.0.1:8000")
    use_backend = st.sidebar.checkbox("Use backend service", value=True)
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

    _render_summary_header(st, selected_day, trade_signal, metrics_row)
    _render_trade_ticket(st, trade_ticket)
    _render_how_to_use(st)

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
        brief_text = _load_daily_brief(output_base, selected_day)
        if brief_text:
            st.markdown(brief_text)
            st.download_button(
                "Download Chinese Brief",
                brief_text.encode("utf-8"),
                file_name=f"{selected_day}_daily_brief_zh.md",
                mime="text/markdown",
            )
        else:
            st.info("这个日期还没有生成中文简报。先运行 `scripts/china_daily_brief.py` 或 Railway 日任务。")

        dashboard_path = Path(output_base) / selected_day / "signals_dashboard.html"
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
            st.dataframe(
                summary_df[
                    [
                        "symbol",
                        "strategy_name",
                        "total_return",
                        "sharpe",
                        "max_drawdown",
                        "win_rate",
                        "profit_factor",
                        "trade_count",
                    ]
                ],
                width="stretch",
            )
            selected_summary = summaries[0]
            equity_path = Path(selected_summary["equity_path"])
            if equity_path.exists():
                equity_df = pd.read_parquet(equity_path)
                if "date" in equity_df.columns:
                    equity_df["date"] = pd.to_datetime(equity_df["date"])
                    st.markdown("**Equity Curve**")
                    st.line_chart(equity_df.set_index("date")[["equity_curve"]], width="stretch")
                    st.markdown("**Drawdown**")
                    st.line_chart(equity_df.set_index("date")[["drawdown"]], width="stretch")

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
