"""
Wiki Ingest Module for QuantTrade Knowledge Base.

Implements the LLM-Wiki pattern: incrementally builds and maintains a structured
wiki from daily pipeline outputs. Three operations:
  - ingest: process a day's outputs and update wiki pages
  - query: search wiki pages for relevant context
  - lint: health-check wiki for inconsistencies
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


WIKI_DIR = Path("wiki")
REGIONS_DIR = WIKI_DIR / "regions"
SIGNALS_DIR = WIKI_DIR / "signals"
INSTRUMENTS_DIR = WIKI_DIR / "instruments"
JOURNAL_DIR = WIKI_DIR / "journal"
OPERATIONS_DIR = WIKI_DIR / "operations"

MAX_SIGNAL_HISTORY_ROWS = 90
MAX_OBSERVATION_LINES = 50

CONFIDENCE_ORDER = {"High": 3, "Medium": 2, "Low": 1}
ACTION_ICONS = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "⚪"}


def _read_page(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _write_page(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _append_to_table(content: str, section_header: str, row: str, max_rows: int = MAX_SIGNAL_HISTORY_ROWS) -> str:
    lines = content.split("\n")
    in_section = False
    table_start = -1
    table_end = -1
    header_line = -1
    separator_line = -1

    for i, line in enumerate(lines):
        if line.strip().startswith(section_header):
            in_section = True
            continue
        if in_section:
            if line.startswith("|") and header_line == -1:
                header_line = i
                continue
            if line.startswith("|-") and separator_line == -1:
                separator_line = i
                table_start = i + 1
                continue
            if table_start != -1 and table_end == -1:
                if not line.startswith("|"):
                    table_end = i
                    break

    if table_start == -1:
        return content

    if table_end == -1:
        table_end = len(lines)

    existing_rows = lines[table_start:table_end]
    new_rows = [row] + existing_rows
    if len(new_rows) > max_rows:
        new_rows = new_rows[:max_rows]

    lines[table_start:table_end] = new_rows
    return "\n".join(lines)


def _prepend_observations(content: str, observation: str) -> str:
    lines = content.split("\n")
    obs_start = -1
    obs_end = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("_Observations accumulated"):
            obs_start = i
            continue
        if obs_start != -1 and obs_end == -1:
            if line.startswith("##") or (line.strip().startswith("_") and i > obs_start):
                obs_end = i
                break

    if obs_start == -1:
        return content

    if obs_end == -1:
        obs_end = len(lines)

    existing_obs = lines[obs_start + 1:obs_end]
    existing_obs = [l for l in existing_obs if l.strip()]
    new_obs = [f"- {observation}"] + existing_obs
    if len(new_obs) > MAX_OBSERVATION_LINES:
        new_obs = new_obs[:MAX_OBSERVATION_LINES]

    lines[obs_start + 1:obs_end] = [""] + new_obs + [""]
    return "\n".join(lines)


def _update_performance_table(content: str, metrics: Dict) -> str:
    lines = content.split("\n")
    in_perf = False
    perf_start = -1
    perf_end = -1

    for i, line in enumerate(lines):
        if line.strip().startswith("## Performance"):
            in_perf = True
            continue
        if in_perf and perf_start == -1:
            if line.startswith("|"):
                perf_start = i
                continue
        if perf_start != -1 and perf_end == -1:
            if not line.startswith("|"):
                perf_end = i
                break

    if perf_start == -1:
        return content

    if perf_end == -1:
        perf_end = len(lines)

    header = lines[perf_start]
    separator = lines[perf_start + 1]
    existing_data = lines[perf_start + 2:perf_end]
    existing_data = [l for l in existing_data if l.strip()]

    metric_keys = [k for k in metrics.keys()]
    new_rows = [f"| {k} | {metrics[k]} |" for k in metric_keys]

    new_section = [header, separator] + new_rows
    lines[perf_start:perf_end] = new_section
    return "\n".join(lines)


def ingest_daily_summary(
    target_date: str,
    summary: Dict,
    output_base: str = "outputs",
) -> None:
    """
    Main ingest: read a day's pipeline summary and update all relevant wiki pages.
    """
    signals = summary.get("signals", {})
    results = summary.get("results", [])

    _update_region_pages(target_date, signals, results)
    _update_meta_group_pages(target_date, signals)
    _update_journal_entry(target_date, summary, output_base)
    _update_operations_log(target_date, summary)
    _update_index(target_date, signals)
    _append_log_entry(target_date, "ingest", f"Processed {summary.get('regions_processed', 0)} regions, {summary.get('regions_successful', 0)} successful")

    print(f"  Wiki ingest complete for {target_date}")


def _update_region_pages(target_date: str, signals: Dict, results: List[Dict]) -> None:
    result_map = {r.get("region", ""): r for r in results}

    for region_id, signal in signals.items():
        if region_id.endswith("_meta"):
            continue

        page_path = REGIONS_DIR / f"{region_id}.md"
        content = _read_page(page_path)
        if not content:
            continue

        action = signal.get("trading_action", "FLAT")
        raw_action = signal.get("raw_trading_action", action)
        confidence = signal.get("confidence", "Low")
        score = signal.get("vote_score") or signal.get("ndvi_change", "n/a")
        signal_text = signal.get("signal", "No data")
        pending = signal.get("pending_count", 0)

        notes = ""
        if action != raw_action:
            notes = f"pending ({raw_action}), {signal.get('pending_count', 0)}/{signal.get('confirmations_required', 2)} confirmations"
        if pending > 0:
            notes = notes or f"pending confirmation {pending}/{signal.get('confirmations_required', 2)}"

        row = f"| {target_date} | {action} | {confidence} | {score} | {notes} |"
        content = _append_to_table(content, "## Signal History", row)

        observation = f"**{target_date}:** {ACTION_ICONS.get(action, '')} {signal_text} (confidence: {confidence}, score: {score})"
        if notes:
            observation += f" — {notes}"
        content = _prepend_observations(content, observation)

        result = result_map.get(region_id, {})
        status = result.get("status", "unknown")
        perf_metrics = {}
        if status == "error":
            perf_metrics["Last run status"] = f"error: {result.get('message', 'unknown')}"
        else:
            perf_metrics["Last run status"] = "success"
            perf_metrics["Last signal action"] = action
            perf_metrics["Last signal confidence"] = confidence

        if perf_metrics:
            content = _update_performance_table(content, perf_metrics)

        _write_page(page_path, content)


def _update_meta_group_pages(target_date: str, signals: Dict) -> None:
    for key, signal in signals.items():
        if not key.endswith("_meta"):
            continue

        group_name = key.replace("_meta", "")
        page_path = REGIONS_DIR / f"{group_name}.md"
        content = _read_page(page_path)
        if not content:
            continue

        action = signal.get("trading_action", "FLAT")
        confidence = signal.get("confidence", "Low")
        vote_score = signal.get("vote_score", "n/a")
        constituents = signal.get("constituents", [])

        key_driver = ""
        if constituents:
            best = max(constituents, key=lambda c: CONFIDENCE_ORDER.get(c.get("confidence", "Low"), 0))
            key_driver = f"{best.get('region', '?')} ({best.get('action', 'FLAT')}, {best.get('confidence', 'Low')})"

        row = f"| {target_date} | {vote_score} | {action} | {confidence} | {key_driver} |"
        content = _append_to_table(content, "## Meta-Signal History", row)

        observation = f"**{target_date}:** {ACTION_ICONS.get(action, '')} Vote={vote_score:.2f}, {confidence} — driven by {key_driver}"
        content = _prepend_observations(content, observation)

        perf_metrics = {
            "Last meta-action": action,
            "Last confidence": confidence,
            "Last vote score": f"{vote_score:.3f}",
        }
        content = _update_performance_table(content, perf_metrics)

        _write_page(page_path, content)


def _update_journal_entry(target_date: str, summary: Dict, output_base: str) -> None:
    signals = summary.get("signals", {})
    results = summary.get("results", [])
    regions_ok = summary.get("regions_successful", 0)
    regions_total = summary.get("regions_processed", 0)

    actionable = [
        (rid, sig) for rid, sig in signals.items()
        if sig.get("actionability") == "Actionable"
    ]

    lines = [
        f"# Trading Journal — {target_date}",
        "",
        f"**Pipeline:** {regions_ok}/{regions_total} regions successful",
        "",
        "## Actionable Signals",
        "",
    ]

    if not actionable:
        lines.append("_No actionable signals today._")
    else:
        for rid, sig in actionable:
            action = sig.get("trading_action", "FLAT")
            instruments = ", ".join(sig.get("instruments", []))
            confidence = sig.get("confidence", "Low")
            signal_text = sig.get("signal", "")
            lines.append(f"- **{rid}** → {ACTION_ICONS.get(action, '')} **{action}** [{confidence}] — {signal_text} → {instruments}")

    lines.append("")
    lines.append("## All Signals")
    lines.append("")
    lines.append("| Region | Action | Raw | Confidence | Score |")
    lines.append("|--------|--------|-----|------------|-------|")
    for rid, sig in signals.items():
        action = sig.get("trading_action", "FLAT")
        raw = sig.get("raw_trading_action", action)
        conf = sig.get("confidence", "Low")
        score = sig.get("vote_score") or sig.get("ndvi_change", "n/a")
        lines.append(f"| {rid} | {action} | {raw} | {conf} | {score} |")

    lines.append("")
    lines.append("## Errors")
    lines.append("")
    errors = [r for r in results if r.get("status") == "error"]
    if not errors:
        lines.append("_No errors._")
    else:
        for err in errors:
            lines.append(f"- **{err.get('region', '?')}**: {err.get('message', 'unknown error')}")

    portfolio_path = Path(output_base) / "paper_trading" / "multi_asset_portfolio.json"
    if portfolio_path.exists():
        try:
            portfolio = json.loads(portfolio_path.read_text())
            positions = portfolio.get("positions", {})
            total_value = portfolio.get("total_value", portfolio.get("cash", 0))
            lines.append("")
            lines.append("## Portfolio State")
            lines.append("")
            lines.append(f"- **Total Value:** ${total_value:,.2f}")
            lines.append(f"- **Cash:** ${portfolio.get('cash', 0):,.2f}")
            if positions:
                lines.append(f"- **Open Positions:** {len(positions)}")
                for ticker, pos in positions.items():
                    direction = pos.get("direction", "?")
                    entry = pos.get("entry_price", 0)
                    lines.append(f"  - {ticker} ({direction}) @ ${entry:.2f}")
        except (json.JSONDecodeError, TypeError):
            pass

    report_path = Path(output_base) / "daily_reports" / f"report_{target_date}.md"
    if report_path.exists():
        lines.append("")
        lines.append("## Daily Report")
        lines.append("")
        lines.append(f"_See: `outputs/daily_reports/report_{target_date}.md`_")

    ledger_path = Path(output_base) / "execution" / "orders.sqlite"
    if ledger_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(ledger_path))
            conn.row_factory = sqlite3.Row
            day_orders = conn.execute(
                "SELECT symbol, side, status, quantity, notional, limit_price, created_at "
                "FROM orders WHERE date(created_at) = ? ORDER BY created_at",
                (target_date,),
            ).fetchall()
            day_risk = conn.execute(
                "SELECT COUNT(*) as c FROM risk_decisions WHERE approved=0 AND date(created_at) = ?",
                (target_date,),
            ).fetchone()["c"]
            conn.close()

            if day_orders:
                lines.append("")
                lines.append("## Execution Orders")
                lines.append("")
                lines.append("| Symbol | Side | Status | Qty | Notional | Time |")
                lines.append("|--------|------|--------|-----|----------|------|")
                for o in day_orders:
                    ts = o["created_at"][11:16] if o["created_at"] else ""
                    qty = f"{o['quantity']:.1f}" if o["quantity"] else ""
                    notional = f"${o['notional']:.0f}" if o["notional"] else ""
                    lines.append(f"| {o['symbol']} | {o['side']} | {o['status']} | {qty} | {notional} | {ts} |")
                if day_risk > 0:
                    lines.append(f"\n_{day_risk} order(s) rejected by risk gate._")
        except Exception:
            pass

    _write_page(JOURNAL_DIR / f"{target_date}.md", "\n".join(lines))


def _update_operations_log(target_date: str, summary: Dict) -> None:
    regions_ok = summary.get("regions_successful", 0)
    regions_total = summary.get("regions_processed", 0)
    errors = [r for r in summary.get("results", []) if r.get("status") == "error"]

    lines = [
        f"# Operations — {target_date}",
        "",
        f"**Status:** {'OK' if regions_ok == regions_total else 'PARTIAL'} ({regions_ok}/{regions_total})",
        "",
    ]

    if errors:
        lines.append("## Errors")
        lines.append("")
        for err in errors:
            lines.append(f"- `{err.get('region', '?')}`: {err.get('message', 'unknown')}")
        lines.append("")

    signals = summary.get("signals", {})
    actionable_count = sum(1 for s in signals.values() if s.get("actionability") == "Actionable")
    lines.append(f"**Actionable signals:** {actionable_count}")

    _write_page(OPERATIONS_DIR / f"{target_date}.md", "\n".join(lines))


def _update_index(target_date: str, signals: Dict) -> None:
    index_path = WIKI_DIR / "index.md"
    content = _read_page(index_path)
    if not content:
        return

    from pipeline.regions import load_registry
    try:
        registry = load_registry()
        regions_config = registry.get("regions", {})
        meta_groups = registry.get("meta_groups", {})
    except Exception:
        regions_config = {}
        meta_groups = {}

    lines = content.split("\n")
    new_lines = []
    section = None

    for line in lines:
        stripped = line.strip()

        if stripped == "## Regions":
            section = "regions"
            new_lines.append(line)
            new_lines.append("")
            new_lines.append("| Region | Type | Instruments | Signal Status | Last Updated |")
            new_lines.append("|--------|------|-------------|---------------|--------------|")
            for region_id, cfg in regions_config.items():
                if not cfg.get("active", True):
                    continue
                sig = signals.get(region_id, {})
                action = sig.get("trading_action", "FLAT")
                instruments = ", ".join(cfg.get("instruments", []))
                rtype = cfg.get("type", "?")
                new_lines.append(f"| [[{region_id}]] | {rtype} | {instruments} | {ACTION_ICONS.get(action, '')} {action} | {target_date} |")
            new_lines.append("")
            section = None
            continue

        if stripped == "## Meta-Signal Groups":
            section = "meta"
            new_lines.append(line)
            new_lines.append("")
            new_lines.append("| Group | Constituents | Instruments | Current Action | Last Updated |")
            new_lines.append("|-------|-------------|-------------|----------------|--------------|")
            for group_name, gcfg in meta_groups.items():
                meta_sig = signals.get(f"{group_name}_meta", {})
                action = meta_sig.get("trading_action", "FLAT")
                instruments = ", ".join(gcfg.get("instruments", []))
                new_lines.append(f"| [[{group_name}]] | — | {instruments} | {ACTION_ICONS.get(action, '')} {action} | {target_date} |")
            new_lines.append("")
            section = None
            continue

        if section == "regions":
            if stripped.startswith("|") or stripped == "":
                continue
            section = None

        if section == "meta":
            if stripped.startswith("|") or stripped == "":
                continue
            section = None

        new_lines.append(line)

    _write_page(index_path, "\n".join(new_lines))


def _append_log_entry(target_date: str, operation: str, details: str) -> None:
    log_path = WIKI_DIR / "log.md"
    content = _read_page(log_path)
    if not content:
        return

    entry = f"## [{target_date}] {operation} | {details}"
    content = content.rstrip() + "\n\n" + entry + "\n"
    _write_page(log_path, content)


def query_wiki(query: str, wiki_dir: str = "wiki") -> List[Tuple[str, str]]:
    """
    Simple keyword search over wiki pages. Returns list of (page_name, content) tuples.
    """
    wiki_path = Path(wiki_dir)
    if not wiki_path.exists():
        return []

    query_lower = query.lower()
    query_terms = set(re.split(r"\s+", query_lower)) - {"the", "a", "an", "is", "of", "for", "in", "on", "to", "and", "or"}

    results = []
    for md_file in sorted(wiki_path.rglob("*.md")):
        content = _read_page(md_file)
        if not content:
            continue
        content_lower = content.lower()
        matches = sum(1 for term in query_terms if term in content_lower)
        if matches > 0:
            page_name = str(md_file.relative_to(wiki_path))
            results.append((page_name, content, matches))

    results.sort(key=lambda x: -x[2])
    return [(name, content) for name, content, _ in results]


def build_wiki_context(query: str, wiki_dir: str = "wiki", max_pages: int = 5, max_chars: int = 6000) -> str:
    """
    Build a context block from wiki pages relevant to a query.
    For use in ui/chat.py system prompt.
    """
    pages = query_wiki(query, wiki_dir)
    if not pages:
        return ""

    parts = ["\n--- Wiki Context ---"]
    total_chars = 0
    for page_name, content in pages[:max_pages]:
        if total_chars >= max_chars:
            break
        trimmed = content[:max_chars - total_chars]
        parts.append(f"\n### [{page_name}]\n{trimmed}")
        total_chars += len(trimmed)

    parts.append("\n--- End Wiki Context ---")
    return "\n".join(parts)


def lint_wiki(wiki_dir: str = "wiki") -> List[str]:
    """
    Health-check the wiki. Returns list of issues found.
    """
    wiki_path = Path(wiki_dir)
    issues = []

    index_content = _read_page(wiki_path / "index.md")
    if not index_content:
        issues.append("CRITICAL: wiki/index.md is missing or empty")

    log_content = _read_page(wiki_path / "log.md")
    if not log_content:
        issues.append("CRITICAL: wiki/log.md is missing or empty")
    elif len(log_content.strip().split("\n")) < 3:
        issues.append("WARNING: wiki/log.md has no entries — wiki has not been ingested into yet")

    for region_file in (wiki_path / "regions").glob("*.md"):
        content = _read_page(region_file)
        if not content:
            issues.append(f"EMPTY: {region_file.name} is empty")
            continue
        if "_Observations accumulated" in content and content.count("_Observations accumulated") > 0:
            obs_section = content.split("_Observations accumulated")[1].split("##")[0] if "##" in content.split("_Observations accumulated")[1] else content.split("_Observations accumulated")[1]
            if not any(l.strip().startswith("- **") for l in obs_section.split("\n")):
                issues.append(f"INFO: {region_file.name} has no observations yet")

    for signal_file in (wiki_path / "signals").glob("*.md"):
        if not _read_page(signal_file).strip():
            issues.append(f"EMPTY: signals/{signal_file.name} is empty")

    expected_regions = [
        "hormuz", "suez", "malacca", "bab_el_mandeb", "panama_canal",
        "walmart_hq", "costco_hq", "cushing", "iowa_corn",
        "detroit_auto", "la_longbeach",
        "brazil_soy_north", "brazil_soy_central", "brazil_soy_southeast",
        "global_oil", "us_retail", "brazil_soy",
    ]
    for region_id in expected_regions:
        if not (wiki_path / "regions" / f"{region_id}.md").exists():
            issues.append(f"MISSING: wiki/regions/{region_id}.md does not exist")

    expected_signals = ["chokepoint", "retail_parking", "oil_storage", "agriculture", "auto_inventory", "port_logistics"]
    for sig_type in expected_signals:
        if not (wiki_path / "signals" / f"{sig_type}.md").exists():
            issues.append(f"MISSING: wiki/signals/{sig_type}.md does not exist")

    return issues
