#!/usr/bin/env python3
"""Generate a small HTML dashboard for active signals and persistence state."""

import argparse
import json
from html import escape
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate active signals dashboard")
    parser.add_argument("--date", required=True, help="Daily summary date")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    output_base = Path(args.output)
    summary = load_json(output_base / args.date / "daily_summary.json")
    state = load_json(output_base / "signal_persistence_state.json") or {}
    if not summary:
        print("Daily summary not found")
        return 1

    rows = []
    for region_id in sorted(summary.get("signals", {}).keys()):
        signal = summary["signals"][region_id]
        score = signal.get("vote_score", signal.get("ndvi_change", ""))
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else ""
        state_key = f"meta:{signal.get('meta_group')}" if signal.get("meta_group") else f"region:{region_id}"
        persistence = state.get(state_key, {})
        rows.append(
            "<tr>"
            f"<td>{escape(region_id)}</td>"
            f"<td>{escape(str(signal.get('trading_action', 'FLAT')))}</td>"
            f"<td>{escape(str(signal.get('raw_trading_action', signal.get('trading_action', 'FLAT'))))}</td>"
            f"<td>{escape(str(signal.get('confidence', 'Low')))}</td>"
            f"<td>{escape(score_text)}</td>"
            f"<td>{escape(str(signal.get('signal', 'No data')))}</td>"
            f"<td>{escape(str(persistence.get('pending_action', '')))}</td>"
            f"<td>{escape(str(persistence.get('pending_count', '')))}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>QuantTrade Signals {escape(args.date)}</title>
  <style>
    :root {{ --bg:#f5f1e8; --fg:#1f2a2e; --accent:#2b6f6a; --line:#cdbfa8; --card:#fffaf1; }}
    body {{ margin:0; font-family: Georgia, serif; background:linear-gradient(180deg,#f8f5ee,#efe7d6); color:var(--fg); }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:32px 20px 48px; }}
    h1 {{ margin:0 0 8px; font-size:34px; }}
    p {{ margin:0 0 20px; color:#4d5b61; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:18px; box-shadow:0 10px 30px rgba(31,42,46,.08); }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ padding:10px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ color:var(--accent); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
    .footer {{ margin-top:16px; font-size:12px; color:#6d7a7f; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Active Signal Dashboard</h1>
    <p>Date: {escape(args.date)} | Regions processed: {summary.get('regions_processed', 0)} | Successful: {summary.get('regions_successful', 0)}</p>
    <div class=\"card\">
      <table>
        <thead>
          <tr>
            <th>Region</th><th>Action</th><th>Raw</th><th>Confidence</th><th>Score</th><th>Signal</th><th>Pending</th><th>Count</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    <div class=\"footer\">Generated from daily summary and signal persistence state.</div>
  </div>
</body>
</html>
"""

    dashboard_path = output_base / args.date / "signals_dashboard.html"
    dashboard_path.write_text(html)
    print(dashboard_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
