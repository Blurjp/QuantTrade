# QuantTrade — Project Knowledge

## Project Overview

QuantTrade is a satellite-data-driven multi-asset trading signal system. It uses remote sensing data (Sentinel-1 SAR, Sentinel-2 optical, MODIS, etc.) to monitor global supply chains, agricultural yields, energy transport, and retail activity, then generates paper-trading signals (LONG/SHORT/FLAT) for commodities, equities, and ETFs.

## Architecture

- **Pipeline**: `pipeline/run_daily.py` is the main orchestrator. Processes 14 active regions, runs detection, generates signals with persistence/confirmation, builds meta-signals, records daily assets.
- **Signals**: 6 types — chokepoint, retail_parking, oil_storage, agriculture, auto_inventory, port_logistics
- **Regions**: 14 active, 3 meta-groups (global_oil, us_retail, brazil_soy)
- **Paper Trading**: `paper_trading/multi_asset_portfolio.py` manages positions with stop-loss/take-profit
- **Deployment**: Railway via `scheduler_service.py` (HTTP + cron, runs hourly)
- **UI**: Streamlit at `ui/app.py` with LLM chat at `ui/chat.py`

## Wiki Knowledge Base

This project uses the LLM-Wiki pattern. The `wiki/` directory is a persistent, compounding knowledge base maintained by the pipeline.

### Directory Structure

```
wiki/
├── index.md          # Content catalog — auto-updated on every ingest
├── log.md            # Append-only chronological log of wiki operations
├── regions/          # Per-region pages (one .md per region + meta-group pages)
├── signals/          # Per-signal-type pages
├── instruments/      # Per-instrument pages
├── journal/          # Daily trading journal entries (one .md per date)
└── operations/       # Pipeline operations log (one .md per date)
```

### Wiki Operations

**Ingest** (`pipeline/wiki_ingest.ingest_daily_summary`):
- Called automatically at the end of `pipeline/run_daily.py` after generating the daily summary
- Updates each region page with the day's signal (appended to Signal History table)
- Prepends observations to each region's Key Observations section
- Updates meta-group pages with vote scores and constituent breakdowns
- Creates a daily journal entry in `wiki/journal/YYYY-MM-DD.md`
- Creates an operations log entry in `wiki/operations/YYYY-MM-DD.md`
- Updates `index.md` with current signal status for all regions
- Appends to `log.md` with a timestamped entry

**Query** (`pipeline/wiki_ingest.query_wiki` / `build_wiki_context`):
- Keyword search over all wiki pages
- Returns ranked results by term match count
- `build_wiki_context()` produces a context block for the chat LLM system prompt

**Lint** (`pipeline/wiki_ingest.lint_wiki`):
- Checks for missing pages, empty pages, stale observations
- Validates all expected region and signal pages exist
- Returns a list of issues (CRITICAL / WARNING / INFO)

### Page Conventions

- Region pages: `wiki/regions/{region_id}.md` — header blockquote with metadata, Signal History table, Key Observations section, Performance table
- Meta-group pages: `wiki/regions/{group_name}.md` — same structure with constituent breakdown
- Journal pages: `wiki/journal/{YYYY-MM-DD}.md` — actionable signals, all signals table, portfolio state, errors
- Signal history tables: capped at 90 rows (most recent first)
- Observations: capped at 50 lines (most recent first)
- Use Obsidian wiki-links: `[[region_id]]` for cross-references

### When to Update the Wiki

- The ingest runs **automatically** at the end of the daily pipeline
- You can manually re-ingest a specific date: `python -c "from pipeline.wiki_ingest import *; import json; s=json.load(open('outputs/2026-04-15/daily_summary.json')); ingest_daily_summary('2026-04-15', s)"`
- When adding new regions: create a seed page in `wiki/regions/` matching the template
- When modifying signal types: update the corresponding `wiki/signals/` page

## Commands

```bash
# Run daily pipeline (includes wiki ingest)
python -m pipeline.run_daily --date 2026-04-15

# Lint the wiki
python -c "from pipeline.wiki_ingest import lint_wiki; print('\n'.join(lint_wiki()))"

# Search the wiki
python -c "from pipeline.wiki_ingest import query_wiki; [print(n) for n,c in query_wiki('hormuz oil')]"

# Run tests
pytest tests/
```
