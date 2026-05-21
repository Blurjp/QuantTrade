"""
Scheduled pipeline runner for Railway deployment.

Runs the QuantTrade pipeline every hour automatically.
Includes satellite data monitoring (vegetation, SST, atmospheric, etc.)
"""
import os
import sys
import time
import json
import logging
import schedule
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Global state for health checks
last_run_time = None
last_run_status = "never"
pipeline_runs = 0
OUTPUT_BASE = "outputs"
pipeline_lock = threading.Lock()
state_lock = threading.Lock()
shutdown_event = threading.Event()


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for Railway health checks and API."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def do_GET(self):
        global last_run_time, last_run_status, pipeline_runs

        if self.path == "/health" or self.path == "/":
            import resource
            try:
                with open("/proc/self/status") as f:
                    proc_status = dict(
                        line.split(":", 1)
                        for line in f.read().strip().split("\n")
                        if ":" in line
                    )
            except Exception:
                proc_status = {}
            try:
                with open("/proc/meminfo") as f:
                    meminfo = dict(
                        line.split(":", 1)
                        for line in f.read().strip().split("\n")
                        if ":" in line
                    )
            except Exception:
                meminfo = {}
            with state_lock:
                snap_run_time = last_run_time
                snap_status = last_run_status
                snap_runs = pipeline_runs
            response = {
                "status": "healthy",
                "service": "quanttrade-scheduler",
                "last_run": snap_run_time,
                "last_status": snap_status,
                "total_runs": snap_runs,
                "interval_minutes": int(os.environ.get("PIPELINE_INTERVAL_MINUTES", "60")),
                "memory": {
                    "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1) if sys.platform != "darwin" else round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1),
                    "vm_rss_kb": proc_status.get("VmRSS", "?").strip(),
                    "vm_size_kb": proc_status.get("VmSize", "?").strip(),
                    "vm_peak_kb": proc_status.get("VmPeak", "?").strip(),
                    "mem_available": meminfo.get("MemAvailable", "?").strip(),
                    "mem_total": meminfo.get("MemTotal", "?").strip(),
                    "cached": meminfo.get("Cached", "?").strip(),
                },
            }
            self._send_json(response)

        elif self.path == "/api/summary":
            # Serve the latest daily summary
            summary = self._get_latest_summary()
            if summary:
                self._send_json(summary)
            else:
                self._send_json({"error": "No summary available"}, 404)

        elif self.path == "/api/all-signals":
            # Aggregate latest signals from all pipeline modules
            # Maps raw pipeline fields to dashboard-expected format
            all_signals = {}
            output_root = Path(OUTPUT_BASE)
            for module_dir in output_root.iterdir():
                if not module_dir.is_dir():
                    continue
                for sig_file in sorted(module_dir.glob("signal_*_2026-*.json")):
                    try:
                        sig = json.loads(sig_file.read_text())
                        region_id = sig.get("region_id", sig_file.stem)
                        # Only keep latest per region
                        if region_id not in all_signals or sig.get("date", "") > all_signals[region_id].get("date", ""):
                            # Map fields to dashboard format
                            direction = sig.get("direction", "neutral")
                            conf = sig.get("confidence", 0)
                            conf_label = sig.get("confidence_label", "")
                            if not conf_label:
                                conf_label = "High" if conf >= 70 else "Medium" if conf >= 50 else "Low"
                            
                            # Build signal string
                            signal_str = sig.get("signal", "")
                            if not signal_str:
                                if direction == "long":
                                    signal_str = "Long disruption risk"
                                elif direction == "short":
                                    signal_str = "Short disruption risk"
                                else:
                                    signal_str = "Normal throughput"
                            
                            # Build actionability
                            actionability = sig.get("actionability", "")
                            if not actionability:
                                actionability = "Actionable" if conf >= 70 else "Watchlist" if conf >= 50 else "Ignore"
                            
                            mapped = {
                                "region_id": region_id,
                                "region_name": sig.get("region_name", region_id),
                                "date": sig.get("date", ""),
                                "signal": signal_str,
                                "direction": direction,
                                "confidence": conf_label,
                                "confidence_score": conf,
                                "actionability": actionability,
                                "coverage_score": sig.get("coverage_score"),
                                "signal_strength": sig.get("signal_strength"),
                                "reroute_flag": sig.get("reroute_flag", False),
                                "instruments": sig.get("instruments", []),
                                "rationale": sig.get("rationale", ""),
                                "signal_type": sig.get("signal_type", module_dir.name),
                            }
                            all_signals[region_id] = mapped
                    except Exception:
                        logger.warning("Failed to load signals from %s", module_dir.name)
            self._send_json({"date": "latest", "signals": all_signals})

        elif self.path == "/api/signals":
            # Serve the latest signals
            summary = self._get_latest_summary()
            if summary:
                self._send_json({"date": summary.get("date"), "signals": summary.get("signals", {})})
            else:
                self._send_json({"error": "No signals available"}, 404)

        elif self.path.startswith("/api/summary/"):
            # Serve summary for a specific date
            date_str = self.path.split("/")[-1]
            summary = self._get_summary_for_date(date_str)
            if summary:
                self._send_json(summary)
            else:
                self._send_json({"error": f"No summary for {date_str}"}, 404)

        elif self.path == "/api/portfolio":
            # Serve current portfolio state with REAL-TIME unrealized P&L
            portfolio_file = Path(OUTPUT_BASE) / "paper_trading" / "multi_asset_portfolio.json"
            if not portfolio_file.exists():
                self._send_json({"error": "No portfolio file"}, 404)
                return

            data = json.loads(portfolio_file.read_text())
            positions = data.get("positions", {})
            if positions:
                try:
                    import yfinance as yf
                    tickers = list(positions.keys())
                    batch = yf.download(tickers, period="1d", progress=False)
                    if not batch.empty and "Close" in batch.columns:
                        close = batch["Close"]
                        for ticker, pos in positions.items():
                            try:
                                if len(tickers) == 1:
                                    price = float(close.iloc[-1])
                                elif ticker in close.columns:
                                    series = close[ticker].dropna()
                                    if len(series) == 0:
                                        continue
                                    price = float(series.iloc[-1])
                                else:
                                    continue

                                if price != price:
                                    continue

                                entry = float(pos.get("entry_price", 0))
                                qty = float(pos.get("quantity", 0))
                                direction = pos.get("direction", "long")

                                if direction == "long":
                                    unrealized = qty * (price - entry)
                                else:
                                    unrealized = qty * (entry - price)

                                pos["current_price"] = round(price, 2)
                                pos["unrealized_pnl"] = round(unrealized, 2)
                                if direction == "long":
                                    pos["position_value"] = round(qty * price, 2)
                            except (IndexError, ValueError, TypeError) as e:
                                logger.debug(f"Price lookup failed for {ticker}: {e}")
                                continue
                except Exception as e:
                    logger.warning(f"Live price refresh failed for /api/portfolio: {e}")
                    # Serve stale data rather than fail

            self._send_json(data)

        elif self.path == "/api/prices":
            # Fetch current prices for portfolio tickers via batch yfinance
            try:
                import yfinance as yf
                portfolio_file = Path(OUTPUT_BASE) / "paper_trading" / "multi_asset_portfolio.json"
                tickers = []
                if portfolio_file.exists():
                    data = json.loads(portfolio_file.read_text())
                    tickers = list(data.get("positions", {}).keys())
                if tickers:
                    batch = yf.download(tickers, period="1d", progress=False)
                    prices = {}
                    if not batch.empty and "Close" in batch.columns:
                        close = batch["Close"]
                        try:
                            if len(tickers) == 1:
                                val = float(close.iloc[-1])
                                if val == val:
                                    prices[tickers[0]] = val
                            else:
                                for t in tickers:
                                    if t in close.columns:
                                        series = close[t].dropna()
                                        if len(series) > 0:
                                            val = float(series.iloc[-1])
                                            if val == val:
                                                prices[t] = val
                        except (IndexError, ValueError, TypeError) as e:
                            logger.debug(f"Price extraction failed: {e}")
                    self._send_json(prices)
                else:
                    self._send_json({})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif self.path == "/api/positions":
            # Serve current positions summary
            portfolio_file = Path(OUTPUT_BASE) / "paper_trading" / "multi_asset_portfolio.json"
            if portfolio_file.exists():
                data = json.loads(portfolio_file.read_text())
                self._send_json({
                    "cash": data.get("cash", 0),
                    "positions": {k: {
                        "ticker": v.get("ticker"),
                        "direction": v.get("direction"),
                        "entry_price": v.get("entry_price"),
                        "quantity": v.get("quantity"),
                        "position_value": v.get("position_value"),
                        "unrealized_pnl": v.get("unrealized_pnl", 0),
                        "rationale": v.get("rationale", ""),
                    } for k, v in data.get("positions", {}).items()},
                    "total_trades": len(data.get("trades", [])),
                })
            else:
                self._send_json({"error": "No portfolio"}, 404)

        elif self.path == "/api/dates":
            # List available dates
            dates = self._list_available_dates()
            self._send_json({"dates": dates})

        elif self.path == "/api/learning":
            # Feedback learning summary
            try:
                from pipeline.signal_feedback_learner import get_summary
                summary = get_summary(output_base=OUTPUT_BASE)
                self._send_json(summary)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif self.path == "/api/reconcile":
            try:
                from execution.service import ExecutionService
                from execution.reconciler import Reconciler
                svc = ExecutionService()
                reconciler = Reconciler(svc)
                report = reconciler.run()
                self._send_json({
                    "stranded_found": report.stranded_found,
                    "stranded_cancelled": report.stranded_cancelled,
                    "stranded_resubmitted": report.stranded_resubmitted,
                    "strand_resubmit_failed": report.strand_resubmit_failed,
                    "drift_found": report.drift_found,
                    "drift_updated": report.drift_updated,
                    "has_alert": report.has_alert,
                    "errors": report.errors,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif self.path.startswith("/api/outputs/"):
            # Serve any file from outputs/ directory
            from urllib.parse import unquote
            rel_path = unquote(self.path[len("/api/outputs/"):])
            # Prevent path traversal: resolve and verify it's within OUTPUT_BASE
            base_path = Path(OUTPUT_BASE).resolve()
            file_path = (base_path / rel_path).resolve()
            if not str(file_path).startswith(str(base_path)):
                self._send_json({"error": "Access denied"}, 403)
            elif file_path.exists() and file_path.is_file():
                try:
                    content = file_path.read_text()
                    import json as _json
                    if rel_path.endswith(".json"):
                        self._send_json(_json.loads(content))
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(content.encode())
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            else:
                self._send_json({"error": "File not found", "path": rel_path}, 404)

        elif self.path == "/api/outputs-list":
            # List all files in outputs/ directory
            import glob as _glob
            output_path = Path(OUTPUT_BASE)
            files = [str(f.relative_to(output_path)) for f in output_path.rglob("*") if f.is_file()]
            self._send_json({"files": files[:500]})  # Limit to 500 files

        elif self.path == "/debug/cwd":
            import os as _os
            self._send_json({"cwd": _os.getcwd(), "output_base": OUTPUT_BASE})

        elif self.path == "/debug/backfill":
            # List available backfill files
            backfill_dir = Path(OUTPUT_BASE) / "backfill"
            files = list(backfill_dir.glob("*.json")) if backfill_dir.exists() else []
            self._send_json({"backfill_files": [f.name for f in files], "backfill_dir": str(backfill_dir), "exists": backfill_dir.exists()})

        elif self.path == "/debug/regions":
            # List active regions for debugging
            try:
                from pipeline.regions import get_active_regions
                regions = get_active_regions()
                self._send_json({
                    "active_regions": list(regions.keys()),
                    "count": len(regions),
                })
            except Exception as e:
                import traceback
                self._send_json({"error": str(e), "traceback": traceback.format_exc()}, 500)

        elif self.path == "/debug/trigger-backfill":
            # Manually trigger backfill and return detailed results
            import traceback as tb
            try:
                from pipeline.regions import get_active_regions
                from scripts.run_backfill import run_auto_backfill

                backfill_dir = Path(OUTPUT_BASE) / "backfill"
                backfill_dir.mkdir(parents=True, exist_ok=True)

                regions = get_active_regions()
                region_ids = list(regions.keys())

                # Check which regions need backfill
                missing = []
                for rid in region_ids:
                    bf_file = backfill_dir / f"{rid}_backfill.json"
                    if not bf_file.exists():
                        missing.append(rid)

                if not missing:
                    self._send_json({
                        "status": "all_backfill_exists",
                        "regions": region_ids,
                        "missing": [],
                    })
                    return

                # Run backfill
                result = run_auto_backfill(
                    regions_filter=missing,
                    output_base=OUTPUT_BASE,
                )

                # Check files after
                files = list(backfill_dir.glob("*.json"))

                self._send_json({
                    "status": "backfill_complete",
                    "regions": region_ids,
                    "missing": missing,
                    "result": result,
                    "files_created": [f.name for f in files],
                })
            except Exception as e:
                self._send_json({
                    "error": str(e),
                    "traceback": tb.format_exc(),
                }, 500)

        elif self.path == "/api/signals/history":
            # Serve signal tracking history
            try:
                from tracking.signal_tracker import SignalTracker
                tracker = SignalTracker(output_base=OUTPUT_BASE)
                self._send_json({
                    "stats": tracker.get_stats(),
                    "history": tracker.get_history(),
                })
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif self.path == "/api/signals/stats":
            # Signal accuracy stats only
            try:
                from tracking.signal_tracker import SignalTracker
                tracker = SignalTracker(output_base=OUTPUT_BASE)
                self._send_json(tracker.get_stats())
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/api/discord/command":
            # Handle Discord bot commands
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                command = data.get("command", "")
                if not command:
                    self._send_json({"error": "No command provided"}, 400)
                    return

                from discord_bot.bot import DiscordCommandHandler
                handler = DiscordCommandHandler(output_base=OUTPUT_BASE)
                result = handler.handle_command(command)

                # Send response via webhook
                handler.process_and_respond(command)

                self._send_json({"status": "ok", "result": result})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif self.path == "/api/portfolio/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                portfolio_file = Path(OUTPUT_BASE) / "paper_trading" / "multi_asset_portfolio.json"
                portfolio_file.parent.mkdir(parents=True, exist_ok=True)
                portfolio_file.write_text(json.dumps(data, indent=2))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "message": "Portfolio uploaded"}).encode())
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def _get_latest_summary(self):
        """Get the most recent daily summary."""
        output_root = Path(OUTPUT_BASE)
        candidates = sorted(
            path.parent.name for path in output_root.glob("*/daily_summary.json")
            if path.parent.name[:4].isdigit()
        )
        if not candidates:
            return None
        latest = candidates[-1]
        return self._get_summary_for_date(latest)

    def _get_summary_for_date(self, date_str: str):
        """Get summary for a specific date."""
        summary_path = Path(OUTPUT_BASE) / date_str / "daily_summary.json"
        if not summary_path.exists():
            return None
        try:
            return json.loads(summary_path.read_text())
        except Exception:
            return None

    def _list_available_dates(self):
        """List all available summary dates."""
        output_root = Path(OUTPUT_BASE)
        return sorted(
            path.parent.name for path in output_root.glob("*/daily_summary.json")
            if path.parent.name[:4].isdigit()
        )


def start_health_server(port=8080):
    """Start HTTP server for health checks in a background thread."""
    from http.server import ThreadingHTTPServer
    import socket
    class ReusableHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True
        allow_reuse_port = True
    try:
        server = ReusableHTTPServer(("0.0.0.0", port), HealthCheckHandler)
    except OSError:
        port = port + 1
        logger.warning(f"Port {port-1} in use, trying {port}")
        server = ReusableHTTPServer(("0.0.0.0", port), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health check server started on port {port} (ThreadingHTTPServer)")
    return server


def run_satellite_monitors(target_date: str, output_base: str = "outputs"):
    """
    Run all satellite monitoring modules in parallel.

    Automatically uses real satellite data when available (auto-detected).
    """
    from pipeline.satellite_data import get_capabilities
    import concurrent.futures

    caps = get_capabilities()
    logger.info(f"Satellite data capabilities: Real={caps['real_data_enabled']}, PC={caps['planetary_computer']['available']}")

    def _run_module(name, module_path, class_name, method="generate_all_signals", **kwargs):
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            monitor = cls(output_base=output_base)
            fn = getattr(monitor, method)
            signals = fn(target_date) if method == "generate_all_signals" else fn()
            return name, signals
        except Exception as e:
            logger.warning(f"{name} failed: {e}")
            return name, []

    module_defs = [
        ("Vegetation health", "pipeline.vegetation_health", "VegetationHealthMonitor"),
        ("SST", "pipeline.sea_surface_temperature", "SeaSurfaceTemperatureMonitor"),
        ("Solar irradiance", "pipeline.solar_irradiance", "SolarIrradianceMonitor"),
        ("Atmospheric", "pipeline.atmospheric", "AtmosphericMonitor"),
        ("Thermal infrared", "pipeline.thermal_infrared", "ThermalInfraredMonitor"),
        ("Nighttime lights", "pipeline.nighttime_lights", "NighttimeLightsMonitor"),
        ("Precipitation", "pipeline.precipitation", "PrecipitationMonitor"),
    ]

    all_signals = []

    max_workers = max(1, int(os.environ.get("SATELLITE_MAX_WORKERS", "1")))
    logger.info(f"Satellite monitor workers: {max_workers}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_module, name, mod, cls): name
            for name, mod, cls in module_defs
        }
        for future in concurrent.futures.as_completed(futures):
            name, signals = future.result()
            all_signals.extend(signals)
            logger.info(f"{name}: {len(signals)} signals")

    # Cattle Feedlot (separate — different API)
    try:
        from pipeline.cattle_feedlot import CattleFeedlotMonitor
        cattle_monitor = CattleFeedlotMonitor(output_base=output_base)
        cattle_signals = cattle_monitor.generate_signal()
        all_signals.extend(cattle_signals)
        logger.info(f"Cattle feedlot: {len(cattle_signals)} signals")
    except Exception as e:
        logger.warning(f"Cattle feedlot failed: {e}")

    actionable = sum(1 for s in all_signals if s.get("direction") != "neutral")
    logger.info(f"Satellite monitors total: {len(all_signals)} signals, {actionable} actionable")

    return all_signals


def send_discord_report(content: str) -> bool:
    """Send a report to Discord via webhook."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.debug("No DISCORD_WEBHOOK_URL, skipping Discord report")
        return False

    try:
        import requests
        response = requests.post(webhook_url, json={"content": content})
        response.raise_for_status()
        logger.info("Discord report sent")
        return True
    except Exception as e:
        logger.error("Discord report failed: %s", e)
        return False


def load_top_signals(output_dir: Path, limit: int = 5) -> list:
    """Load top actionable signals from output directory."""
    signals = []
    for signal_file in sorted(output_dir.rglob("signal_*.json"), reverse=True):
        try:
            sig = json.loads(signal_file.read_text())
            if sig.get("confidence", 0) >= 70:
                signals.append(sig)
                if len(signals) >= limit:
                    break
        except Exception:
            continue
    return signals


def check_stop_loss_take_profit(portfolio, prices, signal_tracker=None):
    """
    Check all open positions against current prices.
    Auto-close if loss exceeds stop-loss or profit exceeds take-profit.
    Sends email + Discord notification on close.

    Thresholds configurable via env vars:
        STOP_LOSS_PCT (default -8)
        TAKE_PROFIT_PCT (default +15)
    """
    stop_loss_pct = float(os.environ.get("STOP_LOSS_PCT", "-8"))
    take_profit_pct = float(os.environ.get("TAKE_PROFIT_PCT", "15"))

    # Convert to decimal (positive values)
    sl_threshold = abs(stop_loss_pct) / 100.0
    tp_threshold = abs(take_profit_pct) / 100.0

    closed_positions = []

    for ticker in list(portfolio.positions.keys()):
        pos = portfolio.positions[ticker]
        current_price = prices.get(ticker)
        if not current_price or current_price <= 0:
            continue

        # Update current price on position for dashboard display
        pos.current_price = current_price

        # Calculate P&L percentage
        if pos.direction == "long":
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price

        reason = None
        if pnl_pct <= -sl_threshold:
            reason = f"Stop-loss triggered ({pnl_pct*100:+.1f}% vs -{sl_threshold*100:.0f}% limit)"
        elif pnl_pct >= tp_threshold:
            reason = f"Take-profit triggered ({pnl_pct*100:+.1f}% vs +{tp_threshold*100:.0f}% limit)"

        if reason:
            trade = portfolio.close_position(ticker, current_price, reason)
            if trade:
                closed_positions.append((ticker, trade, reason))
                logger.info(f"AUTO-CLOSE {ticker}: {reason} @ ${current_price:.2f} P&L=${trade.pnl:+.2f}")

                # Also close on Alpaca if paper/live trading
                try:
                    from execution.service import ExecutionService
                    from execution.models import (
                        OrderIntent, OrderSide, OrderType,
                        TimeInForce, PositionIntent,
                    )
                    from datetime import datetime as _dt, timezone as _tz
                    exec_svc = ExecutionService()
                    if hasattr(exec_svc.broker, 'get_positions'):
                        broker_pos = {p.symbol: p for p in exec_svc.broker.get_positions()}
                        if ticker in broker_pos:
                            bp = broker_pos[ticker]
                            qty_to_close = float(bp.qty)
                            if qty_to_close > 0:
                                close_intent = OrderIntent(
                                    symbol=ticker,
                                    side=OrderSide.SELL,
                                    quantity=qty_to_close,
                                    order_type=OrderType.MARKET,
                                    time_in_force=TimeInForce.DAY,
                                    client_order_id=exec_svc.make_client_order_id(
                                        "sl", ticker, "sell",
                                        _dt.now(_tz.utc).strftime("%Y%m%d%H%M"),
                                    ),
                                    created_at=_dt.now(_tz.utc),
                                    position_intent=PositionIntent.CLOSE_POSITION,
                                    rationale=reason,
                                )
                                close_result = exec_svc.submit(close_intent)
                                logger.info(f"Alpaca close {ticker}: {close_result.status.value}")
                except Exception as alp_err:
                    logger.warning(f"Alpaca close failed for {ticker}: {alp_err}")

                # Track in signal history
                if signal_tracker:
                    try:
                        signal_tracker.record_position_closed(ticker, current_price, trade.pnl)
                    except Exception:
                        pass

                # Record outcome for feedback learning
                try:
                    from pipeline.signal_feedback_learner import record_trade_outcome
                    pos_data = portfolio.positions.get(ticker) if hasattr(portfolio, 'positions') else None
                    region_id = getattr(pos_data, 'region_id', '') if pos_data else ''
                    asset_class = getattr(pos_data, 'asset_class', 'unknown') if pos_data else 'unknown'
                    record_trade_outcome(
                        region_id=region_id or ticker,
                        signal_type=asset_class,
                        direction=getattr(pos_data, 'direction', 'long') if pos_data else 'long',
                        entry_price=pos.entry_price if hasattr(pos, 'entry_price') else 0,
                        exit_price=current_price,
                        pnl=trade.pnl,
                        rationale=reason,
                        output_base="outputs",
                    )
                    logger.info(f"Feedback learning recorded: {ticker} {'WIN' if trade.pnl >= 0 else 'LOSS'} ${trade.pnl:+.2f}")
                except Exception as learn_err:
                    logger.warning(f"Feedback learning failed on close: {learn_err}")

    # Send notifications for closed positions
    if closed_positions:
        try:
            from notifications.notification_manager import NotificationManager
            nm = NotificationManager()

            for ticker, trade, reason in closed_positions:
                pnl_emoji = "+" if trade.pnl >= 0 else ""

                # Email
                subject = f"Position Closed: {ticker} | P&L: {pnl_emoji}${trade.pnl:.2f}"
                body = (
                    f"Position auto-closed: {ticker}\n"
                    f"Reason: {reason}\n"
                    f"Exit Price: ${trade.price:.2f}\n"
                    f"P&L: {pnl_emoji}${trade.pnl:.2f}\n"
                    f"Quantity: {trade.quantity:.2f}\n"
                )
                nm.send_email(subject, body)

                # Discord
                embed = {
                    "title": f"Position Closed: {ticker}",
                    "description": reason,
                    "color": 3066993 if trade.pnl >= 0 else 15158332,
                    "fields": [
                        {"name": "Exit Price", "value": f"${trade.price:.2f}", "inline": True},
                        {"name": "P&L", "value": f"{pnl_emoji}${trade.pnl:.2f}", "inline": True},
                        {"name": "Quantity", "value": f"{trade.quantity:.2f}", "inline": True},
                    ],
                    "timestamp": datetime.utcnow().isoformat(),
                    "footer": {"text": "QuantTrade Auto-Close"},
                }
                nm.send_discord("", embed)

        except Exception as e:
            logger.error(f"Close notification failed: {e}")

        logger.info(f"Stop-loss/take-profit: {len(closed_positions)} positions auto-closed")
        portfolio._save_state()
    else:
        logger.info("Stop-loss/take-profit: all positions within limits")
        portfolio._save_state()


def _run_daily_pipeline_impl():
    global last_run_time, last_run_status, pipeline_runs

    from scripts.run_daily import run_daily_pipeline as _run_pipeline

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Starting pipeline for {today}")
    last_run_time = datetime.now().isoformat()

    # Run satellite monitors inside the child process. The parent process owns
    # timeout/termination so running native geospatial code cannot leak memory.
    logger.info("Running satellite monitoring modules...")
    try:
        satellite_signals = run_satellite_monitors(today, "outputs")
        logger.info(f"Satellite monitoring complete: {len(satellite_signals)} signals")
    except Exception as e:
        logger.error(f"Satellite monitoring failed: {e}")

    # Run main pipeline
    try:
        result = _run_pipeline(target_date=today, output_base="outputs")
        logger.info(f"Pipeline complete: {result.get('regions_processed')} regions")
        last_run_status = "success"
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        last_run_status = f"failed: {str(e)[:50]}"
        return

    # Check signals and send notifications
    logger.info("Checking signals for actionable notifications...")
    try:
        from pipeline.notifications import send_notification_on_signals
        notification_result = send_notification_on_signals(today)
        logger.info(f"Signal check complete: {notification_result.get('actionable_count', 0)} actionable signals")
        if notification_result.get('notified'):
            logger.info(f"Notification sent: {notification_result.get('message', '')}")
        else:
            logger.info(f"No notification needed: {notification_result.get('message', '')}")
    except Exception as e:
        logger.error(f"Signal notification check failed: {e}")

    # Auto-trade: open positions based on high-confidence signals
    logger.info("Auto-trading: checking signals for position openings...")
    try:
        from paper_trading.multi_asset_portfolio import MultiAssetPortfolio
        from pipeline.price_feed import get_prices_for_portfolio
        from pathlib import Path as _Path
        import json as _json

        portfolio = MultiAssetPortfolio(output_base="outputs")

        # Load latest signals from all modules
        signals_dir = _Path("outputs")
        actionable_signals = []

        for signal_file in sorted(signals_dir.rglob("signal_*.json")):
            try:
                sig = _json.loads(signal_file.read_text())
                # Respect trade_direction override (e.g. SST signals are context-only)
                direction = sig.get("trade_direction", sig.get("direction", "neutral"))
                confidence = sig.get("confidence", 0)
                if direction != "neutral" and confidence >= 75:
                    actionable_signals.append(sig)
            except Exception:
                continue

        # Apply learned weights — adjust confidence and threshold per region/signal_type
        try:
            from pipeline.signal_feedback_learner import should_trade
            learned_signals = []
            for sig in actionable_signals:
                rid = sig.get("region_id", "")
                stype = sig.get("signal_type", "")
                raw_conf = sig.get("confidence", 0)
                ok, eff_conf, reason = should_trade(raw_conf, rid, stype, output_base="outputs")
                sig["effective_confidence"] = eff_conf
                sig["learning_reason"] = reason
                if ok:
                    learned_signals.append(sig)
                else:
                    logger.info(f"  Learning blocked: {rid} raw={raw_conf}% eff={eff_conf:.1f}% — {reason}")
            logger.info(f"Auto-trade: {len(actionable_signals)} raw → {len(learned_signals)} after learning filter")
            actionable_signals = learned_signals
        except Exception as learn_err:
            logger.warning(f"Feedback learner not available, using static threshold: {learn_err}")

        # Deduplicate: keep highest confidence per region+type
        best_signals = {}
        for sig in actionable_signals:
            key = f"{sig.get('region_id','')}_{sig.get('signal_type','')}"
            if key not in best_signals or sig.get('confidence', 0) > best_signals[key].get('confidence', 0):
                best_signals[key] = sig

        # Track detected signals
        try:
            from tracking.signal_tracker import SignalTracker
            signal_tracker = SignalTracker(output_base="outputs")
            for sig in best_signals.values():
                signal_tracker.record_signal_detected(sig)
        except Exception as track_err:
            logger.warning(f"Signal tracking (detected) failed: {track_err}")
            signal_tracker = None

        # Collect all signal tickers
        all_signal_tickers = set()
        for sig in best_signals.values():
            for inst in sig.get("instruments", []):
                if isinstance(inst, str):
                    all_signal_tickers.add(inst)

        # Fetch prices for signal tickers - REAL prices only
        from pipeline.price_feed import fetch_all_prices
        signal_prices = fetch_all_prices(list(all_signal_tickers))
        
        # Debug: log which tickers got prices and which didn't
        got_price = {t: p for t, p in signal_prices.items() if p and p > 0}
        no_price = [t for t in all_signal_tickers if t not in got_price]
        if no_price:
            logger.info(f"  No real price for {len(no_price)} tickers: {no_price[:15]}")
        
        prices = get_prices_for_portfolio(portfolio)
        prices.update({k: v for k, v in got_price.items()})
        trades_made = 0

        # Initialize execution service (replaces direct portfolio.open_position calls)
        # FAIL CLOSED: if ExecutionService cannot initialize, auto-trading is skipped.
        # Phase 0 mandates no production bypass; degradation to direct open_position
        # is the exact unsafe path Phase 0 removes.
        try:
            from execution.service import ExecutionService
            execution_svc = ExecutionService()
        except Exception as svc_err:
            logger.error(
                f"ExecutionService init failed — auto-trading DISABLED: {svc_err}"
            )
            execution_svc = None

        # Initialize risk manager
        try:
            from risk.risk_manager import RiskManager
            risk_mgr = RiskManager()
        except Exception as risk_err:
            logger.warning(f"Risk manager init failed: {risk_err}")
            risk_mgr = None

        # ─── Bayesian Fusion + Kelly Sizing (v2/v3) ───
        try:
            from pipeline.bayesian_fusion import get_fusion_engine
            fusion = get_fusion_engine(output_base="outputs")
            all_sig_list = list(best_signals.values())
            held_tickers = list(portfolio.positions.keys())
            decisions = fusion.get_trading_decisions(all_sig_list, held_tickers, portfolio.cash)
            
            logger.info(f"Bayesian fusion: {len(decisions)} tradeable instruments from {len(all_sig_list)} signals")
            for d in decisions[:5]:
                logger.info(f"  {d['ticker']:6} {d['direction']:5} fused={d['fused_confidence']:.0f}% kelly={d['kelly_fraction']*100:.1f}% sources={len(d['sources'])} conflict={d['conflict']}")
        except Exception as fusion_err:
            logger.warning(f"Bayesian fusion failed, falling back to simple mode: {fusion_err}")
            decisions = []
        
        # Fall back to simple sorted signals if fusion unavailable
        use_fusion = len(decisions) > 0
        if not use_fusion:
            sorted_signals = sorted(best_signals.items(), key=lambda x: -x[1].get("confidence", 0))
            logger.info(f"Auto-trade (fallback): {len(sorted_signals)} candidates, {len(prices)} prices, cash=${portfolio.cash:.0f}")

        max_trades = 3
        
        if use_fusion:
            if not execution_svc:
                logger.warning("Skipping Bayesian auto-trade: ExecutionService unavailable")
            else:
                for decision in decisions:
                    if trades_made >= max_trades:
                        break

                    ticker = decision["ticker"]
                    direction = decision["direction"]
                    position_value = decision["position_value"]

                    if ticker not in prices or not prices[ticker]:
                        logger.info(f"  Skip {ticker}: no price")
                        continue

                    price = float(prices[ticker])
                    if price <= 0 or position_value < 500:
                        continue

                    # Risk management check
                    if risk_mgr:
                        approved, reason = risk_mgr.check_risk(
                            ticker=ticker, direction=direction,
                            position_value=position_value, portfolio=portfolio,
                        )
                        if not approved:
                            logger.info(f"Risk check blocked {ticker}: {reason}")
                            continue

                    # Build rationale from sources
                    source_desc = ", ".join(f"{s['type']}({s['confidence']:.0f}%)" for s in decision['sources'])
                    conflict_str = " [CONFLICT]" if decision['conflict'] else ""

                    try:
                        from execution.models import (
                            OrderIntent, OrderSide, OrderType,
                            TimeInForce, PositionIntent,
                        )
                        from datetime import datetime as _dt, timezone as _tz

                        side = OrderSide.BUY if direction == "long" else OrderSide.SELL
                        broker_positions = {p.symbol: p for p in execution_svc.broker.get_positions()}
                        has_long = (
                            ticker in broker_positions
                            and broker_positions[ticker].side == "long"
                        )
                        if side == OrderSide.SELL and has_long:
                            pos_intent = PositionIntent.CLOSE_POSITION
                        elif side == OrderSide.SELL:
                            pos_intent = PositionIntent.OPEN_POSITION
                        else:
                            pos_intent = PositionIntent.OPEN_POSITION

                        intent = OrderIntent(
                            symbol=ticker,
                            side=side,
                            notional=position_value,
                            order_type=OrderType.MARKET,
                            time_in_force=TimeInForce.DAY,
                            client_order_id=execution_svc.make_client_order_id(
                                "bayes", ticker, direction,
                                _dt.now(_tz.utc).strftime("%Y%m%d"),
                            ),
                            created_at=_dt.now(_tz.utc),
                            position_intent=pos_intent,
                            rationale=f"Bayesian: {source_desc}{conflict_str}",
                            metadata={
                                "price": price,
                                "asset_class": decision['sources'][0].get('type', 'commodity') if decision['sources'] else 'commodity',
                            },
                        )
                        result = execution_svc.submit(intent)

                        if result.status.value in ("accepted", "filled"):
                            portfolio.open_position(
                                ticker=ticker, direction=direction,
                                price=price, value=position_value,
                                rationale=f"Bayesian: {source_desc}{conflict_str}",
                                asset_class=decision['sources'][0].get('type', 'commodity') if decision['sources'] else 'commodity',
                            )
                            if ticker in portfolio.positions:
                                portfolio.positions[ticker].region_id = decision['sources'][0].get('region', '') if decision['sources'] else ''
                            trades_made += 1
                            logger.info(f"AUTO-TRADE [BAYES] {direction.upper()}: {ticker} @ ${price:.2f} val=${position_value:.0f} fused={decision['fused_confidence']:.0f}% kelly={decision['kelly_fraction']*100:.1f}%{conflict_str} exec={result.status.value}")
                        else:
                            logger.info(f"AUTO-TRADE [BAYES] REJECTED {ticker}: {result.rejection_reason}")

                        if trades_made > 0 and signal_tracker:
                            try:
                                sig_for_track = {"region_id": decision['sources'][0].get('region',''), "direction": direction, "confidence": decision['fused_confidence'], "instruments": [ticker]}
                                signal_tracker.record_position_opened(sig_for_track, ticker, price)
                            except Exception:
                                pass
                    except Exception as e:
                        logger.error(f"Failed to open {ticker}: {e}")
        else:
            # Fallback: simple sorted signals
            if not execution_svc:
                logger.warning("Skipping fallback auto-trade: ExecutionService unavailable")
            else:
                for key, sig in sorted_signals:
                    if trades_made >= max_trades:
                        break
                    direction = sig.get("trade_direction", sig.get("direction", "neutral"))
                    instruments = sig.get("instruments", [])
                    if not instruments:
                        continue
                    ticker = None
                    for inst in instruments:
                        if isinstance(inst, str) and inst in prices and prices[inst] and inst not in portfolio.positions:
                            ticker = inst
                            break
                    if not ticker:
                        continue
                    price = float(prices[ticker])
                    if price <= 0:
                        continue
                    position_value = portfolio.cash * 0.05
                    if position_value < 500:
                        continue
                    if risk_mgr:
                        approved, reason = risk_mgr.check_risk(ticker=ticker, direction=direction, position_value=position_value, portfolio=portfolio)
                        if not approved:
                            continue
                    try:
                        from execution.models import (
                            OrderIntent as _OI, OrderSide as _OS, OrderType as _OT,
                            TimeInForce as _TIF, PositionIntent as _PI,
                        )
                        from datetime import datetime as _dt2, timezone as _tz2

                        side = _OS.BUY if direction == "long" else _OS.SELL
                        broker_positions = {p.symbol: p for p in execution_svc.broker.get_positions()}
                        has_long = (
                            ticker in broker_positions
                            and broker_positions[ticker].side == "long"
                        )
                        if side == _OS.SELL and has_long:
                            pos_intent = _PI.CLOSE_POSITION
                        else:
                            pos_intent = _PI.OPEN_POSITION

                        intent = _OI(
                            symbol=ticker,
                            side=side,
                            notional=position_value,
                            order_type=_OT.MARKET,
                            time_in_force=_TIF.DAY,
                            client_order_id=execution_svc.make_client_order_id(
                                "fb", ticker, direction,
                                _dt2.now(_tz2.utc).strftime("%Y%m%d"),
                            ),
                            created_at=_dt2.now(_tz2.utc),
                            position_intent=pos_intent,
                            rationale=f"Auto-trade: {sig.get('rationale','')[:150]}",
                            metadata={"price": price},
                        )
                        result = execution_svc.submit(intent)

                        if result.status.value in ("accepted", "filled"):
                            portfolio.open_position(ticker=ticker, direction=direction, price=price, value=position_value, rationale=f"Auto-trade: {sig.get('rationale','')[:150]}", asset_class=sig.get("signal_type","commodity"))
                            if ticker in portfolio.positions:
                                portfolio.positions[ticker].region_id = sig.get("region_id", "")
                            trades_made += 1
                            logger.info(f"AUTO-TRADE [FALLBACK] {direction.upper()}: {ticker} @ ${price:.2f} val=${position_value:.0f} exec={result.status.value}")
                        else:
                            logger.info(f"AUTO-TRADE [FALLBACK] REJECTED {ticker}: {result.rejection_reason}")
                    except Exception as e:
                        logger.error(f"Failed to open {ticker}: {e}")

        # Send notifications for any trades made (both Bayesian and fallback)
        if trades_made > 0:
            try:
                from notifications.notification_manager import NotificationManager
                nm = NotificationManager()
                # Notify about new trades in bulk
                logger.info(f"Trades completed: {trades_made} new positions")
            except Exception as notif_err:
                logger.warning(f"Trade notification failed: {notif_err}")

        if trades_made > 0:
            logger.info(f"Auto-trading complete: {trades_made} positions opened")
        else:
            logger.info("Auto-trading: no new positions opened")

        # Stop-loss / take-profit check
        check_stop_loss_take_profit(portfolio, prices, signal_tracker)

        # Daily P&L report (once per day)
        try:
            from notifications.daily_report import DailyReportGenerator
            daily_report = DailyReportGenerator(output_base="outputs")
            if daily_report.should_send_today():
                sent = daily_report.generate_and_send(portfolio, prices)
                if sent:
                    logger.info("Daily P&L report sent")
                else:
                    logger.info("Daily P&L report: send attempted (check notification config)")
        except Exception as report_err:
            logger.error(f"Daily report failed: {report_err}")

    except Exception as e:
        logger.error(f"Auto-trading failed: {e}")

    # Rebuild asset history
    try:
        from scripts.rebuild_asset_history import rebuild_asset_history
        rebuild_result = rebuild_asset_history(output_base="outputs", initial_capital=100000.0)
        logger.info(f"Asset history rebuilt: {len(rebuild_result.get('daily_assets', []))} points")
    except FileNotFoundError:
        logger.info("Asset history rebuild skipped: portfolio not yet initialized")
    except Exception as e:
        logger.error(f"Asset history rebuild failed: {e}")

    pipeline_runs += 1


def _run_daily_pipeline_child(status_queue):
    try:
        _run_daily_pipeline_impl()
        status_queue.put(("success", None))
    except Exception as exc:
        logger.exception("Pipeline child crashed")
        status_queue.put(("failed", str(exc)[:200]))


def run_daily_pipeline():
    """Run one pipeline cycle in a child process so heavy native memory is reclaimed."""
    global last_run_time, last_run_status, pipeline_runs

    if not pipeline_lock.acquire(blocking=False):
        logger.warning("Pipeline already running, skipping overlapping cycle")
        with state_lock:
            last_run_status = "skipped: already running"
        return

    try:
        import multiprocessing as mp

        timeout_seconds = int(os.environ.get("PIPELINE_TIMEOUT_SECONDS", "900"))
        with state_lock:
            last_run_time = datetime.now().isoformat()
            last_run_status = "running"

        queue = mp.Queue()
        proc = mp.Process(target=_run_daily_pipeline_child, args=(queue,), daemon=False)
        proc.start()
        logger.info("Pipeline child started: pid=%s timeout=%ss", proc.pid, timeout_seconds)

        # Wait for child, checking shutdown_event every 5s
        elapsed = 0
        while proc.is_alive() and elapsed < timeout_seconds:
            if shutdown_event.is_set():
                logger.info("Shutdown requested, terminating pipeline child pid=%s", proc.pid)
                proc.terminate()
                proc.join(10)
                if proc.is_alive():
                    proc.kill()
                    proc.join(5)
                with state_lock:
                    last_run_status = "stopped: shutdown"
                return
            wait = min(5, timeout_seconds - elapsed)
            proc.join(wait)
            elapsed += wait

        if proc.is_alive():
            logger.error("Pipeline child timed out after %ss, terminating pid=%s", timeout_seconds, proc.pid)
            proc.terminate()
            proc.join(30)
            if proc.is_alive():
                logger.error("Pipeline child did not terminate cleanly, killing pid=%s", proc.pid)
                proc.kill()
                proc.join(10)
            with state_lock:
                last_run_status = "failed: timeout"
            return

        if proc.exitcode != 0:
            with state_lock:
                last_run_status = f"failed: child exit {proc.exitcode}"
            logger.error("Pipeline child exited with code %s", proc.exitcode)
            return

        try:
            status, message = queue.get(timeout=5)
        except Exception:
            status, message = "success", None
        if status == "success":
            with state_lock:
                pipeline_runs += 1
                last_run_status = "success"
            logger.info("Pipeline child completed successfully")
        else:
            with state_lock:
                last_run_status = f"failed: {message or 'unknown'}"
            logger.error("Pipeline child failed: %s", message)
    finally:
        pipeline_lock.release()


def ensure_backfill_data(output_base: str = "outputs") -> bool:
    """
    Ensure backfill data exists for all active regions.
    Creates minimal backfill files without requiring numpy/rasterio imports.

    Returns True if backfill was run, False if data already existed.
    """
    from datetime import datetime, timedelta
    import random
    from pathlib import Path
    from pipeline.regions import get_active_regions

    logger.info(f"ensure_backfill_data: Starting with output_base={output_base}")

    backfill_dir = Path(output_base) / "backfill"
    backfill_dir.mkdir(parents=True, exist_ok=True)

    regions = get_active_regions()
    logger.info(f"ensure_backfill_data: Found {len(regions)} active regions")

    missing_regions = []

    for region_id in regions.keys():
        backfill_file = backfill_dir / f"{region_id}_backfill.json"
        if not backfill_file.exists():
            missing_regions.append(region_id)

    logger.info(f"ensure_backfill_data: {len(missing_regions)} missing backfill files")

    if not missing_regions:
        logger.info("All backfill files exist, skipping initialization")
        return False

    logger.info(f"Creating backfill files for {len(missing_regions)} regions")

    # Create minimal backfill files with synthetic historical data
    # This allows the signal generation to work without requiring numpy/rasterio
    today = datetime.now().date()
    created_count = 0

    for region_id in missing_regions:
        try:
            backfill_file = backfill_dir / f"{region_id}_backfill.json"
            region_config = regions.get(region_id, {})
            region_type = region_config.get("type", "chokepoint")

            # Generate 90 days of synthetic historical data
            daily_stats = []
            for i in range(90):
                date = today - timedelta(days=i)
                if region_type in ["oil_storage", "port_logistics"]:
                    # Count-based data
                    daily_stats.append({
                        "date": date.isoformat(),
                        "count": random.randint(50, 150),
                        "valid_pixels": random.randint(800, 1000),
                    })
                else:
                    # Value-based data
                    daily_stats.append({
                        "date": date.isoformat(),
                        "value": random.uniform(0.3, 0.8),
                        "valid_pixels": random.randint(800, 1000),
                    })

            backfill_data = {
                "region_id": region_id,
                "type": region_type,
                "generated_at": datetime.now().isoformat(),
                "source": "synthetic_initialization",
                "daily_stats": daily_stats,
            }

            backfill_file.write_text(json.dumps(backfill_data, indent=2))
            created_count += 1

        except Exception as e:
            logger.error(f"Failed to create backfill for {region_id}: {e}")

    logger.info(f"ensure_backfill_data: Created {created_count}/{len(missing_regions)} backfill files")
    return created_count > 0


def run_paper_trading_cycle():
    """Full paper trading cycle: risk check, auto-trade, reconcile, Discord report.
    Only runs during US market hours (9:30am-4pm ET, Mon-Fri).
    """
    # Market hours check: 9:30 AM - 4:00 PM ET, Monday-Friday
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        logger.debug("Skipping paper trading cycle: weekend")
        return
    if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
        logger.debug("Skipping paper trading cycle: outside market hours (ET)")
        return

    output_dir = Path("outputs")

    # Load portfolio and prices once (reused across cycle)
    from paper_trading.multi_asset_portfolio import MultiAssetPortfolio
    from pipeline.price_feed import get_prices_for_portfolio

    portfolio = MultiAssetPortfolio()
    prices = get_prices_for_portfolio(portfolio)

    # 1. Check stop-loss/take-profit
    try:
        check_stop_loss_take_profit(portfolio, prices)
    except Exception as e:
        logger.error("Stop-loss check failed: %s", e)

    # 2. Run heavy pipeline only if data is stale or missing
    today_summary = output_dir / now.strftime("%Y-%m-%d") / "daily_summary.json"
    should_run_pipeline = True
    if today_summary.exists():
        age = now.timestamp() - today_summary.stat().st_mtime
        should_run_pipeline = age > 7200  # 2 hours

    if should_run_pipeline:
        logger.info("Running satellite pipeline (data stale or missing)")
        run_daily_pipeline()

    # 3. Reconcile orders + sync portfolio from Alpaca
    try:
        from execution.service import ExecutionService
        from execution.reconciler import Reconciler

        svc = ExecutionService()
        reconciler = Reconciler(svc)
        report = reconciler.run()
        if report.has_alert:
            logger.warning("Reconciler alerts: %s", report.summary())

        synced, sync_warnings = portfolio.sync_from_alpaca()
        if synced or sync_warnings:
            logger.info("Portfolio sync: %d closed, %d warnings", synced, len(sync_warnings))
            for w in sync_warnings:
                logger.warning("Portfolio sync: %s", w)
    except Exception as e:
        logger.error("Reconcile/sync failed: %s", e)

    # 4. Send Discord report
    try:
        from paper_trading.daily_multi_report import format_discord_report

        recent_trades = portfolio.trades[-5:] if len(portfolio.trades) > 5 else portfolio.trades
        top_signals = load_top_signals(output_dir, limit=5)

        report = format_discord_report(portfolio, prices, top_signals, recent_trades)
        send_discord_report(report)
    except Exception as e:
        logger.error("Discord report failed: %s", e)


def main():
    global last_run_status

    import signal as _signal

    def _handle_shutdown(signum, frame):
        sig_name = _signal.Signals(signum).name
        logger.info("Received %s, shutting down gracefully...", sig_name)
        shutdown_event.set()

    _signal.signal(_signal.SIGTERM, _handle_shutdown)
    _signal.signal(_signal.SIGINT, _handle_shutdown)

    port = int(os.environ.get("PORT", "8080"))

    logger.info("=" * 60)
    logger.info("QuantTrade Scheduler Started")
    logger.info(f"Health check port: {port}")
    logger.info(f"Paper trading cycle: every 15 minutes (market hours)")
    logger.info("=" * 60)

    # Start health check server for Railway FIRST
    # This ensures the health endpoint responds immediately
    server = start_health_server(port)

    # Mark as ready for health checks
    with state_lock:
        last_run_status = "initialized"

    def delayed_start():
        time.sleep(5)  # Give health server time to start
        run_daily_pipeline()

    initial_thread = threading.Thread(target=delayed_start, daemon=True)
    initial_thread.start()

    # Single consolidated paper trading cycle every 15 minutes
    schedule.every(15).minutes.do(run_paper_trading_cycle)
    logger.info("Paper trading cycle: every 15 minutes")

    while not shutdown_event.is_set():
        schedule.run_pending()
        shutdown_event.wait(30)

    logger.info("Shutting down health server...")
    server.shutdown()
    logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
