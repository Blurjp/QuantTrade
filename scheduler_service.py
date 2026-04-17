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
            response = {
                "status": "healthy",
                "service": "quanttrade-scheduler",
                "last_run": last_run_time,
                "last_status": last_run_status,
                "total_runs": pipeline_runs,
                "interval_minutes": int(os.environ.get("PIPELINE_INTERVAL_MINUTES", "60")),
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
            # Serve current portfolio state
            portfolio_file = Path(OUTPUT_BASE) / "paper_trading" / "multi_asset_portfolio.json"
            if portfolio_file.exists():
                self._send_json(json.loads(portfolio_file.read_text()))
            else:
                self._send_json({"error": "No portfolio file"}, 404)

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
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health check server started on port {port} (ThreadingHTTPServer)")
    return server


def run_satellite_monitors(target_date: str, output_base: str = "outputs"):
    """
    Run all satellite monitoring modules.

    Automatically uses real satellite data when available (auto-detected).
    """
    from pipeline.satellite_data import get_capabilities

    caps = get_capabilities()
    logger.info(f"Satellite data capabilities: Real={caps['real_data_enabled']}, PC={caps['planetary_computer']['available']}")

    all_signals = []

    # Vegetation Health
    try:
        from pipeline.vegetation_health import VegetationHealthMonitor
        monitor = VegetationHealthMonitor(output_base=output_base)
        signals = monitor.generate_all_signals(target_date)
        all_signals.extend(signals)
        logger.info(f"Vegetation health: {len(signals)} signals")
    except Exception as e:
        logger.warning(f"Vegetation health failed: {e}")

    # Sea Surface Temperature
    try:
        from pipeline.sea_surface_temperature import SeaSurfaceTemperatureMonitor
        monitor = SeaSurfaceTemperatureMonitor(output_base=output_base)
        signals = monitor.generate_all_signals(target_date)
        all_signals.extend(signals)
        logger.info(f"SST: {len(signals)} signals")
    except Exception as e:
        logger.warning(f"SST failed: {e}")

    # Solar Irradiance
    try:
        from pipeline.solar_irradiance import SolarIrradianceMonitor
        monitor = SolarIrradianceMonitor(output_base=output_base)
        signals = monitor.generate_all_signals(target_date)
        all_signals.extend(signals)
        logger.info(f"Solar irradiance: {len(signals)} signals")
    except Exception as e:
        logger.warning(f"Solar irradiance failed: {e}")

    # Atmospheric
    try:
        from pipeline.atmospheric import AtmosphericMonitor
        monitor = AtmosphericMonitor(output_base=output_base)
        signals = monitor.generate_all_signals(target_date)
        all_signals.extend(signals)
        logger.info(f"Atmospheric: {len(signals)} signals")
    except Exception as e:
        logger.warning(f"Atmospheric failed: {e}")

    # Thermal Infrared
    try:
        from pipeline.thermal_infrared import ThermalInfraredMonitor
        monitor = ThermalInfraredMonitor(output_base=output_base)
        signals = monitor.generate_all_signals(target_date)
        all_signals.extend(signals)
        logger.info(f"Thermal infrared: {len(signals)} signals")
    except Exception as e:
        logger.warning(f"Thermal infrared failed: {e}")

    # Nighttime Lights
    try:
        from pipeline.nighttime_lights import NighttimeLightsMonitor
        monitor = NighttimeLightsMonitor(output_base=output_base)
        signals = monitor.generate_all_signals(target_date)
        all_signals.extend(signals)
        logger.info(f"Nighttime lights: {len(signals)} signals")
    except Exception as e:
        logger.warning(f"Nighttime lights failed: {e}")

    # Precipitation
    try:
        from pipeline.precipitation import PrecipitationMonitor
        monitor = PrecipitationMonitor(output_base=output_base)
        signals = monitor.generate_all_signals(target_date)
        all_signals.extend(signals)
        logger.info(f"Precipitation: {len(signals)} signals")
    except Exception as e:
        logger.warning(f"Precipitation failed: {e}")

    # Cattle Feedlot
    try:
        from pipeline.cattle_feedlot import CattleFeedlotMonitor
        cattle_monitor = CattleFeedlotMonitor(output_base=output_base)
        cattle_signals = cattle_monitor.generate_signal()
        all_signals.extend(cattle_signals)
        logger.info(f"Cattle feedlot: {len(cattle_signals)} signals")
    except Exception as e:
        logger.warning(f"Cattle feedlot failed: {e}")

    # Count actionable
    actionable = sum(1 for s in all_signals if s.get("direction") != "neutral")
    logger.info(f"Satellite monitors total: {len(all_signals)} signals, {actionable} actionable")

    return all_signals


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

                # Track in signal history
                if signal_tracker:
                    try:
                        signal_tracker.record_position_closed(ticker, current_price, trade.pnl)
                    except Exception:
                        pass

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


def run_daily_pipeline():
    global last_run_time, last_run_status, pipeline_runs

    from scripts.run_daily import run_daily_pipeline as _run_pipeline

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Starting pipeline for {today}")
    last_run_time = datetime.now().isoformat()

    # Run satellite monitors with TIMEOUT to prevent infinite I/O blocking
    PIPELINE_TIMEOUT = int(os.environ.get("PIPELINE_TIMEOUT_SECONDS", "300"))
    logger.info(f"Running satellite monitoring modules (timeout={PIPELINE_TIMEOUT}s)...")
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_satellite_monitors, today, "outputs")
            try:
                satellite_signals = future.result(timeout=PIPELINE_TIMEOUT)
                logger.info(f"Satellite monitoring complete: {len(satellite_signals)} signals")
            except concurrent.futures.TimeoutError:
                logger.warning(f"Satellite monitoring TIMED OUT after {PIPELINE_TIMEOUT}s, continuing...")
                future.cancel()
    except Exception as e:
        logger.error(f"Satellite monitoring failed: {e}")

    # Check signals and send notifications
    logger.info("Checking signals for notifications...")
    try:
        from notifications.signal_monitor import SignalMonitor
        monitor = SignalMonitor()
        summary = monitor.check_all_modules()
        logger.info(f"Signal check complete: {summary['actionable_signals']} actionable, {summary['notified']} notifications sent")
    except Exception as e:
        logger.error(f"Signal monitoring failed: {e}")

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
                direction = sig.get("direction", "neutral")
                confidence = sig.get("confidence", 0)
                if direction != "neutral" and confidence >= 75:
                    actionable_signals.append(sig)
            except Exception:
                continue

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

        # Fetch prices for signal tickers
        from pipeline.price_feed import fetch_all_prices
        signal_prices = fetch_all_prices(list(all_signal_tickers))
        prices = get_prices_for_portfolio(portfolio)
        prices.update({k: v for k, v in signal_prices.items() if v})
        trades_made = 0

        # Initialize risk manager
        try:
            from risk.risk_manager import RiskManager
            risk_mgr = RiskManager()
        except Exception as risk_err:
            logger.warning(f"Risk manager init failed: {risk_err}")
            risk_mgr = None

        # Sort signals by confidence (highest first)
        sorted_signals = sorted(best_signals.items(), key=lambda x: -x[1].get("confidence", 0))

        logger.info(f"Auto-trade: {len(sorted_signals)} candidates, {len(prices)} prices, cash=${portfolio.cash:.0f}")
        for key, sig in sorted_signals[:5]:
            insts = sig.get('instruments', [])
            price_check = {i: prices.get(i) for i in insts if isinstance(i, str)}
            logger.info(f"  Top signal: {key} dir={sig.get('direction')} conf={sig.get('confidence')} inst_prices={price_check}")

        for key, sig in sorted_signals:
            if trades_made >= 3:
                break  # Max 3 new positions per run

            direction = sig.get("direction", "neutral")
            instruments = sig.get("instruments", [])
            if not instruments:
                continue

            # Pick first available instrument with price AND not already held
            ticker = None
            for inst in instruments:
                if isinstance(inst, str) and inst in prices and prices[inst] and inst not in portfolio.positions:
                    ticker = inst
                    break

            if not ticker:
                logger.info(f"  Skip {key}: no available instrument (held={set(instruments) & set(portfolio.positions.keys()) or 'none'}, no_price={[i for i in instruments if i not in prices or not prices[i]]})")
                continue

            price = float(prices[ticker])
            if price <= 0:
                continue

            # Calculate position size (5% of portfolio per trade)
            position_value = portfolio.cash * 0.05
            if position_value < 500:
                continue

            # Risk management check
            if risk_mgr:
                approved, reason = risk_mgr.check_risk(
                    ticker=ticker,
                    direction=direction,
                    position_value=position_value,
                    portfolio=portfolio,
                )
                if not approved:
                    logger.info(f"Risk check blocked {ticker}: {reason}")
                    continue

            try:
                portfolio.open_position(
                    ticker=ticker,
                    direction=direction,
                    price=price,
                    value=position_value,
                    rationale=f"Auto-trade: {sig.get('rationale', '')[:150]}",
                    asset_class=sig.get("signal_type", "commodity"),
                )
                trades_made += 1
                logger.info(f"AUTO-TRADE OPEN {direction.upper()}: {ticker} @ ${price:.2f} value=${position_value:.0f} (conf={sig.get('confidence')}%)")

                # Track position opened
                if signal_tracker:
                    try:
                        signal_tracker.record_position_opened(sig, ticker, price)
                    except Exception:
                        pass

                # Send notifications for the new trade
                try:
                    from notifications.notification_manager import NotificationManager
                    nm = NotificationManager()
                    nm.notify_signal(
                        signal={
                            "signal_type": sig.get("signal_type", "commodity"),
                            "region": sig.get("region", "global"),
                            "region_name": sig.get("region", sig.get("region_name", "global")),
                            "region_id": sig.get("region_id", ""),
                            "direction": direction,
                            "confidence": sig.get("confidence", 0),
                            "rationale": f"AUTO-TRADE: {direction.upper()} {ticker} @ ${price:.2f} — {sig.get('rationale', '')[:100]}",
                            "instruments": [ticker],
                            "date": datetime.now().strftime("%Y-%m-%d"),
                        },
                    )
                    logger.info(f"Trade notification sent for {ticker}")
                except Exception as notif_err:
                    logger.warning(f"Trade notification failed for {ticker}: {notif_err}")

            except Exception as e:
                logger.warning(f"Auto-trade failed for {ticker}: {e}")

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


def main():
    global last_run_status

    refresh_interval = int(os.environ.get("PIPELINE_INTERVAL_MINUTES", "60"))
    port = int(os.environ.get("PORT", "8080"))

    logger.info("=" * 60)
    logger.info("QuantTrade Scheduler Started")
    logger.info(f"Refresh interval: every {refresh_interval} minutes")
    logger.info(f"Health check port: {port}")
    logger.info(f"Real satellite data: auto-detected")
    logger.info("=" * 60)

    # Start health check server for Railway FIRST
    # This ensures the health endpoint responds immediately
    start_health_server(port)

    # Mark as ready for health checks
    last_run_status = "initialized"

    # Run pipeline in background thread after short delay
    # This allows the health check to pass while pipeline runs
    PIPELINE_TIMEOUT = int(os.environ.get("PIPELINE_TIMEOUT_SECONDS", "300"))

    def delayed_start():
        time.sleep(5)  # Give health server time to start
        # Run pipeline with timeout using a separate thread
        def _run_with_timeout():
            run_daily_pipeline()

        t = threading.Thread(target=_run_with_timeout, daemon=True)
        t.start()
        t.join(timeout=PIPELINE_TIMEOUT)  # Wait max PIPELINE_TIMEOUT seconds
        if t.is_alive():
            logger.warning(f"First pipeline run timed out after {PIPELINE_TIMEOUT}s, will retry next cycle")

    initial_thread = threading.Thread(target=delayed_start, daemon=True)
    initial_thread.start()

    # Schedule recurring runs
    schedule.every(refresh_interval).minutes.do(run_daily_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
