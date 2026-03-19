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

        elif self.path == "/api/dates":
            # List available dates
            dates = self._list_available_dates()
            self._send_json({"dates": dates})

        elif self.path == "/debug/cwd":
            import os as _os
            self._send_json({"cwd": _os.getcwd(), "output_base": OUTPUT_BASE})

        elif self.path == "/debug/backfill":
            # List available backfill files
            backfill_dir = Path(OUTPUT_BASE) / "backfill"
            files = list(backfill_dir.glob("*.json")) if backfill_dir.exists() else []
            self._send_json({"backfill_files": [f.name for f in files]})

        else:
            self._send_json({"error": "Not found"}, 404)

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
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health check server started on port {port}")
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

    # Count actionable
    actionable = sum(1 for s in all_signals if s.get("direction") != "neutral")
    logger.info(f"Satellite monitors total: {len(all_signals)} signals, {actionable} actionable")

    return all_signals


def run_daily_pipeline():
    global last_run_time, last_run_status, pipeline_runs

    from scripts.run_daily import run_daily_pipeline as _run_pipeline

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Starting pipeline for {today}")
    last_run_time = datetime.now().isoformat()

    # Run satellite monitors (with auto-detected real data)
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

    # Rebuild asset history
    try:
        from scripts.rebuild_asset_history import rebuild_asset_history
        rebuild_result = rebuild_asset_history(output_base="outputs", initial_capital=100000.0)
        logger.info(f"Asset history rebuilt: {len(rebuild_result.get('daily_assets', []))} points")
    except Exception as e:
        logger.error(f"Asset history rebuild failed: {e}")

    pipeline_runs += 1


def ensure_backfill_data(output_base: str = "outputs") -> bool:
    """
    Ensure backfill data exists for all active regions.

    Returns True if backfill was run, False if data already existed.
    """
    from pathlib import Path
    from pipeline.regions import get_active_regions

    backfill_dir = Path(output_base) / "backfill"
    backfill_dir.mkdir(parents=True, exist_ok=True)

    regions = get_active_regions()
    missing_regions = []

    for region_id in regions.keys():
        backfill_file = backfill_dir / f"{region_id}_backfill.json"
        if not backfill_file.exists():
            missing_regions.append(region_id)

    if not missing_regions:
        logger.info("All backfill files exist, skipping initialization")
        return False

    logger.info(f"Running backfill for {len(missing_regions)} regions...")

    try:
        from scripts.run_backfill import run_auto_backfill

        result = run_auto_backfill(
            regions_filter=missing_regions,
            output_base=output_base,
        )

        logger.info(f"Backfill initialization complete: {result}")
        return True

    except Exception as e:
        logger.error(f"Backfill initialization failed: {e}")
        return False


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
    def delayed_start():
        time.sleep(5)  # Give health server time to start
        # Ensure backfill data exists before first run
        ensure_backfill_data("outputs")
        run_daily_pipeline()

    initial_thread = threading.Thread(target=delayed_start, daemon=True)
    initial_thread.start()

    # Schedule recurring runs
    schedule.every(refresh_interval).minutes.do(run_daily_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
