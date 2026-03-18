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


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for Railway health checks."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        global last_run_time, last_run_status, pipeline_runs

        if self.path == "/health" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            response = {
                "status": "healthy",
                "service": "quanttrade-scheduler",
                "last_run": last_run_time,
                "last_status": last_run_status,
                "total_runs": pipeline_runs,
                "interval_minutes": int(os.environ.get("PIPELINE_INTERVAL_MINUTES", "60")),
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()


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


def main():
    refresh_interval = int(os.environ.get("PIPELINE_INTERVAL_MINUTES", "60"))
    port = int(os.environ.get("PORT", "8080"))

    logger.info("=" * 60)
    logger.info("QuantTrade Scheduler Started")
    logger.info(f"Refresh interval: every {refresh_interval} minutes")
    logger.info(f"Health check port: {port}")
    logger.info(f"Real satellite data: auto-detected")
    logger.info("=" * 60)

    # Start health check server for Railway
    start_health_server(port)

    # Run once at startup
    run_daily_pipeline()

    # Schedule recurring runs
    schedule.every(refresh_interval).minutes.do(run_daily_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
