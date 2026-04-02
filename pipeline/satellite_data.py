"""
Satellite Data Fetcher - Fully Automated Infrastructure

Automatically detects capabilities and enables real satellite data when available.
No manual configuration required - just install packages and optionally set credentials.

Auto-Detection:
- Checks for required packages (pystac_client, planetary_computer, odc-stac, etc.)
- Checks for NASA Earthdata credentials (optional, only for precipitation)
- Automatically enables real data when packages are installed

Data Sources:
- Planetary Computer (Sentinel, Landsat, MODIS, VIIRS) - FREE, no auth needed
- NASA GES DISC (GPM/IMERG precipitation) - requires NASA Earthdata account

Usage:
    # Just use it - real data is auto-enabled if packages are installed
    from pipeline.satellite_data import get_real_data
    data = get_real_data("ndvi", bbox=[-100, 36, -82, 48], date="2024-01-15")

    # Check capabilities
    from pipeline.satellite_data import get_capabilities
    print(get_capabilities())
"""

import hashlib
import json
import logging
import os
import time
import functools
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Capability Detection (Auto-detect what's available)
# =============================================================================

class CapabilityDetector:
    """Auto-detect available satellite data capabilities."""

    _instance = None
    _capabilities = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize and detect capabilities."""
        if CapabilityDetector._capabilities is None:
            CapabilityDetector._capabilities = self._detect_all()
        self._caps = CapabilityDetector._capabilities

    def _detect_all(self) -> Dict:
        """Detect all available capabilities."""
        caps = {
            "planetary_computer": self._check_planetary_computer(),
            "nasa_gesdisc": self._check_nasa_gesdisc(),
            "cache_dir": self._get_cache_dir(),
            "auto_enabled": True,  # Always try real data first
        }

        # Log detected capabilities
        logger.info(f"Satellite data capabilities detected:")
        logger.info(f"  Planetary Computer: {caps['planetary_computer']['available']}")
        logger.info(f"  NASA GES DISC: {caps['nasa_gesdisc']['available']}")
        if caps['nasa_gesdisc']['available']:
            logger.info(f"  NASA credentials: configured")

        return caps

    def _check_planetary_computer(self) -> Dict:
        """Check if Planetary Computer packages are available."""
        packages = {
            "pystac_client": self._try_import("pystac_client"),
            "planetary_computer": self._try_import("planetary_computer"),
            "odc_stac": self._try_import("odc.stac"),
            "xarray": self._try_import("xarray"),
            "rasterio": self._try_import("rasterio"),
        }

        # Planetary Computer works with just pystac_client and odc_stac
        available = packages["pystac_client"] and packages["odc_stac"]

        return {
            "available": available,
            "packages": packages,
            "note": "Free, no authentication required" if available else "Install: pip install pystac-client odc-stac planetary-computer"
        }

    def _check_nasa_gesdisc(self) -> Dict:
        """Check if NASA GES DISC is available."""
        has_requests = self._try_import("requests")
        has_netcdf = self._try_import("netCDF4")

        # Check for credentials (support both naming conventions)
        username = os.environ.get("NASA_EARTHDATA_USERNAME") or os.environ.get("EARTHDATA_USERNAME")
        password = os.environ.get("NASA_EARTHDATA_PASSWORD") or os.environ.get("EARTHDATA_PASSWORD")
        has_credentials = bool(username and password)
        
        # Ensure earthaccess can find credentials
        if has_credentials:
            os.environ.setdefault("EARTHDATA_USERNAME", username)
            os.environ.setdefault("EARTHDATA_PASSWORD", password)

        available = has_requests and has_netcdf and has_credentials

        return {
            "available": available,
            "packages": {
                "requests": has_requests,
                "netCDF4": has_netcdf,
            },
            "has_credentials": has_credentials,
            "note": "Requires NASA Earthdata account (free): https://urs.earthdata.nasa.gov/" if not has_credentials else "Configured"
        }

    def _try_import(self, module: str) -> bool:
        """Try to import a module."""
        try:
            __import__(module)
            return True
        except ImportError:
            return False

    def _get_cache_dir(self) -> Path:
        """Get or create cache directory."""
        cache_dir = Path(os.environ.get(
            "SATELLITE_CACHE_DIR",
            "outputs/satellite_cache"
        ))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @property
    def can_use_planetary_computer(self) -> bool:
        """Check if Planetary Computer is available."""
        return self._caps["planetary_computer"]["available"]

    @property
    def can_use_nasa_gesdisc(self) -> bool:
        """Check if NASA GES DISC is available."""
        return self._caps["nasa_gesdisc"]["available"]

    @property
    def cache_dir(self) -> Path:
        """Get cache directory."""
        return self._caps["cache_dir"]

    def should_use_real_data(self, force_check: bool = False) -> bool:
        """
        Determine if real data should be used.

        Priority:
        1. USE_REAL_SATELLITE_DATA env var if set (true/false)
        2. Auto-detect based on available packages
        """
        env_val = os.environ.get("USE_REAL_SATELLITE_DATA", "").lower()

        if env_val == "true":
            return True
        elif env_val == "false":
            return False

        # Auto-detect: use real data if any source is available
        return self.can_use_planetary_computer or self.can_use_nasa_gesdisc

    def get_capabilities_report(self) -> Dict:
        """Get a detailed capabilities report."""
        return {
            "real_data_enabled": self.should_use_real_data(),
            "planetary_computer": self._caps["planetary_computer"],
            "nasa_gesdisc": self._caps["nasa_gesdisc"],
            "cache_dir": str(self.cache_dir),
            "recommendations": self._get_recommendations(),
        }

    def _get_recommendations(self) -> List[str]:
        """Get recommendations for improving capabilities."""
        recs = []

        pc = self._caps["planetary_computer"]
        if not pc["available"]:
            missing = [k for k, v in pc["packages"].items() if not v]
            recs.append(f"Install missing packages for Planetary Computer: pip install {' '.join(missing)}")

        nasa = self._caps["nasa_gesdisc"]
        if not nasa["has_credentials"] and nasa["packages"].get("requests") and nasa["packages"].get("netCDF4"):
            recs.append("Set NASA_EARTHDATA_USERNAME and NASA_EARTHDATA_PASSWORD for precipitation data")

        return recs


def get_capabilities() -> Dict:
    """Get current satellite data capabilities."""
    detector = CapabilityDetector.get_instance()
    return detector.get_capabilities_report()


def is_real_data_available() -> bool:
    """Check if real satellite data is available."""
    detector = CapabilityDetector.get_instance()
    return detector.should_use_real_data()


# =============================================================================
# Cache Management
# =============================================================================

class DataCache:
    """TTL file cache for satellite data."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_ttl_hours: int = 24
    ):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache files
            default_ttl_hours: Default time-to-live in hours
        """
        self.cache_dir = cache_dir or CapabilityDetector.get_instance().cache_dir
        self.default_ttl_hours = default_ttl_hours

    def _get_cache_key(self, prefix: str, **params) -> str:
        """Generate cache key from parameters."""
        param_str = json.dumps(params, sort_keys=True, default=str)
        hash_str = hashlib.md5(param_str.encode()).hexdigest()[:12]
        return f"{prefix}_{hash_str}"

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[Dict]:
        """Get cached data if valid."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                cached = json.load(f)

            cached_time = datetime.fromisoformat(cached["_cached_at"])
            ttl_hours = cached.get("_ttl_hours", self.default_ttl_hours)

            if datetime.now() - cached_time > timedelta(hours=ttl_hours):
                logger.debug(f"Cache expired for {key}")
                return None

            logger.debug(f"Cache hit for {key}")
            return cached.get("data")

        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            return None

    def set(self, key: str, data: Dict, ttl_hours: Optional[int] = None) -> None:
        """Store data in cache."""
        cache_path = self._get_cache_path(key)

        try:
            cached = {
                "data": data,
                "_cached_at": datetime.now().isoformat(),
                "_ttl_hours": ttl_hours or self.default_ttl_hours
            }

            with open(cache_path, 'w') as f:
                json.dump(cached, f, default=str)

            logger.debug(f"Cached data for {key}")

        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")

    def clear_expired(self) -> int:
        """Clear expired cache entries."""
        cleared = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    cached = json.load(f)

                cached_time = datetime.fromisoformat(cached["_cached_at"])
                ttl_hours = cached.get("_ttl_hours", self.default_ttl_hours)

                if datetime.now() - cached_time > timedelta(hours=ttl_hours):
                    cache_file.unlink()
                    cleared += 1

            except Exception:
                cache_file.unlink()
                cleared += 1

        if cleared > 0:
            logger.info(f"Cleared {cleared} expired cache entries")

        return cleared


# =============================================================================
# Retry Decorator
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator that retries a function with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after error: {e}. Waiting {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} retries exhausted for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# Planetary Computer Fetcher
# =============================================================================

class PlanetaryComputerFetcher:
    """Fetch satellite data from Microsoft Planetary Computer."""

    CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    # Collection IDs for different data types
    COLLECTIONS = {
        "sentinel2": "sentinel-2-l2a",
        "landsat8": "landsat-c2-l2",
        "landsat9": "landsat-c2-l2",
        "modis_terra": "modis-21A1D-061",
        "modis_aqua": "modis-22A2-061",
        "sentinel3_slstr": "sentinel-3-slstr-l2-lst",
        "sentinel5p_no2": "sentinel-5p-l2-netcdf",
        "sentinel5p_so2": "sentinel-5p-l2-netcdf",
        "sentinel5p_co": "sentinel-5p-l2-netcdf",
        "sentinel5p_ch4": "sentinel-5p-l2-netcdf",
        "viirs": "viirs",
    }

    def __init__(self, cache: Optional[DataCache] = None):
        """Initialize fetcher."""
        self.cache = cache or DataCache()
        self._detector = CapabilityDetector.get_instance()
        self._client = None

    @property
    def available(self) -> bool:
        """Check if Planetary Computer is available."""
        return self._detector.can_use_planetary_computer

    def _get_client(self):
        """Lazy load pystac client."""
        if self._client is None and self.available:
            try:
                import pystac_client
                self._client = pystac_client.Client.open(self.CATALOG_URL)
            except Exception as e:
                logger.warning(f"Failed to connect to Planetary Computer: {e}")
                self._client = None
        return self._client

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def search_items(
        self,
        collection: str,
        bbox: List[float],
        date: str,
        days_range: int = 7,
        max_items: int = 10,
        query: Optional[Dict] = None
    ) -> List[Any]:
        """Search for STAC items."""
        if not self.available:
            return []

        collection_id = self.COLLECTIONS.get(collection, collection)

        target_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = (target_date - timedelta(days=days_range)).strftime("%Y-%m-%d")
        end_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
        datetime_range = f"{start_date}/{end_date}"

        client = self._get_client()
        if client is None:
            return []

        try:
            search_params = {
                "collections": [collection_id],
                "bbox": bbox,
                "datetime": datetime_range,
                "max_items": max_items,
            }

            if query:
                search_params["query"] = query

            search = client.search(**search_params)
            items = list(search.items())

            logger.info(f"Found {len(items)} items for {collection_id}")
            return items

        except Exception as e:
            logger.error(f"Search failed for {collection_id}: {e}")
            return []

    def sign_items(self, items: List[Any]) -> List[Any]:
        """Sign items for Planetary Computer access."""
        try:
            import planetary_computer as pc

            signed = []
            for item in items:
                try:
                    signed.append(pc.sign(item))
                except Exception as e:
                    logger.warning(f"Failed to sign item {item.id}: {e}")

            return signed

        except ImportError:
            logger.warning("planetary_computer not installed, items not signed")
            return items

    def load_data(
        self,
        items: List[Any],
        bands: List[str],
        bbox: Optional[List[float]] = None,
        resolution: float = 0.0001
    ) -> Optional[Any]:
        """Load satellite data into xarray Dataset."""
        if not items:
            return None

        try:
            from odc import stac

            signed_items = self.sign_items(items)

            load_params = {
                "items": signed_items,
                "bands": bands,
                "preserve_original_order": True,
            }

            if bbox:
                load_params["bbox"] = bbox
                load_params["resolution"] = resolution

            ds = stac.load(**load_params)
            return ds

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None

    def compute_band_statistics(
        self,
        ds,
        band: str,
        compute_ndvi: bool = False,
        red_band: str = "B04",
        nir_band: str = "B08"
    ) -> Dict[str, float]:
        """Compute statistics from band data."""
        stats = {}

        try:
            if compute_ndvi and red_band in ds and nir_band in ds:
                red = ds[red_band].values.astype(float)
                nir = ds[nir_band].values.astype(float)

                valid = (red > 0) & (nir > 0)
                red = np.where(valid, red, np.nan)
                nir = np.where(valid, nir, np.nan)

                ndvi = (nir - red) / (nir + red + 1e-10)

                stats["ndvi_mean"] = float(np.nanmean(ndvi))
                stats["ndvi_std"] = float(np.nanstd(ndvi))
                stats["ndvi_median"] = float(np.nanmedian(ndvi))
                stats["valid_pixels"] = int(np.sum(valid))

            elif band in ds:
                data = ds[band].values.astype(float)

                valid = ~np.isnan(data) & (data > -9999)
                data = np.where(valid, data, np.nan)

                stats["mean"] = float(np.nanmean(data))
                stats["std"] = float(np.nanstd(data))
                stats["median"] = float(np.nanmedian(data))
                stats["min"] = float(np.nanmin(data))
                stats["max"] = float(np.nanmax(data))
                stats["valid_pixels"] = int(np.sum(valid))

        except Exception as e:
            logger.error(f"Failed to compute statistics: {e}")

        return stats


# =============================================================================
# NASA GES DISC Fetcher (Precipitation)
# =============================================================================

class NASAGESDISCFetcher:
    """Fetch precipitation data from NASA GES DISC (GPM/IMERG)."""

    IMERG_BASE_URL = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07"

    def __init__(self, cache: Optional[DataCache] = None):
        """Initialize fetcher."""
        self.cache = cache or DataCache()
        self._detector = CapabilityDetector.get_instance()

        self.username = os.environ.get("NASA_EARTHDATA_USERNAME")
        self.password = os.environ.get("NASA_EARTHDATA_PASSWORD")

    @property
    def available(self) -> bool:
        """Check if NASA GES DISC is available."""
        return self._detector.can_use_nasa_gesdisc

    def _is_date_fetchable(self, date: str) -> bool:
        """Check if a date is likely to have NASA GPM data available.
        
        NASA GPM IMERG data has:
        - 3-5 day latency (recent data not available)
        - Only historical data (no future data)
        - Archive typically goes back to 2000
        """
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
            today = datetime.now()
            
            # Skip future dates
            if target_date > today:
                return False
            
            # Skip recent dates (NASA has 3-5 day latency)
            min_available_date = today - timedelta(days=7)
            if target_date > min_available_date:
                return False
            
            # Skip very old dates (GPM starts from March 2014)
            gpm_start = datetime(2014, 3, 1)
            if target_date < gpm_start:
                return False
                
            return True
        except:
            return False

    def _build_imerg_url(self, date: str) -> str:
        """Build IMERG data URL for a date."""
        dt = datetime.strptime(date, "%Y-%m-%d")
        year = dt.year
        month = dt.month
        day = dt.day

        filename = f"3B-DAY.MS.MRG.3IMERG.{year}{month:02d}{day:02d}-S000000-E235959.V07B.nc4"
        return f"{self.IMERG_BASE_URL}/{year}/{month:02d}/{filename}"

    def _should_abort_after_404(self, target_date: datetime, fetch_date: datetime, today: datetime) -> bool:
        """Stop retrying nearby dates when the archive for the target window is clearly unavailable."""
        if fetch_date.year == target_date.year and fetch_date.month == target_date.month:
            if (today - target_date).days <= 60:
                return True
        return False

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def fetch_precipitation(
        self,
        bbox: List[float],
        date: str,
        days_range: int = 7
    ) -> Optional[Dict]:
        """Fetch precipitation data for a region using earthaccess."""
        if not self.available:
            return None

        # Check if date is fetchable (not future, not too recent)
        if not self._is_date_fetchable(date):
            logger.debug(f"Date {date} not fetchable (future or too recent)")
            return None

        cache_key = self.cache._get_cache_key(
            "precip",
            bbox=tuple(bbox),
            date=date,
            days=days_range
        )

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            import earthaccess
            from netCDF4 import Dataset

            # Login using environment variables (EARTHDATA_USERNAME/PASSWORD)
            earthaccess.login(strategy="environment")

            target_date = datetime.strptime(date, "%Y-%m-%d")
            today = datetime.now()
            
            min_available_date = today - timedelta(days=7)
            if target_date > min_available_date:
                logger.debug(f"NASA GPM data not yet available for {date}")
                return None
            
            gpm_start = datetime(2014, 3, 1)
            if target_date < gpm_start:
                return None
            
            days_range = min(days_range, 7)
            start_date = (target_date - timedelta(days=days_range - 1)).strftime("%Y-%m-%d")
            end_date = date

            # Search for granules using earthaccess
            results = earthaccess.search_data(
                short_name='GPM_3IMERGDF',
                version='07',
                temporal=(start_date, end_date),
                bounding_box=tuple(bbox)
            )

            if not results:
                logger.info(f"No NASA GPM granules found for {start_date} to {end_date}")
                return None

            logger.info(f"Found {len(results)} NASA GPM granules")

            # Download granules
            downloaded = earthaccess.download(results, "/tmp/gpm_data")
            
            precip_values = []
            
            for local_path in downloaded:
                if not local_path or not Path(local_path).exists():
                    continue
                try:
                    with Dataset(str(local_path)) as nc:
                        # GPM IMERG v07 uses "precipitation" (not "precipitationCal")
                        precip_var = "precipitation" if "precipitation" in nc.variables else "precipitationCal"
                        precip = nc.variables[precip_var][:]
                        lats = nc.variables["lat"][:]
                        lons = nc.variables["lon"][:]

                        west, south, east, north = bbox
                        lat_mask = (lats >= south) & (lats <= north)
                        lon_mask = (lons >= west) & (lons <= east)

                        # Handle both (time, lon, lat) and (lat, lon) dimensions
                        if precip.ndim == 3:
                            precip_region = precip[0, lon_mask, :][:, lat_mask]
                        else:
                            precip_region = precip[lat_mask, :][:, lon_mask]
                        daily_mean = float(np.nanmean(precip_region)) * 24
                        precip_values.append(daily_mean)
                except Exception as e:
                    logger.warning(f"Failed to read GPM data file: {e}")
                finally:
                    try:
                        Path(local_path).unlink(missing_ok=True)
                    except:
                        pass

            if not precip_values:
                return None

            mean_precip = np.mean(precip_values)
            std_precip = np.std(precip_values)

            result = {
                "daily_precip_mm": round(mean_precip, 2),
                "precip_std": round(std_precip, 2),
                "days_averaged": len(precip_values),
                "data_source": "GPM_IMERG",
                "fetch_date": date,
            }

            self.cache.set(cache_key, result, ttl_hours=24)
            return result

        except ImportError as e:
            logger.warning(f"Required packages not installed: {e}")
            return None

        except Exception as e:
            logger.error(f"Failed to fetch precipitation data: {e}")
            return None


# =============================================================================
# Unified Interface - Automatically uses real data when available
# =============================================================================

# Global instances for reuse
_fetcher_pc = None
_fetcher_nasa = None
_cache = None

def _get_fetchers() -> Tuple[PlanetaryComputerFetcher, NASAGESDISCFetcher, DataCache]:
    """Get or create global fetcher instances."""
    global _fetcher_pc, _fetcher_nasa, _cache

    if _cache is None:
        _cache = DataCache()

    if _fetcher_pc is None:
        _fetcher_pc = PlanetaryComputerFetcher(_cache)

    if _fetcher_nasa is None:
        _fetcher_nasa = NASAGESDISCFetcher(_cache)

    return _fetcher_pc, _fetcher_nasa, _cache


def get_real_data(
    data_type: str,
    bbox: List[float],
    date: str,
    collection: Optional[str] = None,
    bands: Optional[List[str]] = None,
    **kwargs
) -> Optional[Dict]:
    """
    Unified interface for fetching real satellite data.

    Automatically uses real data if packages are installed, falls back to None.

    Args:
        data_type: Type of data ("ndvi", "sst", "thermal", "nighttime", "atmospheric", "precipitation")
        bbox: Bounding box [west, south, east, north]
        date: Date string (YYYY-MM-DD)
        collection: Collection ID (optional, uses default if not provided)
        bands: List of bands to fetch (optional)
        **kwargs: Additional parameters

    Returns:
        Dictionary with data metrics or None if fetch failed
    """
    pc_fetcher, nasa_fetcher, cache = _get_fetchers()

    if data_type == "ndvi":
        if not pc_fetcher.available:
            return None

        collection = collection or "sentinel2"
        bands = bands or ["B04", "B08"]

        items = pc_fetcher.search_items(
            collection=collection,
            bbox=bbox,
            date=date,
            query={"eo:cloud_cover": {"lt": kwargs.get("max_cloud_cover", 30)}}
        )

        if not items:
            return None

        ds = pc_fetcher.load_data(items, bands, bbox=bbox)
        if ds is None:
            return None

        stats = pc_fetcher.compute_band_statistics(ds, "", compute_ndvi=True)
        stats["data_source"] = "Sentinel-2 (Real)"
        stats["fetch_date"] = date
        return stats

    elif data_type == "sst":
        if not pc_fetcher.available:
            return None

        collection = collection or "modis_terra"
        bands = bands or ["LST_Day_1km"]

        items = pc_fetcher.search_items(collection=collection, bbox=bbox, date=date)

        if not items:
            return None

        ds = pc_fetcher.load_data(items, bands, bbox=bbox)
        if ds is None:
            return None

        stats = pc_fetcher.compute_band_statistics(ds, bands[0])
        if "mean" in stats:
            stats["sst_celsius"] = round(stats["mean"] - 273.15, 2)
        stats["data_source"] = "MODIS (Real)"
        stats["fetch_date"] = date
        return stats

    elif data_type == "thermal":
        if not pc_fetcher.available:
            return None

        collection = collection or "landsat9"
        bands = bands or ["ST_B10"]

        items = pc_fetcher.search_items(
            collection=collection,
            bbox=bbox,
            date=date,
            query={"eo:cloud_cover": {"lt": kwargs.get("max_cloud_cover", 20)}}
        )

        if not items:
            return None

        ds = pc_fetcher.load_data(items, bands, bbox=bbox)
        if ds is None:
            return None

        stats = pc_fetcher.compute_band_statistics(ds, bands[0])
        if "mean" in stats:
            stats["temperature_celsius"] = round(stats["mean"] - 273.15, 2)
        stats["data_source"] = "Landsat (Real)"
        stats["fetch_date"] = date
        return stats

    elif data_type == "nighttime":
        if not pc_fetcher.available:
            return None

        collection = collection or "viirs"
        bands = bands or ["DNB"]

        items = pc_fetcher.search_items(collection=collection, bbox=bbox, date=date)

        if not items:
            return None

        ds = pc_fetcher.load_data(items, bands, bbox=bbox)
        if ds is None:
            return None

        stats = pc_fetcher.compute_band_statistics(ds, bands[0])
        stats["intensity"] = stats.get("mean", 0)
        stats["data_source"] = "VIIRS (Real)"
        stats["fetch_date"] = date
        return stats

    elif data_type == "atmospheric":
        if not pc_fetcher.available:
            return None

        gas_type = kwargs.get("gas", "no2")
        collection = collection or f"sentinel5p_{gas_type}"
        bands = bands or [gas_type.upper()]

        items = pc_fetcher.search_items(collection=collection, bbox=bbox, date=date)

        if not items:
            return None

        ds = pc_fetcher.load_data(items, bands, bbox=bbox)
        if ds is None:
            return None

        stats = pc_fetcher.compute_band_statistics(ds, bands[0])
        stats["concentration"] = stats.get("mean", 0)
        stats["data_source"] = "Sentinel-5P (Real)"
        stats["fetch_date"] = date
        stats["gas_type"] = gas_type
        return stats

    elif data_type == "precipitation":
        return nasa_fetcher.fetch_precipitation(
            bbox=bbox,
            date=date,
            days_range=kwargs.get("days_range", 7)
        )

    else:
        logger.warning(f"Unknown data type: {data_type}")
        return None


# =============================================================================
# Utility Functions
# =============================================================================

def enable_real_data():
    """Force enable real satellite data fetching."""
    os.environ["USE_REAL_SATELLITE_DATA"] = "true"


def disable_real_data():
    """Force disable real satellite data fetching (use simulated)."""
    os.environ["USE_REAL_SATELLITE_DATA"] = "false"


def is_real_data_enabled() -> bool:
    """Check if real data fetching is enabled (auto-detect or env var)."""
    return is_real_data_available()


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    cache = DataCache()
    cache_files = list(cache.cache_dir.glob("*.json"))
    total_size = sum(f.stat().st_size for f in cache_files)

    return {
        "cache_dir": str(cache.cache_dir),
        "file_count": len(cache_files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }


def install_missing_packages():
    """Attempt to install missing packages for satellite data."""
    caps = get_capabilities()

    packages_to_install = []

    # Check Planetary Computer packages
    pc = caps.get("planetary_computer", {})
    if not pc.get("available"):
        missing = [k for k, v in pc.get("packages", {}).items() if not v]
        if missing:
            # Map package names to pip names
            pip_names = {
                "pystac_client": "pystac-client",
                "planetary_computer": "planetary-computer",
                "odc_stac": "odc-stac",
            }
            packages_to_install.extend([pip_names.get(p, p) for p in missing])

    if packages_to_install:
        logger.info(f"Installing missing packages: {packages_to_install}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + packages_to_install)
            logger.info("Packages installed successfully. Restart to use.")
        except Exception as e:
            logger.error(f"Failed to install packages: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("SATELLITE DATA FETCHER - AUTOMATED CAPABILITIES")
    print("=" * 70)

    # Show capabilities
    caps = get_capabilities()
    print(f"\nReal Data Enabled: {caps['real_data_enabled']}")
    print(f"\nPlanetary Computer:")
    print(f"  Available: {caps['planetary_computer']['available']}")
    print(f"  Note: {caps['planetary_computer']['note']}")

    print(f"\nNASA GES DISC (Precipitation):")
    print(f"  Available: {caps['nasa_gesdisc']['available']}")
    print(f"  Note: {caps['nasa_gesdisc']['note']}")

    print(f"\nCache: {caps['cache_dir']}")

    if caps['recommendations']:
        print(f"\nRecommendations:")
        for rec in caps['recommendations']:
            print(f"  - {rec}")

    # Show cache stats
    stats = get_cache_stats()
    print(f"\nCache Stats: {stats['file_count']} files, {stats['total_size_mb']} MB")
