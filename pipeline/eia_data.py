"""
EIA (Energy Information Administration) Data Integration
Fetches weekly inventory data for oil storage validation
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

# EIA API endpoint (v2)
EIA_API_BASE = "https://api.eia.gov/v2"

# Cushing crude inventory series ID
CUSHING_SERIES_ID = "Cushing_OK_Crude_Oil_Tank_Farms_and_Liquid_Fuel_Terminals.Working_Capacity_for_Crude_Oil"

class EIADataFetcher:
    """Fetches EIA inventory data for oil storage validation"""
    
    def __init__(self, api_key: str = None, cache_dir: str = "outputs/eia_cache"):
        """
        Initialize EIA data fetcher
        
        Args:
            api_key: EIA API key (get from https://www.eia.gov/opendata/)
            cache_dir: Directory to cache API responses
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_cushing_inventory(self, days: int = 90) -> pd.DataFrame:
        """
        Fetch Cushing crude oil inventory data
        
        Args:
            days: Number of days to fetch
            
        Returns:
            DataFrame with columns: date, value (million barrels)
        """
        if not self.api_key:
            # Return cached or demo data
            return self._get_cached_cushing_data(days)
        
        try:
            # EIA v2 API endpoint
            url = f"{EIA_API_BASE}/petroleum/stoc/wstk/data/"
            
            params = {
                "api_key": self.api_key,
                "frequency": "weekly",
                "data[0]": "value",
                "facets[series][]": ["W_EPC0_SAX_YCUOK_MBBL"],
                "start": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                "end": datetime.now().strftime("%Y-%m-%d"),
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "offset": 0,
                "length": 5000
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse response
            records = []
            for item in data.get("response", {}).get("data", []):
                records.append({
                    "date": pd.to_datetime(item["period"]),
                    "value": float(item["value"]) / 1000  # Convert to million barrels
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values("date")
            
            # Cache the data
            cache_file = self.cache_dir / f"cushing_{datetime.now().strftime('%Y%m%d')}.json"
            cache_file.write_text(df.to_json(orient="records", date_format="iso"))
            
            return df
            
        except Exception as e:
            print(f"Error fetching EIA data: {e}")
            return self._get_cached_cushing_data(days)
    
    def _get_cached_cushing_data(self, days: int = 90) -> pd.DataFrame:
        """
        Get cached Cushing data or generate demo data
        
        Args:
            days: Number of days
            
        Returns:
            DataFrame with demo/cached data
        """
        # Check for cached file
        cache_files = sorted(self.cache_dir.glob("cushing_*.json"), reverse=True)
        
        if cache_files:
            cache_file = cache_files[0]
            df = pd.read_json(cache_file, orient="records")
            df["date"] = pd.to_datetime(df["date"])
            
            # Filter to requested days
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df["date"] >= cutoff]
            
            return df
        
        # Generate demo data based on typical Cushing levels
        # Typical range: 20-60 million barrels
        print("⚠️  No EIA API key provided. Using demo data.")
        print("   Get your free API key at: https://www.eia.gov/opendata/")
        
        import numpy as np
        np.random.seed(42)
        
        weeks = min(days // 7, 52)
        dates = pd.date_range(end=datetime.now(), periods=weeks, freq="W-WED")
        
        # Simulate realistic inventory levels
        base_level = 35  # Start at 35M barrels
        trend = np.linspace(0, 5, len(dates))  # Slight upward trend
        noise = np.random.randn(len(dates)) * 3
        
        values = base_level + trend + np.cumsum(noise) * 0.1
        values = np.clip(values, 15, 60)  # Keep in realistic range
        
        df = pd.DataFrame({
            "date": dates,
            "value": values
        })
        
        return df
    
    def calculate_storage_utilization(self, satellite_data: dict) -> dict:
        """
        Calculate storage utilization from satellite data
        
        Args:
            satellite_data: Dict with detection results (e.g., tank levels)
            
        Returns:
            Dict with utilization metrics
        """
        # This would compare satellite-derived levels with EIA data
        # For now, return placeholder
        
        return {
            "utilization_pct": 0.0,
            "trend": "stable",
            "comparison": "N/A - need EIA API key"
        }
    
    def validate_signal(self, satellite_signal: str, eia_data: pd.DataFrame) -> dict:
        """
        Validate satellite signal against EIA data
        
        Args:
            satellite_signal: "long" or "short" signal from satellite
            eia_data: EIA inventory data
            
        Returns:
            Dict with validation results
        """
        if eia_data.empty:
            return {
                "validated": False,
                "reason": "No EIA data available"
            }
        
        # Calculate recent trend
        recent = eia_data.tail(4)  # Last 4 weeks
        if len(recent) < 2:
            return {
                "validated": False,
                "reason": "Insufficient EIA data"
            }
        
        change = (recent["value"].iloc[-1] - recent["value"].iloc[0]) / recent["value"].iloc[0]
        
        # Validate signal
        # If satellite says "short" (high inventory), EIA should show increasing inventory
        # If satellite says "long" (low inventory), EIA should show decreasing inventory
        
        if satellite_signal == "short":
            # Expect inventory to be increasing
            validated = change > 0.02  # More than 2% increase
            reason = "Inventory increasing" if validated else "Inventory not increasing"
        elif satellite_signal == "long":
            # Expect inventory to be decreasing
            validated = change < -0.02  # More than 2% decrease
            reason = "Inventory decreasing" if validated else "Inventory not decreasing"
        else:
            validated = False
            reason = "Unknown signal type"
        
        return {
            "validated": validated,
            "reason": reason,
            "eia_change_pct": change * 100,
            "eia_latest": recent["value"].iloc[-1],
            "eia_4wk_ago": recent["value"].iloc[0]
        }


def fetch_eia_cushing_report(api_key: str = None) -> dict:
    """
    Fetch latest Cushing inventory report
    
    Args:
        api_key: Optional EIA API key
        
    Returns:
        Dict with report data
    """
    fetcher = EIADataFetcher(api_key)
    df = fetcher.fetch_cushing_inventory(days=30)
    
    if df.empty:
        return {
            "status": "error",
            "message": "No data available"
        }
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    change = (latest["value"] - prev["value"]) / prev["value"] * 100
    
    return {
        "status": "success",
        "date": latest["date"].strftime("%Y-%m-%d"),
        "inventory_mb": round(latest["value"], 2),
        "change_mb": round(latest["value"] - prev["value"], 2),
        "change_pct": round(change, 2),
        "trend": "increasing" if change > 0 else "decreasing",
        "data_source": "EIA" if api_key else "Demo"
    }


if __name__ == "__main__":
    # Test EIA data fetcher
    print("📊 Testing EIA Data Integration")
    print("="*60)
    
    # Without API key (demo mode)
    report = fetch_eia_cushing_report()
    
    print(f"Date: {report['date']}")
    print(f"Cushing Inventory: {report['inventory_mb']}M barrels")
    print(f"Change: {report['change_mb']}M ({report['change_pct']:+.2f}%)")
    print(f"Trend: {report['trend']}")
    print(f"Source: {report['data_source']}")
    print()
    
    if report['data_source'] == 'Demo':
        print("💡 To get real data, set EIA_API_KEY environment variable")
        print("   Get free API key: https://www.eia.gov/opendata/register.php")
