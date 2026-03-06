"""
AIS Data Loading & Validation

Provides utilities for loading, validating, and preparing AIS data
for calibration against satellite-derived metrics.
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
from shapely.geometry import Point, box
from shapely.ops import unary_union


# Required columns for AIS data
REQUIRED_COLUMNS = ['mmsi', 'datetime', 'latitude', 'longitude']

# Optional but useful columns
OPTIONAL_COLUMNS = [
    'speed', 'heading', 'course', 'status',
    'vessel_name', 'vessel_type', 'length', 'width', 'draught'
]


def validate_ais_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate AIS data against expected schema.

    Args:
        df: AIS DataFrame to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Validate mmsi (9-digit integer)
    if 'mmsi' in df.columns:
        invalid_mmsi = df[
            (df['mmsi'] < 100000000) | (df['mmsi'] > 999999999)
        ]
        if len(invalid_mmsi) > 0:
            errors.append(f"Invalid MMSI values: {len(invalid_mmsi)} rows")

    # Validate latitude
    if 'latitude' in df.columns:
        invalid_lat = df[
            (df['latitude'] < -90) | (df['latitude'] > 90)
        ]
        if len(invalid_lat) > 0:
            errors.append(f"Invalid latitude values: {len(invalid_lat)} rows")

    # Validate longitude
    if 'longitude' in df.columns:
        invalid_lon = df[
            (df['longitude'] < -180) | (df['longitude'] > 180)
        ]
        if len(invalid_lon) > 0:
            errors.append(f"Invalid longitude values: {len(invalid_lon)} rows")

    # Validate datetime
    if 'datetime' in df.columns:
        try:
            pd.to_datetime(df['datetime'], utc=True)
        except Exception as e:
            errors.append(f"Invalid datetime format: {str(e)}")

    # Validate speed (0-102.2 knots)
    if 'speed' in df.columns:
        invalid_speed = df[
            (df['speed'] < 0) | (df['speed'] > 102.2)
        ]
        if len(invalid_speed) > 0:
            errors.append(f"Invalid speed values: {len(invalid_speed)} rows")

    # Validate heading (0-359 or 511 for N/A)
    if 'heading' in df.columns:
        invalid_heading = df[
            (df['heading'] < 0) | (~df['heading'].isin(range(360)) & (df['heading'] != 511))
        ]
        if len(invalid_heading) > 0:
            errors.append(f"Invalid heading values: {len(invalid_heading)} rows")

    is_valid = len(errors) == 0
    return is_valid, errors


def load_ais_data(
    ais_path: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load AIS data from Parquet file with optional filtering.

    Args:
        ais_path: Path to AIS Parquet file
        start_date: Optional start date filter
        end_date: Optional end date filter
        aoi_bounds: Optional AOI bounds (min_lon, min_lat, max_lon, max_lat)
        validate: Whether to validate data (default True)

    Returns:
        Filtered AIS DataFrame
    """
    df = pd.read_parquet(ais_path)

    # Validate if requested
    if validate:
        is_valid, errors = validate_ais_data(df)
        if not is_valid:
            print(f"Warning: AIS data validation issues:")
            for error in errors:
                print(f"  - {error}")

    # Ensure datetime is parsed
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    # Filter by date range
    if start_date is not None:
        start_ts = pd.Timestamp(start_date, tz='UTC')
        df = df[df['datetime'] >= start_ts]

    if end_date is not None:
        end_ts = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
        df = df[df['datetime'] < end_ts]

    # Filter by AOI bounds
    if aoi_bounds is not None:
        min_lon, min_lat, max_lon, max_lat = aoi_bounds
        df = df[
            (df['latitude'] >= min_lat) &
            (df['latitude'] <= max_lat) &
            (df['longitude'] >= min_lon) &
            (df['longitude'] <= max_lon)
        ]

    print(f"Loaded {len(df)} AIS messages")
    if 'mmsi' in df.columns:
        print(f"Unique vessels: {df['mmsi'].nunique()}")

    return df


def filter_ais_by_geojson(
    ais_df: pd.DataFrame,
    aoi_path: str
) -> pd.DataFrame:
    """
    Filter AIS data to AOI defined in GeoJSON.

    Args:
        ais_df: AIS DataFrame
        aoi_path: Path to AOI GeoJSON file

    Returns:
        Filtered AIS DataFrame
    """
    with open(aoi_path) as f:
        aoi = json.load(f)

    from shapely.geometry import shape

    aoi_geom = shape(aoi['features'][0]['geometry'])

    # Create points and filter
    points = [Point(row['longitude'], row['latitude'])
              for _, row in ais_df.iterrows()]

    mask = [aoi_geom.contains(p) for p in points]

    return ais_df[mask].reset_index(drop=True)


def compute_ais_coverage(
    ais_df: pd.DataFrame,
    aoi_bounds: Tuple[float, float, float, float],
    grid_size: float = 0.1
) -> dict:
    """
    Compute AIS coverage over AOI grid.

    Args:
        ais_df: AIS DataFrame
        aoi_bounds: AOI bounds (min_lon, min_lat, max_lon, max_lat)
        grid_size: Grid cell size in degrees

    Returns:
        Coverage statistics dict
    """
    min_lon, min_lat, max_lon, max_lat = aoi_bounds

    # Create grid
    lons = np.arange(min_lon, max_lon + grid_size, grid_size)
    lats = np.arange(min_lat, max_lat + grid_size, grid_size)

    # Count messages per cell
    grid_counts = np.zeros((len(lats) - 1, len(lons) - 1))

    for _, row in ais_df.iterrows():
        lon_idx = int((row['longitude'] - min_lon) / grid_size)
        lat_idx = int((row['latitude'] - min_lat) / grid_size)

        if 0 <= lon_idx < len(lons) - 1 and 0 <= lat_idx < len(lats) - 1:
            grid_counts[lat_idx, lon_idx] += 1

    # Compute coverage
    cells_with_data = (grid_counts > 0).sum()
    total_cells = grid_counts.size

    return {
        'cells_total': int(total_cells),
        'cells_with_ais': int(cells_with_data),
        'coverage_fraction': cells_with_data / total_cells if total_cells > 0 else 0,
        'mean_messages_per_cell': float(grid_counts[grid_counts > 0].mean()) if cells_with_data > 0 else 0
    }


def prepare_ais_for_calibration(
    ais_df: pd.DataFrame,
    gate_coords: List[Tuple[float, float]]
) -> pd.DataFrame:
    """
    Prepare AIS data for calibration by computing daily statistics.

    Args:
        ais_df: AIS DataFrame
        gate_coords: Gate line coordinates [(lon1, lat1), (lon2, lat2)]

    Returns:
        Daily AIS statistics DataFrame
    """
    from shapely.geometry import LineString

    gate_line = LineString(gate_coords)

    # Add date column
    ais_df = ais_df.copy()
    ais_df['date'] = pd.to_datetime(ais_df['datetime']).dt.date

    # Group by vessel and detect crossings
    crossings = []

    for mmsi, vessel_df in ais_df.groupby('mmsi'):
        vessel_df = vessel_df.sort_values('datetime')

        points = [Point(row['longitude'], row['latitude'])
                  for _, row in vessel_df.iterrows()]

        for i in range(1, len(points)):
            segment = LineString([points[i-1], points[i]])

            if segment.crosses(gate_line):
                crossing_point = segment.intersection(gate_line)

                # Determine direction from heading or position change
                heading = vessel_df.iloc[i-1].get('heading', None)
                if heading is not None and heading != 511:
                    direction = "in" if heading < 180 else "out"
                else:
                    # Use position change
                    dy = points[i].y - points[i-1].y
                    direction = "in" if dy > 0 else "out"

                crossings.append({
                    'mmsi': mmsi,
                    'date': vessel_df.iloc[i]['date'],
                    'datetime': vessel_df.iloc[i]['datetime'],
                    'crossing_lon': crossing_point.x,
                    'crossing_lat': crossing_point.y,
                    'direction': direction
                })

    crossings_df = pd.DataFrame(crossings)

    # Aggregate to daily
    if len(crossings_df) == 0:
        return pd.DataFrame(columns=['date', 'ais_gc_in', 'ais_gc_out', 'ais_gc_total'])

    daily = []
    for date_val, group in crossings_df.groupby('date'):
        gc_in = len(group[group['direction'] == 'in'])
        gc_out = len(group[group['direction'] == 'out'])

        daily.append({
            'date': date_val.isoformat() if hasattr(date_val, 'isoformat') else str(date_val),
            'ais_gc_in': gc_in,
            'ais_gc_out': gc_out,
            'ais_gc_total': gc_in + gc_out
        })

    return pd.DataFrame(daily)


def create_sample_ais_data(
    output_path: str,
    start_date: date,
    end_date: date,
    aoi_bounds: Tuple[float, float, float, float],
    num_vessels: int = 100,
    seed: int = 42
) -> str:
    """
    Create synthetic AIS data for testing purposes.

    Args:
        output_path: Output Parquet file path
        start_date: Start date
        end_date: End date
        aoi_bounds: AOI bounds (min_lon, min_lat, max_lon, max_lat)
        num_vessels: Number of vessels to simulate
        seed: Random seed

    Returns:
        Output file path
    """
    np.random.seed(seed)

    min_lon, min_lat, max_lon, max_lat = aoi_bounds

    # Generate random vessels
    mmsis = np.random.randint(100000000, 999999999, size=num_vessels)

    records = []
    current_date = start_date

    while current_date <= end_date:
        # Generate messages for each vessel
        for mmsi in mmsis:
            # Random number of messages per day (1-10)
            num_msgs = np.random.randint(1, 11)

            for _ in range(num_msgs):
                # Random position within AOI
                lat = np.random.uniform(min_lat, max_lat)
                lon = np.random.uniform(min_lon, max_lon)

                # Random time during the day
                hour = np.random.randint(0, 24)
                minute = np.random.randint(0, 60)

                dt = datetime(
                    current_date.year, current_date.month, current_date.day,
                    hour, minute, 0
                )

                records.append({
                    'mmsi': int(mmsi),
                    'datetime': dt,
                    'latitude': lat,
                    'longitude': lon,
                    'speed': np.random.uniform(0, 25),
                    'heading': np.random.randint(0, 360),
                    'course': np.random.randint(0, 360),
                    'status': np.random.choice([0, 1, 5, 8])
                })

        current_date = date(
            current_date.year,
            current_date.month,
            current_date.day + 1
        )

    df = pd.DataFrame(records)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"Created synthetic AIS data: {len(df)} messages, {num_vessels} vessels")
    print(f"Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AIS Data Utilities")
    parser.add_argument("--validate", type=str, help="Validate AIS Parquet file")
    parser.add_argument("--create-sample", type=str, help="Create sample AIS data")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.validate:
        df = pd.read_parquet(args.validate)
        is_valid, errors = validate_ais_data(df)
        if is_valid:
            print("✓ AIS data is valid")
        else:
            print("✗ Validation errors:")
            for e in errors:
                print(f"  - {e}")

    elif args.create_sample:
        from datetime import date as parse_date

        start = parse_date.fromisoformat(args.start) if args.start else date(2024, 2, 22)
        end = parse_date.fromisoformat(args.end) if args.end else date(2024, 2, 28)

        # Strait of Hormuz bounds
        aoi_bounds = (56.0, 24.5, 58.0, 27.0)

        create_sample_ais_data(
            output_path=args.create_sample,
            start_date=start,
            end_date=end,
            aoi_bounds=aoi_bounds
        )

    else:
        parser.print_help()
