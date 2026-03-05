#!/usr/bin/env python3
from pystac_client import Client
import json
from collections import defaultdict

with open('configs/aoi_hormuz.geojson') as f:
    aoi = json.load(f)
aoi_geom = aoi['features'][0]['geometry']

client = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')

# Search Feb 2024
results = defaultdict(list)
for day in range(1, 29):
    start = f'2024-02-{day:02d'
    end = f'2024-02-{day}02d'

    search = client.search(
        collections=['sentinel-1-grd'],
        intersects=aoi_geom,
        datetime=f'{start}/{end}'
    )
    items = list(search.items())
    if items:
        # Filter for IW + VV
        filtered = []
        for item in items:
            mode = item.properties.get('sar:instrument_mode', '')
            pols = item.properties.get('sar:polarizations', [])
            if mode == 'IW' and 'VV' in pols:
                filtered.append(item)

        if filtered:
            date_str = items[0].properties.get('datetime', '')[:10]
            results[date_str].append(len(filtered))

for date_str in sorted(results.keys()):
    print(f'{date_str}: {len(results[date_str])} scenes')
