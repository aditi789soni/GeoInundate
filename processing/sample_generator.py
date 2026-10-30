"""
Sample Sentinel-2 Flood Dataset Generator.
Generates realistic multi-band Sentinel-2 imagery (Green, NIR, SWIR)
representing pre- and post-flood disaster states, complete with real-world UTM
georeferencing and administrative boundary layers.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, Polygon
import geopandas as gpd

from config import SAMPLE_DIR


def generate_sample_flood_data(output_dir: Path = SAMPLE_DIR) -> dict:
    """
    Synthesize realistic Sentinel-2 Surface Reflectance (L2A) datasets:
    - Pre-disaster: Permanent river channel with dry surrounding floodplains.
    - Post-disaster: River overflow causing extensive inundation of floodplains.
    - Band 1: Green (B03)
    - Band 2: NIR (B08)
    - Band 3: SWIR1 (B11)

    Georeferenced in UTM Zone 43N (EPSG:32643) with 10m resolution.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_path = output_dir / "pre_disaster_sentinel2.tif"
    post_path = output_dir / "post_disaster_sentinel2.tif"
    admin_path = output_dir / "sample_administrative_boundaries.geojson"

    # Grid dimensions (500 x 500 pixels at 10m = 5km x 5km)
    height, width = 500, 500
    pixel_size = 10.0  # 10 meters

    # UTM Zone 43N (Indus / Flood basin coordinates)
    origin_x = 350000.0  # Easting (m)
    origin_y = 3100000.0  # Northing (m)
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    crs = "EPSG:32643"

    # Coordinate meshgrid for terrain/river generation
    y, x = np.mgrid[0:height, 0:width]

    # Sinuous river channel path: x_center = 250 + 60 * sin(y / 45) + 30 * cos(y / 80)
    river_center = 250 + 60 * np.sin(y / 45.0) + 30 * np.cos(y / 80.0)
    dist_to_river = np.abs(x - river_center)

    # Elevation model: gently slopes toward river channel with subtle depressions
    elevation = 20.0 + 0.08 * dist_to_river + 3.0 * np.sin(x / 40.0) * np.cos(y / 50.0)
    # Add an oxbow lake depression on left bank
    oxbow_mask = ((x - 140) ** 2 / (35 ** 2) + (y - 200) ** 2 / (70 ** 2)) < 1.0

    # Permanent river channel (width ~ 28 pixels = 280m)
    perm_river = dist_to_river < 14.0
    perm_water = perm_river | oxbow_mask

    # Flood inundation: water level rises by 4.5m, filling river and low-lying land
    flood_water = (elevation < 25.2) | (dist_to_river < 55.0) | oxbow_mask

    # Add realistic texture / agricultural field variability
    np.random.seed(42)
    noise = np.random.normal(0, 0.015, (height, width)).astype(np.float32)

    def create_scene_bands(water_mask: np.ndarray) -> np.ndarray:
        """Create B03 (Green), B08 (NIR), B11 (SWIR) reflectance arrays (0-10000 scale)."""
        # Vegetation / Land:
        # Green ~ 0.08-0.14, NIR ~ 0.35-0.55 (high chlorophyll), SWIR ~ 0.12-0.22 (moderate moisture)
        green = 0.10 + 0.03 * np.sin(x / 20.0) + noise
        nir = 0.42 + 0.08 * np.cos(y / 25.0) + noise
        swir = 0.18 + 0.04 * np.sin((x + y) / 30.0) + noise

        # Water features:
        # Low reflectance across all bands, very strong absorption in NIR & SWIR
        # Green ~ 0.07, NIR ~ 0.015, SWIR ~ 0.008
        green[water_mask] = 0.07 + np.random.normal(0, 0.004, np.sum(water_mask))
        nir[water_mask] = 0.015 + np.random.normal(0, 0.002, np.sum(water_mask))
        swir[water_mask] = 0.008 + np.random.normal(0, 0.001, np.sum(water_mask))

        # Clip reflectance to physically valid [0.001, 1.0]
        green = np.clip(green, 0.001, 1.0)
        nir = np.clip(nir, 0.001, 1.0)
        swir = np.clip(swir, 0.001, 1.0)

        # Scale to standard Sentinel-2 L2A integer DN (scale factor 10,000)
        b3_dn = (green * 10000).astype(np.uint16)
        b8_dn = (nir * 10000).astype(np.uint16)
        b11_dn = (swir * 10000).astype(np.uint16)

        return np.stack([b3_dn, b8_dn, b11_dn], axis=0)

    pre_bands = create_scene_bands(perm_water)
    post_bands = create_scene_bands(flood_water)

    # Write GeoTIFF files
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "lzw",
    }

    with rasterio.open(pre_path, "w", **profile) as dst:
        dst.write(pre_bands)
        dst.set_band_description(1, "B03_Green_560nm")
        dst.set_band_description(2, "B08_NIR_842nm")
        dst.set_band_description(3, "B11_SWIR_1610nm")

    with rasterio.open(post_path, "w", **profile) as dst:
        dst.write(post_bands)
        dst.set_band_description(1, "B03_Green_560nm")
        dst.set_band_description(2, "B08_NIR_842nm")
        dst.set_band_description(3, "B11_SWIR_1610nm")

    # Create Administrative Boundary polygons (4 districts covering the 5km x 5km scene)
    # min_x, max_x = origin_x, origin_x + 5000
    # min_y, max_y = origin_y - 5000, origin_y
    x_mid = origin_x + 2500.0
    y_mid = origin_y - 2500.0
    x_max = origin_x + 5000.0
    y_min = origin_y - 5000.0

    districts = [
        {"name": "North-West Basin", "geom": box(origin_x, y_mid, x_mid, origin_y)},
        {"name": "North-East Plateau", "geom": box(x_mid, y_mid, x_max, origin_y)},
        {"name": "South-West Lowlands", "geom": box(origin_x, y_min, x_mid, y_mid)},
        {"name": "South-East Delta", "geom": box(x_mid, y_min, x_max, y_mid)},
    ]

    gdf_admin = gpd.GeoDataFrame(
        [{"district": d["name"], "geometry": d["geom"]} for d in districts],
        crs=crs,
    )
    # Reproject to WGS84 for GeoJSON
    gdf_admin_wgs84 = gdf_admin.to_crs("EPSG:4326")
    gdf_admin_wgs84.to_file(admin_path, driver="GeoJSON")

    return {
        "pre_disaster_tif": str(pre_path),
        "post_disaster_tif": str(post_path),
        "admin_boundaries_geojson": str(admin_path),
    }


if __name__ == "__main__":
    result = generate_sample_flood_data()
    print("Generated sample datasets successfully:")
    for k, v in result.items():
        print(f"  {k}: {v}")
