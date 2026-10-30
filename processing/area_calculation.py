"""
Geospatial Area Calculation & Impact Statistics Module for GeoInundate.
Enforces projected equal-area / UTM coordinate reference systems for mathematically
accurate square kilometre measurements, polygon metrics, and administrative spatial overlays.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import geopandas as gpd
import numpy as np
import pyproj
from shapely.geometry import box

from processing.raster_loader import RasterMetadata


@dataclass
class FloodImpactSummary:
    """Consolidated metrics of flood inundation impact."""
    pre_water_area_km2: float
    post_water_area_km2: float
    newly_inundated_area_km2: float
    receded_water_area_km2: float
    total_scene_area_km2: float
    inundation_percentage: float
    num_flood_polygons: int
    mean_flood_patch_km2: float
    max_flood_patch_km2: float
    projection_used: str
    district_breakdown: Optional[List[Dict[str, float]]] = None


def get_projected_crs(gdf: gpd.GeoDataFrame) -> str:
    """
    Determine an accurate projected CRS (in meters) for geometry area calculation.
    If already projected in meters, preserves CRS.
    If geographic (WGS84 degrees), computes optimal local UTM zone.
    """
    if gdf.crs is None:
        return "EPSG:3857"

    crs_obj = pyproj.CRS.from_user_input(gdf.crs)

    # Check if already projected and units are meters
    if crs_obj.is_projected:
        axis_units = [axis.unit_name for axis in crs_obj.axis_info]
        if "metre" in axis_units or "meter" in axis_units:
            return str(gdf.crs)

    # If geographic, find centroid of data bounding box to determine UTM EPSG
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy] in degrees
    center_lon = (bounds[0] + bounds[2]) / 2.0
    center_lat = (bounds[1] + bounds[3]) / 2.0

    utm_zone = int((center_lon + 180) // 6) + 1
    epsg_code = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
    return f"EPSG:{epsg_code}"


def calculate_polygon_areas_km2(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Reproject GeoDataFrame to a projected meter-based CRS and append
    accurate 'area_m2' and 'area_km2' columns to each polygon.
    """
    if gdf.empty:
        gdf_res = gdf.copy()
        gdf_res["area_m2"] = 0.0
        gdf_res["area_km2"] = 0.0
        return gdf_res

    target_crs = get_projected_crs(gdf)
    gdf_proj = gdf.to_crs(target_crs) if str(gdf.crs) != target_crs else gdf.copy()

    # Shapely area in projected CRS gives square meters
    areas_m2 = gdf_proj.geometry.area
    gdf_res = gdf.copy()
    gdf_res["area_m2"] = areas_m2
    gdf_res["area_km2"] = areas_m2 / 1_000_000.0

    return gdf_res


def calculate_flood_impact(
    inundation_gdf: gpd.GeoDataFrame,
    permanent_gdf: gpd.GeoDataFrame,
    pre_water_mask: np.ndarray,
    post_water_mask: np.ndarray,
    metadata: RasterMetadata,
    admin_gdf: Optional[gpd.GeoDataFrame] = None,
) -> FloodImpactSummary:
    """
    Calculate comprehensive flood inundation statistics combining vector geometry
    measurements with pixel-count radiometric verification.
    """
    # 1. Pixel-based baseline area calculations
    # Note: Ensure resolution is in meters; if in degrees, use bounding box transform
    pixel_area_m2 = abs(metadata.res_x * metadata.res_y)
    crs_obj = pyproj.CRS.from_user_input(metadata.crs)
    if not crs_obj.is_projected:
        # Approximate meter conversion for EPSG:4326 pixel sizes
        center_lat = (metadata.bounds_wgs84[1] + metadata.bounds_wgs84[3]) / 2.0
        meters_per_deg_lat = 111139.0
        meters_per_deg_lon = 111139.0 * np.cos(np.radians(center_lat))
        pixel_area_m2 = abs(metadata.res_x * meters_per_deg_lon) * abs(metadata.res_y * meters_per_deg_lat)

    total_pixels = metadata.width * metadata.height
    total_scene_km2 = (total_pixels * pixel_area_m2) / 1_000_000.0

    pre_pixels = np.sum(pre_water_mask == 1)
    post_pixels = np.sum(post_water_mask == 1)

    pre_area_km2 = (pre_pixels * pixel_area_m2) / 1_000_000.0
    post_area_km2 = (post_pixels * pixel_area_m2) / 1_000_000.0

    # 2. Vector polygon-based area metrics for newly inundated areas
    gdf_inundated_with_area = calculate_polygon_areas_km2(inundation_gdf)

    if not gdf_inundated_with_area.empty:
        newly_inundated_km2 = float(gdf_inundated_with_area["area_km2"].sum())
        num_polygons = len(gdf_inundated_with_area)
        mean_patch_km2 = float(gdf_inundated_with_area["area_km2"].mean())
        max_patch_km2 = float(gdf_inundated_with_area["area_km2"].max())
    else:
        # Fallback to pixel calculation if vectorization had no polygons
        inundated_pixels = np.sum((pre_water_mask == 0) & (post_water_mask == 1))
        newly_inundated_km2 = (inundated_pixels * pixel_area_m2) / 1_000_000.0
        num_polygons = 0
        mean_patch_km2 = 0.0
        max_patch_km2 = 0.0

    receded_pixels = np.sum((pre_water_mask == 1) & (post_water_mask == 0))
    receded_km2 = (receded_pixels * pixel_area_m2) / 1_000_000.0

    inundation_pct = (newly_inundated_km2 / total_scene_km2 * 100.0) if total_scene_km2 > 0 else 0.0

    # 3. Administrative boundary breakdown (if available)
    district_stats = None
    if admin_gdf is not None and not admin_gdf.empty and not gdf_inundated_with_area.empty:
        try:
            # Align CRS
            proj_crs = get_projected_crs(admin_gdf)
            admin_proj = admin_gdf.to_crs(proj_crs)
            inund_proj = gdf_inundated_with_area.to_crs(proj_crs)

            # Spatial intersection
            intersection = gpd.overlay(inund_proj, admin_proj, how="intersection")
            if not intersection.empty:
                intersection["sub_area_km2"] = intersection.geometry.area / 1_000_000.0

                # Identify district column name
                district_col = None
                for col in ["district", "name", "admin_name", "shapeName", "NAME_1", "NAME_2"]:
                    if col in intersection.columns:
                        district_col = col
                        break

                if district_col:
                    grouped = (
                        intersection.groupby(district_col)["sub_area_km2"]
                        .sum()
                        .reset_index()
                    )
                    district_stats = [
                        {
                            "district": str(row[district_col]),
                            "flooded_area_km2": round(float(row["sub_area_km2"]), 4),
                        }
                        for _, row in grouped.iterrows()
                    ]
        except Exception:
            district_stats = None

    proj_name = get_projected_crs(inundation_gdf) if not inundation_gdf.empty else metadata.crs

    return FloodImpactSummary(
        pre_water_area_km2=round(float(pre_area_km2), 4),
        post_water_area_km2=round(float(post_area_km2), 4),
        newly_inundated_area_km2=round(float(newly_inundated_km2), 4),
        receded_water_area_km2=round(float(receded_km2), 4),
        total_scene_area_km2=round(float(total_scene_km2), 4),
        inundation_percentage=round(float(inundation_pct), 2),
        num_flood_polygons=int(num_polygons),
        mean_flood_patch_km2=round(float(mean_patch_km2), 4),
        max_flood_patch_km2=round(float(max_patch_km2), 4),
        projection_used=proj_name,
        district_breakdown=district_stats,
    )
