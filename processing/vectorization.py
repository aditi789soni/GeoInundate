"""
Geospatial Vectorization Module for GeoInundate.
Converts classified raster masks to clean vector polygons (GeoPandas / Shapely)
and packages outputs into GeoJSON and Shapefile ZIP archives.
"""

from pathlib import Path
import shutil
import tempfile
from typing import Optional, Union
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.validation import make_valid

from processing.raster_loader import RasterMetadata, save_geotiff


def mask_to_geodataframe(
    binary_mask: np.ndarray,
    metadata: RasterMetadata,
    category_label: str = "Inundated",
    min_area_m2: float = 100.0,
) -> gpd.GeoDataFrame:
    """
    Vectorize a binary raster mask into a GeoPandas GeoDataFrame with cleaned geometries.
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8)

    # rasterio.features.shapes yields (geojson_dict_geometry, value)
    feature_shapes = shapes(
        mask_u8,
        mask=(mask_u8 == 1),
        transform=metadata.transform,
    )

    records = []
    for geom_dict, val in feature_shapes:
        if val == 1:
            geom = shape(geom_dict)
            if not geom.is_valid:
                geom = make_valid(geom)
            if not geom.is_empty:
                records.append({
                    "geometry": geom,
                    "class": category_label,
                    "pixel_val": int(val),
                })

    if not records:
        # Return empty GeoDataFrame with correct CRS
        return gpd.GeoDataFrame(
            columns=["geometry", "class", "pixel_val"],
            crs=metadata.crs,
            geometry="geometry",
        )

    gdf = gpd.GeoDataFrame(records, crs=metadata.crs, geometry="geometry")

    # Buffer(0) cleanup for any remaining topological imperfections
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: g.buffer(0) if not g.is_valid else g
    )
    gdf = gdf[~gdf.is_empty & gdf.geometry.notnull()].reset_index(drop=True)

    return gdf


def vectorize_flood_results(
    inundation_mask: np.ndarray,
    permanent_water_mask: np.ndarray,
    metadata: RasterMetadata,
) -> dict:
    """
    Generate separate and combined vector layers for:
    - Newly Inundated Flood polygons
    - Permanent Water polygons
    """
    gdf_inundated = mask_to_geodataframe(
        inundation_mask, metadata, category_label="Newly Inundated"
    )
    gdf_permanent = mask_to_geodataframe(
        permanent_water_mask, metadata, category_label="Permanent Water"
    )

    # Combine into unified change GeoDataFrame
    combined = []
    if not gdf_inundated.empty:
        combined.append(gdf_inundated)
    if not gdf_permanent.empty:
        combined.append(gdf_permanent)

    if combined:
        gdf_all = gpd.GeoDataFrame(
            gpd.pd.concat(combined, ignore_index=True),
            crs=metadata.crs,
        )
    else:
        gdf_all = gpd.GeoDataFrame(
            columns=["geometry", "class", "pixel_val"],
            crs=metadata.crs,
            geometry="geometry",
        )

    return {
        "inundation_gdf": gdf_inundated,
        "permanent_gdf": gdf_permanent,
        "all_water_gdf": gdf_all,
    }


def export_geojson(gdf: gpd.GeoDataFrame, output_path: Union[str, Path]) -> Path:
    """Export GeoDataFrame as a standard WGS84 GeoJSON file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Ensure output is in WGS84 (EPSG:4326) as per GeoJSON specification
    gdf_wgs84 = gdf.to_crs("EPSG:4326") if gdf.crs and str(gdf.crs) != "EPSG:4326" else gdf
    gdf_wgs84.to_file(out_file, driver="GeoJSON")
    return out_file


def export_shapefile_zip(gdf: gpd.GeoDataFrame, zip_path: Union[str, Path]) -> Path:
    """
    Export GeoDataFrame to ESRI Shapefile and bundle into a self-contained ZIP archive.
    """
    out_zip = Path(zip_path)
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_shp_path = Path(tmp_dir) / "flood_polygons.shp"
        # Truncate column names if necessary for Shapefile DBF limits
        gdf_copy = gdf.copy()
        gdf_copy.to_file(tmp_shp_path, driver="ESRI Shapefile")

        # Zip all shapefile companion components (.shp, .shx, .dbf, .prj, .cpg)
        base_zip = out_zip.with_suffix("")
        shutil.make_archive(
            base_name=str(base_zip),
            format="zip",
            root_dir=tmp_dir,
        )

    return out_zip


def export_classification_raster(
    change_mask: np.ndarray,
    metadata: RasterMetadata,
    output_path: Union[str, Path],
) -> Path:
    """Save categorical flood change classification as a georeferenced GeoTIFF."""
    return save_geotiff(
        data=change_mask,
        metadata=metadata,
        output_path=output_path,
        dtype="uint8",
        nodata=255,
    )
