"""
GeoInundate Test Suite.
Tests raster ingestion, spectral index computation, Otsu thresholding,
flood change detection, vectorization, area calculation, and report generation.
"""

from pathlib import Path
import numpy as np
import pytest
import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from processing.area_calculation import (
    calculate_flood_impact,
    calculate_polygon_areas_km2,
    get_projected_crs,
)
from processing.flood_detection import (
    classify_water_mask,
    compute_otsu_threshold,
    detect_flood_change,
)
from processing.raster_loader import (
    RasterMetadata,
    align_and_reproject,
    clean_reflectance,
    read_raster_from_source,
    resample_band,
    save_geotiff,
)
from processing.spectral_indices import (
    calculate_mndwi,
    calculate_ndwi,
    compute_spectral_index,
    index_to_rgba,
)
from processing.vectorization import (
    export_classification_raster,
    export_geojson,
    export_shapefile_zip,
    mask_to_geodataframe,
    vectorize_flood_results,
)
from visualization.report_generator import generate_markdown_report, generate_pdf_report


@pytest.fixture
def dummy_metadata():
    transform = from_origin(350000.0, 3100000.0, 10.0, 10.0)
    return RasterMetadata(
        width=100,
        height=100,
        count=3,
        crs="EPSG:32643",
        crs_epsg=32643,
        transform=transform,
        res_x=10.0,
        res_y=10.0,
        bounds_native=(350000.0, 3099000.0, 351000.0, 3100000.0),
        bounds_wgs84=(71.4, 28.0, 71.5, 28.1),
        nodata=0.0,
        dtype="uint16",
        driver="GTiff",
    )


# 1. RASTER LOADING & PREPROCESSING TESTS
def test_clean_reflectance():
    # Test DN scale factor 10,000 -> 0-1
    raw = np.array([0, 1000, 5000, 10000], dtype=np.uint16)
    cleaned = clean_reflectance(raw, nodata=0)
    assert np.isnan(cleaned[0])  # Nodata masked to NaN
    assert np.isclose(cleaned[1], 0.1, atol=1e-3)
    assert np.isclose(cleaned[2], 0.5, atol=1e-3)
    assert np.isclose(cleaned[3], 1.0, atol=1e-3)


def test_resample_band():
    src_data = np.ones((50, 50), dtype=np.float32) * 5.0
    src_transform = from_origin(0, 500, 10, 10)
    tgt_transform = from_origin(0, 500, 5, 5)
    target_shape = (100, 100)

    resampled = resample_band(
        data=src_data,
        src_transform=src_transform,
        target_shape=target_shape,
        target_transform=tgt_transform,
    )
    assert resampled.shape == (100, 100)
    assert np.isclose(np.mean(resampled), 5.0, atol=1e-2)


# 2. SPECTRAL INDEX TESTS
def test_calculate_mndwi():
    # Water feature: Green high relative to SWIR -> Positive MNDWI
    green = np.array([[0.10, 0.15], [0.10, 0.10]], dtype=np.float32)
    swir = np.array([[0.01, 0.02], [0.25, 0.10]], dtype=np.float32)

    mndwi = calculate_mndwi(green, swir)
    assert mndwi.shape == (2, 2)
    assert mndwi[0, 0] > 0.5   # Water pixel
    assert mndwi[1, 0] < 0.0   # Land / Soil pixel
    assert mndwi[1, 1] == 0.0  # Equal values


def test_mndwi_division_by_zero():
    # Denominator zero edge case
    green = np.zeros((3, 3), dtype=np.float32)
    swir = np.zeros((3, 3), dtype=np.float32)

    mndwi = calculate_mndwi(green, swir)
    assert np.all(np.isnan(mndwi))  # Must not crash, should be NaN


def test_calculate_ndwi():
    green = np.array([[0.12, 0.08]], dtype=np.float32)
    nir = np.array([[0.02, 0.45]], dtype=np.float32)

    ndwi = calculate_ndwi(green, nir)
    assert ndwi[0, 0] > 0.5  # Water
    assert ndwi[0, 1] < 0.0  # Dense vegetation


def test_index_to_rgba():
    idx = np.array([[-0.5, 0.0], [0.5, np.nan]], dtype=np.float32)
    rgba = index_to_rgba(idx, colormap_name="Blues")
    assert rgba.shape == (2, 2, 4)
    assert rgba[1, 1, 3] == 0  # NaN pixel transparent alpha


# 3. OTSU THRESHOLDING & FLOOD DETECTION TESTS
def test_compute_otsu_threshold():
    # Bimodal distribution: 50% land at -0.3, 50% water at +0.3
    np.random.seed(42)
    land = np.random.normal(-0.3, 0.05, 500)
    water = np.random.normal(0.3, 0.05, 500)
    data = np.concatenate([land, water]).reshape(50, 20)

    otsu = compute_otsu_threshold(data)
    assert -0.15 < otsu < 0.15  # Optimal threshold should lie between the two peaks


def test_classify_water_mask():
    idx = np.array([[-0.4, -0.2], [0.3, 0.5]], dtype=np.float32)
    mask, thresh_used, otsu_val = classify_water_mask(
        idx, threshold=0.0, use_otsu=False, apply_morphology=False, min_size_pixels=0
    )
    assert np.array_equal(mask, [[0, 0], [1, 1]])


def test_detect_flood_change():
    # Pre-flood: River only in column 0
    pre_idx = np.array([[0.5, -0.3], [0.5, -0.3]], dtype=np.float32)
    # Post-flood: River expanded to column 1 (new inundation)
    post_idx = np.array([[0.5, 0.4], [0.5, 0.4]], dtype=np.float32)

    res = detect_flood_change(
        pre_idx, post_idx, pre_threshold=0.0, post_threshold=0.0, use_otsu=False,
        apply_morphology=False, min_size_pixels=0
    )

    # Column 0: Permanent Water (1)
    # Column 1: Newly Inundated (2)
    assert np.all(res.permanent_water_mask[:, 0] == 1)
    assert np.all(res.inundation_mask[:, 1] == 1)
    assert res.pixel_counts["permanent_water"] == 2
    assert res.pixel_counts["newly_inundated"] == 2


# 4. VECTORIZATION & AREA CALCULATION TESTS
def test_mask_to_geodataframe(dummy_metadata):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 20:50] = 1  # 30x30 pixels square

    gdf = mask_to_geodataframe(mask, dummy_metadata, category_label="Inundated")
    assert not gdf.empty
    assert len(gdf) == 1
    assert gdf.iloc[0]["class"] == "Inundated"
    assert gdf.crs == dummy_metadata.crs


def test_calculate_polygon_areas_km2(dummy_metadata):
    mask = np.zeros((100, 100), dtype=np.uint8)
    # 50x50 pixels at 10m res = 500m x 500m = 250,000 m² = 0.25 km²
    mask[10:60, 10:60] = 1

    gdf = mask_to_geodataframe(mask, dummy_metadata, category_label="Inundated")
    gdf_areas = calculate_polygon_areas_km2(gdf)

    assert "area_km2" in gdf_areas.columns
    computed_km2 = gdf_areas.iloc[0]["area_km2"]
    assert np.isclose(computed_km2, 0.25, atol=0.01)


def test_get_projected_crs():
    # Test geographic CRS auto-reprojection to UTM
    poly = Polygon([(73.0, 31.0), (73.1, 31.0), (73.1, 31.1), (73.0, 31.1)])
    gdf_geo = gpd.GeoDataFrame([{"geometry": poly}], crs="EPSG:4326")

    proj_crs = get_projected_crs(gdf_geo)
    assert "EPSG:32643" in proj_crs or "326" in proj_crs


def test_calculate_flood_impact(dummy_metadata):
    pre_mask = np.zeros((100, 100), dtype=np.uint8)
    post_mask = np.zeros((100, 100), dtype=np.uint8)
    pre_mask[10:30, 10:30] = 1   # Permanent 20x20 px
    post_mask[10:50, 10:50] = 1  # Expanded to 40x40 px

    inund_mask = (pre_mask == 0) & (post_mask == 1)
    perm_mask = (pre_mask == 1) & (post_mask == 1)

    inund_gdf = mask_to_geodataframe(inund_mask.astype(np.uint8), dummy_metadata)
    perm_gdf = mask_to_geodataframe(perm_mask.astype(np.uint8), dummy_metadata)

    summary = calculate_flood_impact(
        inundation_gdf=inund_gdf,
        permanent_gdf=perm_gdf,
        pre_water_mask=pre_mask,
        post_water_mask=post_mask,
        metadata=dummy_metadata,
    )

    assert summary.pre_water_area_km2 > 0
    assert summary.newly_inundated_area_km2 > 0
    assert summary.total_scene_area_km2 == 1.0  # 1000m x 1000m = 1 km²


# 5. EXPORT & REPORT TESTS
def test_export_reports(tmp_path, dummy_metadata):
    pre_mask = np.zeros((50, 50), dtype=np.uint8)
    post_mask = np.ones((50, 50), dtype=np.uint8)
    inund_gdf = mask_to_geodataframe(post_mask, dummy_metadata)
    perm_gdf = mask_to_geodataframe(pre_mask, dummy_metadata)

    summary = calculate_flood_impact(
        inundation_gdf=inund_gdf,
        permanent_gdf=perm_gdf,
        pre_water_mask=pre_mask,
        post_water_mask=post_mask,
        metadata=dummy_metadata,
    )

    # Test PDF Generation
    pdf_bytes = generate_pdf_report(
        event_name="Test Flood Event",
        impact=summary,
        metadata=dummy_metadata,
        index_name="MNDWI",
        method_name="Otsu Adaptive",
    )
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

    # Test Markdown Generation
    md = generate_markdown_report(
        event_name="Test Flood Event",
        impact=summary,
        metadata=dummy_metadata,
        index_name="MNDWI",
        method_name="Otsu Adaptive",
    )
    assert "# GeoInundate" in md
    assert "Newly Inundated Area" in md

    # Test GeoJSON Export
    geojson_path = tmp_path / "test.geojson"
    export_geojson(inund_gdf, geojson_path)
    assert geojson_path.exists()

    # Test Shapefile ZIP Export
    zip_path = tmp_path / "test_shp.zip"
    export_shapefile_zip(inund_gdf, zip_path)
    assert zip_path.exists()
