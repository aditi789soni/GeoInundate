"""
Raster Ingestion & Preprocessing Module for GeoInundate.
Handles GeoTIFF loading, CRS/resolution validation, band alignment,
reprojection, and sample dataset provision.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, transform_bounds


@dataclass
class RasterMetadata:
    """Encapsulates spatial and radiometric metadata of an ingested raster."""
    width: int
    height: int
    count: int
    crs: str
    crs_epsg: Optional[int]
    transform: Affine
    res_x: float
    res_y: float
    bounds_native: Tuple[float, float, float, float]
    bounds_wgs84: Tuple[float, float, float, float]
    nodata: Optional[float]
    dtype: str
    driver: str

    def to_dict(self) -> Dict[str, Union[str, int, float, None, Tuple]]:
        return {
            "Width (px)": self.width,
            "Height (px)": self.height,
            "Bands": self.count,
            "CRS": self.crs,
            "EPSG Code": self.crs_epsg,
            "Resolution X (m/deg)": round(self.res_x, 4),
            "Resolution Y (m/deg)": round(self.res_y, 4),
            "Native Bounds": [round(b, 2) for b in self.bounds_native],
            "WGS84 Bounds (Lat/Lon)": [round(b, 5) for b in self.bounds_wgs84],
            "NoData Value": self.nodata,
            "Data Type": self.dtype,
            "Driver": self.driver,
        }


def extract_metadata(src: rasterio.DatasetReader) -> RasterMetadata:
    """Extract standard spatial and radiometric metadata from a rasterio reader."""
    bounds = src.bounds
    native_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    crs_str = str(src.crs) if src.crs else "Unknown"
    epsg = src.crs.to_epsg() if src.crs else None

    # Calculate WGS84 bounding box (min_lon, min_lat, max_lon, max_lat)
    if src.crs and src.crs.is_valid:
        try:
            wgs84_bounds = transform_bounds(
                src.crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top
            )
        except Exception:
            wgs84_bounds = native_bounds
    else:
        wgs84_bounds = native_bounds

    res_x, res_y = abs(src.res[0]), abs(src.res[1])

    return RasterMetadata(
        width=src.width,
        height=src.height,
        count=src.count,
        crs=crs_str,
        crs_epsg=epsg,
        transform=src.transform,
        res_x=res_x,
        res_y=res_y,
        bounds_native=native_bounds,
        bounds_wgs84=wgs84_bounds,
        nodata=src.nodata,
        dtype=str(src.dtypes[0]),
        driver=src.driver,
    )


def read_raster_from_source(
    source: Union[str, Path, BytesIO, bytes]
) -> Tuple[np.ndarray, RasterMetadata]:
    """
    Open and read a raster from file path, BytesIO buffer, or raw bytes.
    Returns:
        (band_array [count, height, width], RasterMetadata)
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Raster file not found: {path}")
        with rasterio.open(path) as src:
            meta = extract_metadata(src)
            data = src.read()
            return data, meta

    elif isinstance(source, (BytesIO, bytes)):
        raw_bytes = source.getvalue() if isinstance(source, BytesIO) else source
        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as src:
                meta = extract_metadata(src)
                data = src.read()
                return data, meta
    else:
        raise TypeError(f"Unsupported raster source type: {type(source)}")


def clean_reflectance(
    band_array: np.ndarray, nodata: Optional[float] = None
) -> np.ndarray:
    """
    Convert raw band values to normalized reflectance (0.0 to 1.0) and handle nodata/NaNs.
    Sentinel-2 L2A BOA data is typically scaled by 10,000 (e.g. 0 to 10,000 DN).
    """
    arr = band_array.astype(np.float32)

    # Mask native nodata or negative reflectance
    if nodata is not None and not np.isnan(nodata):
        arr[arr == nodata] = np.nan
    arr[arr < 0] = np.nan

    # Auto-detect Sentinel-2 10,000 scale factor vs 0-1 float
    valid_mask = ~np.isnan(arr)
    if np.any(valid_mask):
        max_val = np.nanmax(arr[valid_mask])
        if max_val > 1.5:  # Scaled integer DN (e.g., 0-10000)
            arr = arr / 10000.0

    return arr


def resample_band(
    data: np.ndarray,
    src_transform: Affine,
    target_shape: Tuple[int, int],
    target_transform: Affine,
    resampling_method: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Resample a 2D raster band array (e.g., 20m SWIR) to match a target grid shape (e.g., 10m Green).
    """
    src_h, src_w = data.shape
    tgt_h, tgt_w = target_shape

    if (src_h, src_w) == (tgt_h, tgt_w) and src_transform == target_transform:
        return data.copy()

    destination = np.empty((tgt_h, tgt_w), dtype=data.dtype)
    reproject(
        source=data,
        destination=destination,
        src_transform=src_transform,
        src_crs="EPSG:4326",  # Placeholder when aligning pixel coordinates
        dst_transform=target_transform,
        dst_crs="EPSG:4326",
        resampling=resampling_method,
    )
    return destination


def align_and_reproject(
    src_data: np.ndarray,
    src_meta: RasterMetadata,
    ref_meta: RasterMetadata,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, RasterMetadata]:
    """
    Reproject and align a source raster dataset to match a reference raster's
    grid dimensions, CRS, transform, and bounding box.
    """
    # If already identical in CRS and dimensions, return directly
    if (
        src_meta.crs == ref_meta.crs
        and (src_meta.height, src_meta.width) == (ref_meta.height, ref_meta.width)
        and src_meta.transform == ref_meta.transform
    ):
        return src_data.copy(), src_meta

    num_bands = src_data.shape[0] if src_data.ndim == 3 else 1
    src_crs = rasterio.crs.CRS.from_string(src_meta.crs)
    ref_crs = rasterio.crs.CRS.from_string(ref_meta.crs)

    dest_data = np.empty((num_bands, ref_meta.height, ref_meta.width), dtype=np.float32)

    for b in range(num_bands):
        band = src_data[b] if src_data.ndim == 3 else src_data
        reproject(
            source=band,
            destination=dest_data[b],
            src_transform=src_meta.transform,
            src_crs=src_crs,
            dst_transform=ref_meta.transform,
            dst_crs=ref_crs,
            resampling=resampling,
        )

    # Return aligned data and adjusted metadata matching reference grid
    new_meta = RasterMetadata(
        width=ref_meta.width,
        height=ref_meta.height,
        count=num_bands,
        crs=ref_meta.crs,
        crs_epsg=ref_meta.crs_epsg,
        transform=ref_meta.transform,
        res_x=ref_meta.res_x,
        res_y=ref_meta.res_y,
        bounds_native=ref_meta.bounds_native,
        bounds_wgs84=ref_meta.bounds_wgs84,
        nodata=ref_meta.nodata,
        dtype=str(dest_data.dtype),
        driver=ref_meta.driver,
    )
    return dest_data, new_meta


def save_geotiff(
    data: np.ndarray,
    metadata: RasterMetadata,
    output_path: Union[str, Path],
    dtype: Optional[str] = None,
    nodata: Optional[float] = None,
) -> Path:
    """Save a 2D or 3D numpy array as a georeferenced GeoTIFF."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        count = 1
        height, width = data.shape
        data_to_write = data[np.newaxis, :, :]
    else:
        count, height, width = data.shape
        data_to_write = data

    out_dtype = dtype or str(data.dtype)
    crs = rasterio.crs.CRS.from_string(metadata.crs)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": out_dtype,
        "crs": crs,
        "transform": metadata.transform,
        "nodata": nodata if nodata is not None else metadata.nodata,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(count):
            dst.write(data_to_write[i].astype(out_dtype), i + 1)

    return out_path
