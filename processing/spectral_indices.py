"""
Spectral Index Calculation Module for GeoInundate.
Implements MNDWI (Xu, 2006) and NDWI (McFeeters, 1996) with robust division-by-zero
protection, spatial resolution harmonization, and colormap visualization utilities.
"""

from typing import Optional, Tuple
import matplotlib.cm as cm
import numpy as np
from rasterio.transform import Affine

from processing.raster_loader import clean_reflectance, resample_band


def calculate_mndwi(
    green: np.ndarray,
    swir: np.ndarray,
    green_transform: Optional[Affine] = None,
    swir_transform: Optional[Affine] = None,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate Modified Normalized Difference Water Index (MNDWI):
    MNDWI = (Green - SWIR) / (Green + SWIR)

    Reference:
        Xu, H. (2006). Modification of normalised difference water index (NDWI)
        to enhance open water features in remotely sensed imagery.
        International Journal of Remote Sensing, 27(14), 3025-3033.

    Args:
        green: Sentinel-2 Band 3 (Green, ~560 nm).
        swir: Sentinel-2 Band 11 (SWIR1, ~1610 nm).
        green_transform: Optional affine transform of the green band.
        swir_transform: Optional affine transform of the SWIR band.
        nodata: Optional nodata value to mask.

    Returns:
        2D float32 numpy array with MNDWI values in range [-1.0, 1.0].
    """
    # Harmonize spatial resolution if shapes differ (e.g., SWIR at 20m vs Green at 10m)
    if green.shape != swir.shape and green_transform is not None and swir_transform is not None:
        swir = resample_band(
            data=swir,
            src_transform=swir_transform,
            target_shape=green.shape,
            target_transform=green_transform,
        )

    # Clean and normalize reflectance
    g = clean_reflectance(green, nodata=nodata)
    s = clean_reflectance(swir, nodata=nodata)

    # Calculate index with division-by-zero protection
    numerator = g - s
    denominator = g + s

    # Invalid mask: NaN or denominator near zero
    invalid_mask = np.isnan(g) | np.isnan(s) | (np.abs(denominator) < 1e-7)

    mndwi = np.zeros_like(numerator, dtype=np.float32)
    # Perform division safely on valid entries
    np.divide(numerator, denominator, out=mndwi, where=~invalid_mask)

    # Enforce theoretical bounds [-1.0, 1.0]
    mndwi = np.clip(mndwi, -1.0, 1.0)
    mndwi[invalid_mask] = np.nan

    return mndwi


def calculate_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    green_transform: Optional[Affine] = None,
    nir_transform: Optional[Affine] = None,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate McFeeters Normalized Difference Water Index (NDWI):
    NDWI = (Green - NIR) / (Green + NIR)

    Reference:
        McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI)
        in the delineation of open water features.
        International Journal of Remote Sensing, 17(7), 1425-1432.
    """
    if green.shape != nir.shape and green_transform is not None and nir_transform is not None:
        nir = resample_band(
            data=nir,
            src_transform=nir_transform,
            target_shape=green.shape,
            target_transform=green_transform,
        )

    g = clean_reflectance(green, nodata=nodata)
    n = clean_reflectance(nir, nodata=nodata)

    numerator = g - n
    denominator = g + n

    invalid_mask = np.isnan(g) | np.isnan(n) | (np.abs(denominator) < 1e-7)

    ndwi = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=ndwi, where=~invalid_mask)

    ndwi = np.clip(ndwi, -1.0, 1.0)
    ndwi[invalid_mask] = np.nan

    return ndwi


def compute_spectral_index(
    index_name: str,
    bands: dict,
    green_transform: Optional[Affine] = None,
    other_transform: Optional[Affine] = None,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """
    Unified router for spectral water index calculations.
    bands dictionary expected to contain 'green' and either 'swir' (for MNDWI) or 'nir' (for NDWI).
    """
    name = index_name.upper()
    if name == "MNDWI":
        if "green" not in bands or "swir" not in bands:
            raise KeyError("MNDWI calculation requires both 'green' and 'swir' bands.")
        return calculate_mndwi(
            green=bands["green"],
            swir=bands["swir"],
            green_transform=green_transform,
            swir_transform=other_transform,
            nodata=nodata,
        )
    elif name == "NDWI":
        if "green" not in bands or "nir" not in bands:
            raise KeyError("NDWI calculation requires both 'green' and 'nir' bands.")
        return calculate_ndwi(
            green=bands["green"],
            nir=bands["nir"],
            green_transform=green_transform,
            nir_transform=other_transform,
            nodata=nodata,
        )
    else:
        raise ValueError(f"Unsupported spectral index '{index_name}'. Choose 'MNDWI' or 'NDWI'.")


def index_to_rgba(
    index_arr: np.ndarray,
    colormap_name: str = "Blues",
    vmin: float = -0.5,
    vmax: float = 0.5,
) -> np.ndarray:
    """
    Render a float spectral index array into an RGBA image array (uint8, shape [H, W, 4])
    suitable for UI display and Leaflet/Folium image overlay.
    """
    valid = ~np.isnan(index_arr)
    # Normalize range [vmin, vmax] to [0.0, 1.0]
    norm_val = np.clip((index_arr - vmin) / (vmax - vmin), 0.0, 1.0)

    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = cm.get_cmap(colormap_name)
    rgba = cmap(norm_val)  # float 0-1 RGBA
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    # Set invalid / nodata pixels to completely transparent
    rgba_uint8[~valid, 3] = 0

    return rgba_uint8
