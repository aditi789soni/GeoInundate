"""
Automated Flood Detection & Change Classification Module for GeoInundate.
Combines Otsu's adaptive histogram thresholding (OpenCV), morphological filtering,
connected-component noise reduction, and pre/post multi-temporal change detection.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from config import (
    CLASS_LAND,
    CLASS_PERMANENT_WATER,
    CLASS_NEW_INUNDATION,
    CLASS_RECEDED_WATER,
    CLASS_NODATA,
    MORPHOLOGY_KERNEL_SIZE,
    MORPHOLOGY_ITERATIONS,
    DEFAULT_MIN_POLYGON_PIXELS,
)


@dataclass
class FloodDetectionResult:
    """Encapsulates the complete flood classification output and diagnostic metrics."""
    pre_water_mask: np.ndarray        # Binary (0/1) pre-disaster water
    post_water_mask: np.ndarray       # Binary (0/1) post-disaster water
    change_mask: np.ndarray           # Multiclass: 0=Land, 1=Permanent, 2=Inundated, 3=Receded, 255=NoData
    inundation_mask: np.ndarray       # Binary (0/1) newly inundated areas only
    permanent_water_mask: np.ndarray  # Binary (0/1) permanent baseline water only
    pre_threshold_used: float         # Threshold applied to pre-disaster index
    post_threshold_used: float        # Threshold applied to post-disaster index
    pre_otsu_computed: float          # Otsu value calculated for pre-disaster
    post_otsu_computed: float         # Otsu value calculated for post-disaster
    pixel_counts: Dict[str, int]      # Pixel count per classification class


def compute_otsu_threshold(index_arr: np.ndarray) -> float:
    """
    Compute Otsu's optimal bimodal threshold on a continuous spectral index [-1.0, 1.0].
    Transforms float range to 8-bit unsigned integer (0-255) for OpenCV Otsu algorithm,
    then projects back to continuous index domain.
    """
    valid_mask = ~np.isnan(index_arr)
    valid_vals = index_arr[valid_mask]

    if valid_vals.size == 0:
        return 0.0

    # Scale valid values [-1.0, 1.0] to uint8 [0, 255]
    clipped = np.clip(valid_vals, -1.0, 1.0)
    scaled = np.round((clipped + 1.0) * 127.5).astype(np.uint8)

    # Calculate Otsu threshold via OpenCV
    otsu_val, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert mapping: uint8 threshold back to [-1.0, 1.0] float
    float_thresh = (otsu_val / 127.5) - 1.0
    return float(float_thresh)


def apply_morphological_cleaning(
    binary_mask: np.ndarray,
    kernel_size: int = MORPHOLOGY_KERNEL_SIZE,
    iterations: int = MORPHOLOGY_ITERATIONS,
) -> np.ndarray:
    """
    Apply mathematical morphology:
    1. Opening (erosion followed by dilation) to remove isolated salt noise.
    2. Closing (dilation followed by erosion) to bridge narrow gaps and fill holes in water bodies.
    """
    h, w = binary_mask.shape
    if h < kernel_size or w < kernel_size:
        return binary_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Opening removes speckle noise
    opened = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=iterations)
    # Closing seals internal holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def filter_connected_components(
    binary_mask: np.ndarray, min_size_pixels: int = DEFAULT_MIN_POLYGON_PIXELS
) -> np.ndarray:
    """
    Remove small disconnected patches below a minimum pixel count threshold.
    Effectively eliminates spurious single-pixel sensor noise.
    """
    if min_size_pixels <= 0 or binary_mask.size <= min_size_pixels:
        return binary_mask

    mask_u8 = (binary_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    filtered = np.zeros_like(mask_u8)
    # stats format: [x, y, width, height, area]
    for label_idx in range(1, num_labels):  # Skip background (0)
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_size_pixels:
            filtered[labels == label_idx] = 1

    return filtered


def classify_water_mask(
    index_arr: np.ndarray,
    threshold: Optional[float] = None,
    use_otsu: bool = True,
    apply_morphology: bool = True,
    min_size_pixels: int = DEFAULT_MIN_POLYGON_PIXELS,
) -> Tuple[np.ndarray, float, float]:
    """
    Segment water from land on a single spectral index array.
    Returns:
        (binary_water_mask [0/1], threshold_used, otsu_computed)
    """
    valid_mask = ~np.isnan(index_arr)
    otsu_val = compute_otsu_threshold(index_arr)

    # Determine threshold to apply
    if use_otsu or threshold is None:
        thresh = otsu_val
    else:
        thresh = float(threshold)

    # Initial thresholding: Water has positive or high index values
    water_binary = (index_arr > thresh) & valid_mask

    # Morphological cleaning & noise filtering
    if apply_morphology:
        water_binary = apply_morphological_cleaning(water_binary)

    if min_size_pixels > 0:
        water_binary = filter_connected_components(water_binary, min_size_pixels=min_size_pixels)

    return water_binary.astype(np.uint8), thresh, otsu_val


def detect_flood_change(
    pre_index: np.ndarray,
    post_index: np.ndarray,
    pre_threshold: Optional[float] = None,
    post_threshold: Optional[float] = None,
    use_otsu: bool = True,
    apply_morphology: bool = True,
    min_size_pixels: int = DEFAULT_MIN_POLYGON_PIXELS,
) -> FloodDetectionResult:
    """
    Perform multi-temporal flood change detection:
    - Pre-disaster water mask
    - Post-disaster water mask
    - Disaggregates permanent water, newly inundated land, and receded water.
    """
    # 1. Segment pre-disaster water
    pre_water, pre_thresh_used, pre_otsu = classify_water_mask(
        index_arr=pre_index,
        threshold=pre_threshold,
        use_otsu=use_otsu,
        apply_morphology=apply_morphology,
        min_size_pixels=min_size_pixels,
    )

    # 2. Segment post-disaster water
    post_water, post_thresh_used, post_otsu = classify_water_mask(
        index_arr=post_index,
        threshold=post_threshold,
        use_otsu=use_otsu,
        apply_morphology=apply_morphology,
        min_size_pixels=min_size_pixels,
    )

    # 3. Handle nodata / invalid pixels across both dates
    invalid_mask = np.isnan(pre_index) | np.isnan(post_index)

    # 4. Synthesize categorical change mask
    # 0 = Land, 1 = Permanent Water, 2 = Newly Inundated, 3 = Receded Water, 255 = NoData
    change_mask = np.full(pre_index.shape, CLASS_LAND, dtype=np.uint8)

    permanent = (pre_water == 1) & (post_water == 1) & (~invalid_mask)
    inundated = (pre_water == 0) & (post_water == 1) & (~invalid_mask)
    receded = (pre_water == 1) & (post_water == 0) & (~invalid_mask)

    change_mask[permanent] = CLASS_PERMANENT_WATER
    change_mask[inundated] = CLASS_NEW_INUNDATION
    change_mask[receded] = CLASS_RECEDED_WATER
    change_mask[invalid_mask] = CLASS_NODATA

    pixel_counts = {
        "land": int(np.sum(change_mask == CLASS_LAND)),
        "permanent_water": int(np.sum(change_mask == CLASS_PERMANENT_WATER)),
        "newly_inundated": int(np.sum(change_mask == CLASS_NEW_INUNDATION)),
        "receded_water": int(np.sum(change_mask == CLASS_RECEDED_WATER)),
        "nodata": int(np.sum(change_mask == CLASS_NODATA)),
    }

    return FloodDetectionResult(
        pre_water_mask=pre_water,
        post_water_mask=post_water,
        change_mask=change_mask,
        inundation_mask=inundated.astype(np.uint8),
        permanent_water_mask=permanent.astype(np.uint8),
        pre_threshold_used=pre_thresh_used,
        post_threshold_used=post_thresh_used,
        pre_otsu_computed=pre_otsu,
        post_otsu_computed=post_otsu,
        pixel_counts=pixel_counts,
    )
