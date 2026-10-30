"""
GeoInundate Configuration Module
Centralized parameters for satellite processing, flood classification,
geospatial styling, and application defaults.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure runtime directories exist
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# App Metadata
APP_NAME = "GeoInundate"
APP_SUBTITLE = "Automated Satellite Water Inundation & Disaster Change Detection System"
APP_VERSION = "1.0.0"

# Sentinel-2 Band Mappings & Wavelengths
BAND_CONFIG = {
    "SENTINEL_2": {
        "B03": {"name": "Green", "wavelength_nm": 560, "resolution_m": 10},
        "B08": {"name": "NIR", "wavelength_nm": 842, "resolution_m": 10},
        "B11": {"name": "SWIR1", "wavelength_nm": 1610, "resolution_m": 20},
    }
}

# Spectral Index Formulae & References
INDICES = {
    "MNDWI": {
        "name": "Modified Normalized Difference Water Index",
        "reference": "Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery.",
        "bands": ("Green", "SWIR"),
        "formula": "(Green - SWIR) / (Green + SWIR)",
        "description": "Recommended for high-contrast water extraction. Suppresses built-up and bare soil features effectively.",
    },
    "NDWI": {
        "name": "Normalized Difference Water Index",
        "reference": "McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features.",
        "bands": ("Green", "NIR"),
        "formula": "(Green - NIR) / (Green + NIR)",
        "description": "Classic water index. Highly responsive to open water bodies, though can occasionally mistake built-up shadows.",
    },
}

# Flood Detection & Processing Defaults
DEFAULT_INDEX = "MNDWI"
DEFAULT_USE_OTSU = True
DEFAULT_MANUAL_THRESHOLD = 0.05
DEFAULT_MIN_POLYGON_PIXELS = 10  # Minimum connected components to filter noise
MORPHOLOGY_KERNEL_SIZE = 3
MORPHOLOGY_ITERATIONS = 1

# Classification Pixel Values
CLASS_LAND = 0
CLASS_PERMANENT_WATER = 1
CLASS_NEW_INUNDATION = 2
CLASS_RECEDED_WATER = 3
CLASS_NODATA = 255

# Map & Visualization Color Palette (Hex)
COLORS = {
    "land": "#737373",              # Slate Gray
    "permanent_water": "#1D4ED8",    # Deep Blue
    "new_inundation": "#DC2626",     # Vivid Red
    "receded_water": "#F59E0B",      # Amber / Orange
    "mndwi_water": "#06B6D4",        # Cyan
    "basemap_tint": "#0F172A",
}

# Administrative GeoJSON default attributes
ADMIN_COLUMNS = ["district", "name", "admin_name", "shapeName", "NAME_1", "NAME_2"]
