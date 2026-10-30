"""
Interactive Geospatial Map Module for GeoInundate.
Leverages Folium and Leaflet to provide multi-layered disaster visualizations,
spectral index overlays, vector boundaries, layer toggles, and legends.
"""

from typing import Optional
import folium
from folium import plugins
import geopandas as gpd
import numpy as np

from config import (
    COLORS,
    CLASS_PERMANENT_WATER,
    CLASS_NEW_INUNDATION,
    CLASS_RECEDED_WATER,
)
from processing.raster_loader import RasterMetadata
from processing.spectral_indices import index_to_rgba


def create_change_rgba(change_mask: np.ndarray) -> np.ndarray:
    """
    Create an RGBA image array for Folium overlay:
    - Permanent water: Blue (#1D4ED8, alpha 200)
    - Newly Inundated: Red (#DC2626, alpha 220)
    - Receded water: Amber (#F59E0B, alpha 180)
    - All other pixels: Fully transparent
    """
    h, w = change_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Permanent Water (Blue)
    perm = change_mask == CLASS_PERMANENT_WATER
    rgba[perm] = [29, 78, 216, 200]

    # Newly Inundated (Red)
    inund = change_mask == CLASS_NEW_INUNDATION
    rgba[inund] = [220, 38, 38, 230]

    # Receded Water (Orange/Amber)
    reced = change_mask == CLASS_RECEDED_WATER
    rgba[reced] = [245, 158, 11, 190]

    return rgba


def build_interactive_map(
    metadata: RasterMetadata,
    change_mask: Optional[np.ndarray] = None,
    pre_index: Optional[np.ndarray] = None,
    post_index: Optional[np.ndarray] = None,
    inundation_gdf: Optional[gpd.GeoDataFrame] = None,
    permanent_gdf: Optional[gpd.GeoDataFrame] = None,
    admin_gdf: Optional[gpd.GeoDataFrame] = None,
    basemap_choice: str = "CartoDB dark_matter",
) -> folium.Map:
    """
    Construct a full-featured interactive Folium map centered on the ingested raster.
    Includes raster overlays, vector polygon overlays, boundary outlines, and map legend.
    """
    # Calculate map center from WGS84 bounds: (min_lon, min_lat, max_lon, max_lat)
    min_lon, min_lat, max_lon, max_lat = metadata.bounds_wgs84
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    # Bounds formatted as [[min_lat, min_lon], [max_lat, max_lon]] for Folium ImageOverlay
    img_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    # Initialize map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=basemap_choice,
        control_scale=True,
    )

    # Add alternative basemaps
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite Imagery",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(m)

    # 1. Spectral Index Overlays
    if pre_index is not None:
        pre_rgba = index_to_rgba(pre_index, colormap_name="Blues", vmin=-0.5, vmax=0.5)
        folium.raster_layers.ImageOverlay(
            image=pre_rgba,
            bounds=img_bounds,
            name="Pre-Disaster Water Index (MNDWI)",
            opacity=0.75,
            interactive=True,
            show=False,
        ).add_to(m)

    if post_index is not None:
        post_rgba = index_to_rgba(post_index, colormap_name="Blues", vmin=-0.5, vmax=0.5)
        folium.raster_layers.ImageOverlay(
            image=post_rgba,
            bounds=img_bounds,
            name="Post-Disaster Water Index (MNDWI)",
            opacity=0.75,
            interactive=True,
            show=False,
        ).add_to(m)

    # 2. Categorical Flood Change Raster Overlay
    if change_mask is not None:
        change_rgba = create_change_rgba(change_mask)
        folium.raster_layers.ImageOverlay(
            image=change_rgba,
            bounds=img_bounds,
            name="Flood Classification Raster (Red=Inundation, Blue=Permanent)",
            opacity=0.85,
            interactive=True,
            show=True,
        ).add_to(m)

    # 3. Vector Polygon Layers
    if permanent_gdf is not None and not permanent_gdf.empty:
        perm_wgs84 = permanent_gdf.to_crs("EPSG:4326")
        folium.GeoJson(
            perm_wgs84,
            name="Permanent Water Polygons",
            style_function=lambda feature: {
                "fillColor": COLORS["permanent_water"],
                "color": "#1E40AF",
                "weight": 1.5,
                "fillOpacity": 0.5,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["class"],
                aliases=["Water Type:"],
                localize=True,
            ),
            show=False,
        ).add_to(m)

    if inundation_gdf is not None and not inundation_gdf.empty:
        inund_wgs84 = inundation_gdf.to_crs("EPSG:4326")
        # Include area if available
        fields = ["class"]
        aliases = ["Status:"]
        if "area_km2" in inund_wgs84.columns:
            inund_wgs84["area_km2_display"] = inund_wgs84["area_km2"].round(4)
            fields.append("area_km2_display")
            aliases.append("Patch Area (km²):")

        folium.GeoJson(
            inund_wgs84,
            name="Newly Inundated Polygons (Vector)",
            style_function=lambda feature: {
                "fillColor": COLORS["new_inundation"],
                "color": "#991B1B",
                "weight": 2.0,
                "fillOpacity": 0.65,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=fields,
                aliases=aliases,
                localize=True,
            ),
            show=True,
        ).add_to(m)

    # 4. Administrative Boundary Overlay
    if admin_gdf is not None and not admin_gdf.empty:
        admin_wgs84 = admin_gdf.to_crs("EPSG:4326")
        district_col = None
        for col in ["district", "name", "admin_name", "shapeName", "NAME_1", "NAME_2"]:
            if col in admin_wgs84.columns:
                district_col = col
                break

        fields = [district_col] if district_col else []
        aliases = ["District:"] if district_col else []

        folium.GeoJson(
            admin_wgs84,
            name="Administrative Boundaries",
            style_function=lambda feature: {
                "fillColor": "none",
                "color": "#FBBF24",
                "weight": 2.5,
                "dashArray": "5, 5",
            },
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases) if fields else None,
            show=True,
        ).add_to(m)

    # 5. Fullscreen control & Measure tool
    plugins.Fullscreen(position="topright").add_to(m)
    plugins.MeasureControl(position="bottomleft", primary_length_unit="kilometers").add_to(m)

    # 6. Layer Control
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # 7. Professional HTML Legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        right: 25px;
        z-index: 1000;
        background-color: rgba(15, 23, 42, 0.90);
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 13px;
        line-height: 1.6;
    ">
        <b style="color: #38BDF8; font-size: 14px;">🌊 Flood Inundation Legend</b><br>
        <div style="margin-top: 6px;">
            <span style="display:inline-block; width:14px; height:14px; background:{COLORS['new_inundation']}; border-radius:3px; margin-right:8px; vertical-align:middle;"></span>
            <b>Newly Inundated (Flood)</b><br>
            <span style="display:inline-block; width:14px; height:14px; background:{COLORS['permanent_water']}; border-radius:3px; margin-right:8px; vertical-align:middle;"></span>
            Permanent Water Body<br>
            <span style="display:inline-block; width:14px; height:14px; background:{COLORS['receded_water']}; border-radius:3px; margin-right:8px; vertical-align:middle;"></span>
            Receded Water<br>
            <span style="display:inline-block; width:14px; height:4px; background:#FBBF24; margin-right:8px; vertical-align:middle;"></span>
            Admin Boundary
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m
