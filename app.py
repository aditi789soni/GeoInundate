"""
GeoInundate – Automated Satellite Water Inundation & Disaster Change Detection System
Main Streamlit Application Dashboard.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import geopandas as gpd

from config import (
    APP_NAME,
    APP_SUBTITLE,
    APP_VERSION,
    BAND_CONFIG,
    COLORS,
    DEFAULT_INDEX,
    DEFAULT_MANUAL_THRESHOLD,
    DEFAULT_MIN_POLYGON_PIXELS,
    DEFAULT_USE_OTSU,
    INDICES,
    SAMPLE_DIR,
    OUTPUTS_DIR,
    ASSETS_DIR,
)
from processing.area_calculation import calculate_flood_impact
from processing.flood_detection import detect_flood_change
from processing.raster_loader import (
    align_and_reproject,
    clean_reflectance,
    read_raster_from_source,
    save_geotiff,
)
from processing.sample_generator import generate_sample_flood_data
from processing.spectral_indices import compute_spectral_index, index_to_rgba
from processing.vectorization import (
    export_classification_raster,
    export_geojson,
    export_shapefile_zip,
    vectorize_flood_results,
)
from visualization.charts import (
    create_area_comparison_chart,
    create_district_impact_chart,
    create_index_histogram,
    create_land_cover_donut,
)
from visualization.map_view import build_interactive_map
from visualization.report_generator import generate_markdown_report, generate_pdf_report

# Page Configuration
st.set_page_config(
    page_title=f"{APP_NAME} | Satellite Flood Inundation Pipeline",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS Styling
css_file = ASSETS_DIR / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state():
    """Initialize persistent pipeline session state variables."""
    defaults = {
        "pre_data": None,
        "pre_meta": None,
        "post_data": None,
        "post_meta": None,
        "pre_index": None,
        "post_index": None,
        "index_choice": DEFAULT_INDEX,
        "flood_result": None,
        "vector_results": None,
        "impact_summary": None,
        "admin_gdf": None,
        "use_otsu": DEFAULT_USE_OTSU,
        "manual_threshold": DEFAULT_MANUAL_THRESHOLD,
        "min_patch_pixels": DEFAULT_MIN_POLYGON_PIXELS,
        "event_name": "Sentinel-2 River Basin Flood Event",
        "pipeline_executed": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


def run_pipeline():
    """Execute the end-to-end inundation and change detection pipeline."""
    if st.session_state["pre_data"] is None or st.session_state["post_data"] is None:
        st.error("Please load or upload both pre-disaster and post-disaster satellite images.")
        return

    with st.spinner("Aligning and reprojecting multi-temporal rasters..."):
        # Align post raster to pre raster grid if resolutions or CRS differ
        post_aligned, post_meta_aligned = align_and_reproject(
            src_data=st.session_state["post_data"],
            src_meta=st.session_state["post_meta"],
            ref_meta=st.session_state["pre_meta"],
        )
        st.session_state["post_data"] = post_aligned
        st.session_state["post_meta"] = post_meta_aligned

    with st.spinner(f"Computing {st.session_state['index_choice']} spectral water index..."):
        # Map bands: Band 1=Green, Band 2=NIR, Band 3=SWIR (default 3-band Sentinel-2)
        pre_bands = {
            "green": st.session_state["pre_data"][0],
            "nir": st.session_state["pre_data"][1] if st.session_state["pre_data"].shape[0] > 1 else st.session_state["pre_data"][0],
            "swir": st.session_state["pre_data"][2] if st.session_state["pre_data"].shape[0] > 2 else st.session_state["pre_data"][0],
        }
        post_bands = {
            "green": st.session_state["post_data"][0],
            "nir": st.session_state["post_data"][1] if st.session_state["post_data"].shape[0] > 1 else st.session_state["post_data"][0],
            "swir": st.session_state["post_data"][2] if st.session_state["post_data"].shape[0] > 2 else st.session_state["post_data"][0],
        }

        st.session_state["pre_index"] = compute_spectral_index(
            index_name=st.session_state["index_choice"],
            bands=pre_bands,
            green_transform=st.session_state["pre_meta"].transform,
            other_transform=st.session_state["pre_meta"].transform,
            nodata=st.session_state["pre_meta"].nodata,
        )
        st.session_state["post_index"] = compute_spectral_index(
            index_name=st.session_state["index_choice"],
            bands=post_bands,
            green_transform=st.session_state["post_meta"].transform,
            other_transform=st.session_state["post_meta"].transform,
            nodata=st.session_state["post_meta"].nodata,
        )

    with st.spinner("Segmenting water & executing change detection..."):
        threshold_val = None if st.session_state["use_otsu"] else st.session_state["manual_threshold"]
        st.session_state["flood_result"] = detect_flood_change(
            pre_index=st.session_state["pre_index"],
            post_index=st.session_state["post_index"],
            pre_threshold=threshold_val,
            post_threshold=threshold_val,
            use_otsu=st.session_state["use_otsu"],
            apply_morphology=True,
            min_size_pixels=st.session_state["min_patch_pixels"],
        )

    with st.spinner("Vectorizing raster masks & validating geometries..."):
        st.session_state["vector_results"] = vectorize_flood_results(
            inundation_mask=st.session_state["flood_result"].inundation_mask,
            permanent_water_mask=st.session_state["flood_result"].permanent_water_mask,
            metadata=st.session_state["pre_meta"],
        )

    with st.spinner("Calculating square kilometre metrics in projected CRS..."):
        st.session_state["impact_summary"] = calculate_flood_impact(
            inundation_gdf=st.session_state["vector_results"]["inundation_gdf"],
            permanent_gdf=st.session_state["vector_results"]["permanent_gdf"],
            pre_water_mask=st.session_state["flood_result"].pre_water_mask,
            post_water_mask=st.session_state["flood_result"].post_water_mask,
            metadata=st.session_state["pre_meta"],
            admin_gdf=st.session_state["admin_gdf"],
        )

    st.session_state["pipeline_executed"] = True
    st.success("✅ Analysis completed successfully across all processing stages!")


def load_sample_data():
    """Load bundled Sentinel-2 sample flood dataset."""
    pre_path = SAMPLE_DIR / "pre_disaster_sentinel2.tif"
    post_path = SAMPLE_DIR / "post_disaster_sentinel2.tif"
    admin_path = SAMPLE_DIR / "sample_administrative_boundaries.geojson"

    # Generate if not already present
    if not pre_path.exists() or not post_path.exists():
        generate_sample_flood_data(SAMPLE_DIR)

    pre_data, pre_meta = read_raster_from_source(pre_path)
    post_data, post_meta = read_raster_from_source(post_path)

    st.session_state["pre_data"] = pre_data
    st.session_state["pre_meta"] = pre_meta
    st.session_state["post_data"] = post_data
    st.session_state["post_meta"] = post_meta

    if admin_path.exists():
        st.session_state["admin_gdf"] = gpd.read_file(admin_path)

    run_pipeline()


# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <h2 style="color: #38BDF8; margin: 0; font-size: 1.8rem; font-weight: 800;">🛰️ {APP_NAME}</h2>
            <p style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">{APP_SUBTITLE}</p>
            <span class="badge-pill badge-sentinel">v{APP_VERSION}</span>
            <span class="badge-pill badge-mndwi">Sentinel-2</span>
            <span class="badge-pill badge-otsu">Autonomous</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "🏠 1. Home / Overview",
            "🛰️ 2. Satellite Ingestion",
            "🔄 3. Pre/Post Comparison",
            "🔬 4. Spectral Water Index",
            "🌊 5. Flood Detection",
            "📊 6. Impact Statistics",
            "🗺️ 7. Interactive Flood Map",
            "📥 8. Download Results",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚡ Quick Start")
    if st.button("🚀 Load Sample Flood Dataset", use_container_width=True, type="primary"):
        load_sample_data()

    if st.session_state["pipeline_executed"]:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 6px; padding: 10px; margin-top: 12px;">
                <b style="color: #34D399;">Pipeline Active</b><br>
                <small style="color: #E2E8F0;">Inundated Area: <b>{st.session_state['impact_summary'].newly_inundated_area_km2:.3f} km²</b></small>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- PAGE ROUTING ---

# 1. HOME / OVERVIEW
if "🏠 1. Home" in nav_selection:
    st.markdown(
        f"""
        <div class="main-header">
            <h1>🌊 {APP_NAME}</h1>
            <p>Automated Satellite Water Inundation & Disaster Change Detection System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        st.markdown(
            """
            ### Autonomous Remote Sensing Pipeline
            **GeoInundate** is an open-source, automated geospatial disaster response system
            engineered to detect, quantify, and map surface water inundation from multi-temporal
            Sentinel-2 satellite imagery.

            #### Core Capabilities:
            - **Multi-Temporal Ingestion:** Ingests pre- and post-disaster GeoTIFF rasters with automated CRS & resolution validation.
            - **Spectral Indexing:** Computes Modified Normalized Difference Water Index (**MNDWI**, Xu 2006) and **NDWI** (McFeeters 1996).
            - **Otsu's Adaptive Thresholding:** Dynamically calculates optimal water/land bimodal cutoffs via OpenCV.
            - **True-Area Vectorization:** Converts raster masks to cleaned polygon boundaries (GeoPandas / Shapely) and computes square kilometre metrics in local projected UTM coordinate systems.
            - **Interactive Leaflet Mapping:** High-resolution split-layer overlays of permanent water vs newly inundated floodplains.
            - **Disaster Reports:** Generates publication-ready PDF assessments and GIS vector downloads (GeoJSON & Shapefile ZIP).
            """
        )
        st.button("👉 Load Sample Sentinel-2 Dataset & Explore", on_click=load_sample_data, type="primary")

    with col2:
        st.markdown("### 🛠️ Architecture Workflow")
        st.markdown(
            """
            ```mermaid
            flowchart TD
                A["Pre & Post GeoTIFFs"] --> B["CRS & Grid Alignment"]
                B --> C["Spectral Index (MNDWI)"]
                C --> D["Otsu Adaptive Threshold"]
                D --> E["Morphological Cleaning"]
                E --> F["Change Detection (Pre vs Post)"]
                F --> G["Permanent vs Inundated Masks"]
                G --> H["Projected UTM Vectorization"]
                H --> I["Interactive Map & Reports"]
            ```
            """
        )

    with col3:
        st.markdown("### 📋 System Status")
        st.info(
            f"""
            - **App Version:** {APP_VERSION}
            - **Sentinel-2 Bands:** B03 (Green), B08 (NIR), B11 (SWIR)
            - **Default Index:** {DEFAULT_INDEX}
            - **Thresholding:** Otsu Adaptive + Manual Fine-Tuning
            - **Vector Formats:** GeoJSON, Shapefile ZIP, GeoTIFF
            - **Public Access:** Free, zero-login, open STAC/sample ready
            """
        )

# 2. SATELLITE IMAGE UPLOAD
elif "🛰️ 2. Satellite" in nav_selection:
    st.markdown("## 🛰️ Satellite Data Ingestion & Band Alignment")
    st.markdown("Upload pre- and post-disaster GeoTIFF imagery or load verified open sample scenes.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📅 Pre-Disaster Imagery (Baseline)")
        pre_file = st.file_uploader(
            "Upload Pre-Disaster GeoTIFF (Multi-band or B3/B8/B11)",
            type=["tif", "tiff"],
            key="upload_pre",
        )
        if pre_file is not None:
            pre_data, pre_meta = read_raster_from_source(BytesIO(pre_file.read()))
            st.session_state["pre_data"] = pre_data
            st.session_state["pre_meta"] = pre_meta
            st.success(f"Loaded Pre-Disaster Raster: {pre_meta.width}x{pre_meta.height} px, CRS: {pre_meta.crs}")

    with c2:
        st.markdown("#### 🌊 Post-Disaster Imagery (Flood Event)")
        post_file = st.file_uploader(
            "Upload Post-Disaster GeoTIFF (Multi-band or B3/B8/B11)",
            type=["tif", "tiff"],
            key="upload_post",
        )
        if post_file is not None:
            post_data, post_meta = read_raster_from_source(BytesIO(post_file.read()))
            st.session_state["post_data"] = post_data
            st.session_state["post_meta"] = post_meta
            st.success(f"Loaded Post-Disaster Raster: {post_meta.width}x{post_meta.height} px, CRS: {post_meta.crs}")

    st.markdown("#### 🗺️ Administrative Boundaries (Optional)")
    admin_file = st.file_uploader(
        "Upload District / Boundary GeoJSON or Shapefile ZIP",
        type=["geojson", "json"],
        key="upload_admin",
    )
    if admin_file is not None:
        try:
            admin_bytes = BytesIO(admin_file.read())
            st.session_state["admin_gdf"] = gpd.read_file(admin_bytes)
            st.success(f"Loaded {len(st.session_state['admin_gdf'])} administrative boundary units.")
        except Exception as e:
            st.error(f"Error parsing boundary file: {e}")

    # Metadata Inspection Table
    if st.session_state["pre_meta"] is not None and st.session_state["post_meta"] is not None:
        st.markdown("### 📊 Ingested Raster Metadata Comparison")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Pre-Disaster Metadata:**")
            st.json(st.session_state["pre_meta"].to_dict())
        with col_m2:
            st.markdown("**Post-Disaster Metadata:**")
            st.json(st.session_state["post_meta"].to_dict())

        st.markdown("---")
        if st.button("⚡ Execute Full Inundation Pipeline", type="primary", use_container_width=True):
            run_pipeline()

# 3. PRE/POST DISASTER COMPARISON
elif "🔄 3. Pre/Post" in nav_selection:
    st.markdown("## 🔄 Pre- and Post-Disaster Satellite Comparison")

    if st.session_state["pre_data"] is None or st.session_state["post_data"] is None:
        st.warning("⚠️ No satellite imagery loaded. Please load the sample dataset or upload GeoTIFFs.")
        st.button("🚀 Load Sample Dataset", on_click=load_sample_data, type="primary")
    else:
        # Create False Color Composite (SWIR-NIR-Green) or Natural Color
        def make_rgb_preview(bands_arr: np.ndarray) -> np.ndarray:
            """Create normalized 8-bit RGB preview from multi-band data."""
            # Use B11 (SWIR), B08 (NIR), B03 (Green) for standard water false-color composite
            if bands_arr.shape[0] >= 3:
                r = clean_reflectance(bands_arr[2])  # SWIR
                g = clean_reflectance(bands_arr[1])  # NIR
                b = clean_reflectance(bands_arr[0])  # Green
            else:
                r = g = b = clean_reflectance(bands_arr[0])

            rgb = np.dstack([r, g, b])
            # 2% - 98% percentile stretch
            valid = ~np.isnan(rgb)
            p2, p98 = np.nanpercentile(rgb[valid], 2), np.nanpercentile(rgb[valid], 98)
            rgb_stretched = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
            return (rgb_stretched * 255).astype(np.uint8)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### 📅 Pre-Disaster Composite (SWIR-NIR-Green)")
            st.image(
                make_rgb_preview(st.session_state["pre_data"]),
                caption=f"Baseline Scene ({st.session_state['pre_meta'].width}x{st.session_state['pre_meta'].height} px)",
                use_container_width=True,
            )
            st.caption("Deep blue / dark channels represent permanent river channel and baseline water.")

        with col_c2:
            st.markdown("#### 🌊 Post-Disaster Composite (SWIR-NIR-Green)")
            st.image(
                make_rgb_preview(st.session_state["post_data"]),
                caption=f"Flood Scene ({st.session_state['post_meta'].width}x{st.session_state['post_meta'].height} px)",
                use_container_width=True,
            )
            st.caption("Expanded dark/cyan areas illustrate extensive floodplain inundation and overflow.")

# 4. SPECTRAL WATER INDEX
elif "🔬 4. Spectral" in nav_selection:
    st.markdown("## 🔬 Spectral Water Index Analysis (MNDWI / NDWI)")

    if st.session_state["pre_index"] is None:
        st.warning("⚠️ Spectral index not yet calculated. Please load dataset and run pipeline.")
        st.button("🚀 Load Sample Dataset", on_click=load_sample_data, type="primary")
    else:
        st.markdown(
            f"""
            **Active Index:** `{st.session_state['index_choice']}`  
            *Formula:* `{INDICES[st.session_state['index_choice']]['formula']}`  
            *{INDICES[st.session_state['index_choice']]['description']}*
            """
        )

        col_idx1, col_idx2 = st.columns(2)
        with col_idx1:
            st.markdown("#### Pre-Disaster Index Image")
            pre_rgba = index_to_rgba(st.session_state["pre_index"], colormap_name="Blues", vmin=-0.5, vmax=0.5)
            st.image(pre_rgba, caption="Pre-Disaster Water Index (-0.5 to +0.5 stretched)", use_container_width=True)

        with col_idx2:
            st.markdown("#### Post-Disaster Index Image")
            post_rgba = index_to_rgba(st.session_state["post_index"], colormap_name="Blues", vmin=-0.5, vmax=0.5)
            st.image(post_rgba, caption="Post-Disaster Water Index (-0.5 to +0.5 stretched)", use_container_width=True)

        st.markdown("### 📈 Index Distribution & Thresholding Cutoff")
        thresh = (
            st.session_state["flood_result"].post_threshold_used
            if st.session_state["flood_result"]
            else DEFAULT_MANUAL_THRESHOLD
        )
        fig_hist = create_index_histogram(
            pre_index=st.session_state["pre_index"],
            post_index=st.session_state["post_index"],
            pre_threshold=thresh,
            post_threshold=thresh,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# 5. FLOOD DETECTION & SEGMENTATION
elif "🌊 5. Flood" in nav_selection:
    st.markdown("## 🌊 Automated Flood Detection & Change Segmentation")

    if st.session_state["flood_result"] is None:
        st.warning("⚠️ Flood detection not yet executed. Load dataset and run pipeline.")
        st.button("🚀 Load Sample Dataset", on_click=load_sample_data, type="primary")
    else:
        st.markdown("### ⚙️ Segmentation Parameter Tuning")
        with st.expander("🛠️ Advanced Segmentation Controls", expanded=True):
            cp1, cp2, cp3 = st.columns(3)
            with cp1:
                use_otsu = st.toggle("Use Otsu's Adaptive Threshold", value=st.session_state["use_otsu"])
                st.session_state["use_otsu"] = use_otsu
            with cp2:
                manual_thresh = st.slider(
                    "Manual Index Threshold",
                    min_value=-0.4,
                    max_value=0.4,
                    value=float(st.session_state["manual_threshold"]),
                    step=0.01,
                    disabled=use_otsu,
                )
                st.session_state["manual_threshold"] = manual_thresh
            with cp3:
                min_pixels = st.number_input(
                    "Min Patch Size (Noise Filter, px)",
                    min_value=0,
                    max_value=200,
                    value=int(st.session_state["min_patch_pixels"]),
                    step=5,
                )
                st.session_state["min_patch_pixels"] = min_pixels

            if st.button("🔄 Re-calculate Flood Segmentation", type="primary"):
                run_pipeline()
                st.rerun()

        res = st.session_state["flood_result"]
        st.markdown("### 🎯 Otsu Diagnostic Cutoffs")
        st.info(
            f"""
            - **Pre-Disaster Computed Otsu Threshold:** `{res.pre_otsu_computed:.4f}` (Applied: `{res.pre_threshold_used:.4f}`)
            - **Post-Disaster Computed Otsu Threshold:** `{res.post_otsu_computed:.4f}` (Applied: `{res.post_threshold_used:.4f}`)
            - **Class Pixel Distribution:** Land: `{res.pixel_counts['land']:,}` | Permanent Water: `{res.pixel_counts['permanent_water']:,}` | Newly Inundated: `{res.pixel_counts['newly_inundated']:,}` | Receded: `{res.pixel_counts['receded_water']:,}`
            """
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 🟦 Baseline Permanent Water Mask")
            st.image((res.permanent_water_mask * 255).astype(np.uint8), caption="Permanent Baseline Water Bodies", use_container_width=True)

        with col_m2:
            st.markdown("#### 🟥 Newly Inundated Land Mask (Flood)")
            st.image((res.inundation_mask * 255).astype(np.uint8), caption="Newly Flooded Terrain", use_container_width=True)

# 6. IMPACT STATISTICS & CHARTS
elif "📊 6. Impact" in nav_selection:
    st.markdown("## 📊 Flood Impact Analytics & Disaster Statistics")

    if st.session_state["impact_summary"] is None:
        st.warning("⚠️ No impact statistics available. Run pipeline first.")
        st.button("🚀 Load Sample Dataset", on_click=load_sample_data, type="primary")
    else:
        impact = st.session_state["impact_summary"]

        # KPI Metric Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card danger">
                    <div class="kpi-title">Newly Inundated Area</div>
                    <div class="kpi-value" style="color: #DC2626;">{impact.newly_inundated_area_km2:.3f} <span style="font-size:1rem;">km²</span></div>
                    <div class="kpi-sub">{impact.inundation_percentage:.2f}% of scene affected</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card info">
                    <div class="kpi-title">Permanent Baseline Water</div>
                    <div class="kpi-value" style="color: #38BDF8;">{impact.pre_water_area_km2:.3f} <span style="font-size:1rem;">km²</span></div>
                    <div class="kpi-sub">River channel & oxbows</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card warning">
                    <div class="kpi-title">Post-Event Total Water</div>
                    <div class="kpi-value" style="color: #F59E0B;">{impact.post_water_area_km2:.3f} <span style="font-size:1rem;">km²</span></div>
                    <div class="kpi-sub">Total surface water extent</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card success">
                    <div class="kpi-title">Flood Polygons</div>
                    <div class="kpi-value" style="color: #10B981;">{impact.num_flood_polygons:,}</div>
                    <div class="kpi-sub">Max patch: {impact.max_flood_patch_km2:.3f} km²</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        c_g1, c_g2 = st.columns(2)
        with c_g1:
            st.plotly_chart(create_area_comparison_chart(impact), use_container_width=True)
        with c_g2:
            st.plotly_chart(create_land_cover_donut(impact), use_container_width=True)

        if impact.district_breakdown:
            st.markdown("### 🏛️ District-Wise Inundation Impact")
            c_d1, c_d2 = st.columns([1.2, 1])
            with c_d1:
                st.plotly_chart(create_district_impact_chart(impact.district_breakdown), use_container_width=True)
            with c_d2:
                df_dist = pd.DataFrame(impact.district_breakdown)
                df_dist["Share (%)"] = (df_dist["flooded_area_km2"] / impact.newly_inundated_area_km2 * 100.0).round(2)
                df_dist.columns = ["District / Region", "Flooded Area (km²)", "Share (%)"]
                st.dataframe(df_dist, use_container_width=True)

# 7. INTERACTIVE GEOSPATIAL MAP
elif "🗺️ 7. Interactive" in nav_selection:
    st.markdown("## 🗺️ Interactive Geospatial Flood Inundation Map")

    if st.session_state["pre_meta"] is None or st.session_state["flood_result"] is None:
        st.warning("⚠️ No processed flood layers available. Please run pipeline.")
        st.button("🚀 Load Sample Dataset", on_click=load_sample_data, type="primary")
    else:
        st.markdown(
            """
            *Use the top-right layer control to toggle between Inundated Polygons, Permanent Water,
            MNDWI Indices, and Administrative Boundaries.*
            """
        )

        folium_map = build_interactive_map(
            metadata=st.session_state["pre_meta"],
            change_mask=st.session_state["flood_result"].change_mask,
            pre_index=st.session_state["pre_index"],
            post_index=st.session_state["post_index"],
            inundation_gdf=st.session_state["vector_results"]["inundation_gdf"],
            permanent_gdf=st.session_state["vector_results"]["permanent_gdf"],
            admin_gdf=st.session_state["admin_gdf"],
        )

        st_folium(folium_map, width="100%", height=650)

# 8. DOWNLOAD RESULTS & REPORTS
elif "📥 8. Download" in nav_selection:
    st.markdown("## 📥 Download Geospatial Results & Disaster Reports")

    if st.session_state["impact_summary"] is None:
        st.warning("⚠️ No results available to export. Run pipeline first.")
        st.button("🚀 Load Sample Dataset", on_click=load_sample_data, type="primary")
    else:
        impact = st.session_state["impact_summary"]
        meta = st.session_state["pre_meta"]
        res = st.session_state["flood_result"]
        vec = st.session_state["vector_results"]

        st.markdown("### 📑 Disaster Assessment Report")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            event_name = st.text_input("Event Name for Report", value=st.session_state["event_name"])
            st.session_state["event_name"] = event_name

            # Generate PDF Report
            pdf_bytes = generate_pdf_report(
                event_name=event_name,
                impact=impact,
                metadata=meta,
                index_name=st.session_state["index_choice"],
                method_name="Otsu Adaptive Thresholding" if st.session_state["use_otsu"] else f"Manual Cutoff ({st.session_state['manual_threshold']})",
            )
            st.download_button(
                label="📄 Download Official PDF Report",
                data=pdf_bytes,
                file_name="GeoInundate_Flood_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )

        with col_r2:
            md_content = generate_markdown_report(
                event_name=event_name,
                impact=impact,
                metadata=meta,
                index_name=st.session_state["index_choice"],
                method_name="Otsu Adaptive Thresholding" if st.session_state["use_otsu"] else "Manual Cutoff",
            )
            st.download_button(
                label="📝 Download Markdown Summary (.md)",
                data=md_content,
                file_name="GeoInundate_Flood_Summary.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### 🌐 Vector & Raster Geospatial Exports")
        col_e1, col_e2, col_e3 = st.columns(3)

        with col_e1:
            st.markdown("#### 1. GeoJSON Vectors")
            geojson_path = OUTPUTS_DIR / "newly_inundated_flood_polygons.geojson"
            export_geojson(vec["inundation_gdf"], geojson_path)
            with open(geojson_path, "rb") as f:
                st.download_button(
                    label="📥 Download GeoJSON",
                    data=f.read(),
                    file_name="newly_inundated_flood_polygons.geojson",
                    mime="application/geo+json",
                    use_container_width=True,
                )
            st.caption("Standard WGS84 GeoJSON containing newly inundated flood polygons.")

        with col_e2:
            st.markdown("#### 2. Shapefile ZIP")
            shp_zip_path = OUTPUTS_DIR / "flood_polygons_shp.zip"
            export_shapefile_zip(vec["inundation_gdf"], shp_zip_path)
            with open(shp_zip_path, "rb") as f:
                st.download_button(
                    label="📦 Download Shapefile ZIP",
                    data=f.read(),
                    file_name="flood_polygons_shapefile.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            st.caption("Complete ESRI Shapefile bundle (.shp, .shx, .dbf, .prj).")

        with col_e3:
            st.markdown("#### 3. GeoTIFF Mask")
            tif_mask_path = OUTPUTS_DIR / "flood_classification_mask.tif"
            export_classification_raster(res.change_mask, meta, tif_mask_path)
            with open(tif_mask_path, "rb") as f:
                st.download_button(
                    label="🗺️ Download GeoTIFF Mask",
                    data=f.read(),
                    file_name="flood_classification_mask.tif",
                    mime="image/tiff",
                    use_container_width=True,
                )
            st.caption("Georeferenced categorical raster (0=Land, 1=Perm, 2=Inundated).")



