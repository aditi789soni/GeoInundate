"""
Analytics and Interactive Charts Module for GeoInundate.
Constructs publication-ready Plotly charts matching modern dark-mode GIS interfaces.
"""

from typing import List, Optional
import numpy as np
import plotly.graph_objects as go

from config import COLORS
from processing.area_calculation import FloodImpactSummary

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(30, 41, 59, 0.4)",
    font=dict(color="#F8FAFC", family="sans-serif", size=13),
    margin=dict(l=40, r=30, t=50, b=40),
)


def create_area_comparison_chart(impact: FloodImpactSummary) -> go.Figure:
    """
    Bar chart comparing pre-event, post-event, and newly inundated water areas in km².
    """
    categories = [
        "Pre-Disaster Water",
        "Post-Disaster Water",
        "Newly Inundated Land",
        "Receded Water",
    ]
    values = [
        impact.pre_water_area_km2,
        impact.post_water_area_km2,
        impact.newly_inundated_area_km2,
        impact.receded_water_area_km2,
    ]
    bar_colors = [
        COLORS["permanent_water"],
        "#3B82F6",
        COLORS["new_inundation"],
        COLORS["receded_water"],
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                text=[f"{v:.3f} km²" for v in values],
                textposition="auto",
                marker=dict(
                    color=bar_colors,
                    line=dict(color="#334155", width=1.5),
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(text="<b>Water Inundation Extent (km²)</b>", font=dict(size=16)),
        yaxis=dict(title="Area (Square Kilometres)", gridcolor="#334155"),
        xaxis=dict(tickangle=-10),
        **DARK_LAYOUT,
    )
    return fig


def create_land_cover_donut(impact: FloodImpactSummary) -> go.Figure:
    """
    Donut chart illustrating total landscape classification proportions.
    """
    dry_land_km2 = max(
        0.0,
        impact.total_scene_area_km2
        - (impact.pre_water_area_km2 + impact.newly_inundated_area_km2),
    )

    labels = ["Dry Land", "Permanent Water", "Newly Inundated", "Receded Water"]
    values = [
        dry_land_km2,
        impact.pre_water_area_km2,
        impact.newly_inundated_area_km2,
        impact.receded_water_area_km2,
    ]
    colors = [
        "#475569",
        COLORS["permanent_water"],
        COLORS["new_inundation"],
        COLORS["receded_water"],
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                textinfo="label+percent",
                marker=dict(colors=colors, line=dict(color="#0F172A", width=2)),
            )
        ]
    )

    fig.update_layout(
        title=dict(text="<b>Scene Landscape Composition</b>", font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        **DARK_LAYOUT,
    )
    return fig


def create_district_impact_chart(district_breakdown: List[dict]) -> Optional[go.Figure]:
    """
    Horizontal bar chart depicting flooded area per administrative district.
    """
    if not district_breakdown:
        return None

    # Sort districts by inundated area descending
    sorted_districts = sorted(
        district_breakdown, key=lambda x: x["flooded_area_km2"], reverse=True
    )
    districts = [d["district"] for d in sorted_districts]
    areas = [d["flooded_area_km2"] for d in sorted_districts]

    fig = go.Figure(
        data=[
            go.Bar(
                x=areas,
                y=districts,
                orientation="h",
                text=[f"{a:.3f} km²" for a in areas],
                textposition="auto",
                marker=dict(
                    color=COLORS["new_inundation"],
                    line=dict(color="#7F1D1D", width=1),
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(text="<b>Flood Inundation by Administrative District</b>", font=dict(size=16)),
        xaxis=dict(title="Flooded Area (km²)", gridcolor="#334155"),
        yaxis=dict(autorange="reversed"),
        **DARK_LAYOUT,
    )
    return fig


def create_index_histogram(
    pre_index: np.ndarray,
    post_index: np.ndarray,
    pre_threshold: float,
    post_threshold: float,
) -> go.Figure:
    """
    Histogram comparing pre- and post-disaster spectral index distributions
    with water segmentation threshold cutoffs.
    """
    pre_valid = pre_index[~np.isnan(pre_index)]
    post_valid = post_index[~np.isnan(post_index)]

    # Subsample for chart rendering responsiveness if array is large
    if pre_valid.size > 20000:
        pre_sample = np.random.choice(pre_valid, size=20000, replace=False)
    else:
        pre_sample = pre_valid

    if post_valid.size > 20000:
        post_sample = np.random.choice(post_valid, size=20000, replace=False)
    else:
        post_sample = post_valid

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=pre_sample,
            name="Pre-Disaster Index",
            opacity=0.6,
            marker_color="#3B82F6",
            nbinsx=60,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=post_sample,
            name="Post-Disaster Index",
            opacity=0.6,
            marker_color="#DC2626",
            nbinsx=60,
        )
    )

    # Add vertical threshold lines
    fig.add_vline(
        x=post_threshold,
        line_width=2.5,
        line_dash="dash",
        line_color="#FBBF24",
        annotation_text=f"Threshold: {post_threshold:.3f}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(text="<b>Spectral Water Index Distribution</b>", font=dict(size=16)),
        barmode="overlay",
        xaxis=dict(title="Index Value [-1.0 to 1.0]", gridcolor="#334155"),
        yaxis=dict(title="Pixel Frequency", gridcolor="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **DARK_LAYOUT,
    )
    return fig
