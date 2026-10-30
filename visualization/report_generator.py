"""
Disaster Analysis Report Generation Module for GeoInundate.
Produces professional, publication-quality PDF reports (ReportLab) and
structured Markdown summaries for emergency response and GIS records.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from processing.area_calculation import FloodImpactSummary
from processing.raster_loader import RasterMetadata


def generate_pdf_report(
    event_name: str,
    impact: FloodImpactSummary,
    metadata: RasterMetadata,
    index_name: str,
    method_name: str,
    analysis_date: Optional[str] = None,
    satellite_source: str = "Sentinel-2 MSI (Copernicus)",
) -> bytes:
    """
    Generate a formal PDF Flood Inundation & Change Detection Assessment Report.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#475569"),
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0369A1"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#991B1B"),
    )

    story = []

    # Title Banner
    story.append(Paragraph("GeoInundate – Satellite Flood Inundation Assessment", title_style))
    story.append(Spacer(1, 4))
    date_str = analysis_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"<b>Event:</b> {event_name} | <b>Report Generated:</b> {date_str}", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=12))

    # Executive Summary Callout Box
    summary_text = (
        f"<b>CRITICAL FINDINGS:</b> An automated multi-temporal remote sensing assessment identified "
        f"<b>{impact.newly_inundated_area_km2:.3f} km²</b> of newly inundated flood land across "
        f"{impact.num_flood_polygons:,} discrete flood patches, representing "
        f"<b>{impact.inundation_percentage:.2f}%</b> of the analyzed scene. "
        f"Permanent baseline water was measured at <b>{impact.pre_water_area_km2:.3f} km²</b>, "
        f"expanding to a total surface water extent of <b>{impact.post_water_area_km2:.3f} km²</b> post-event."
    )
    summary_table = Table([[Paragraph(summary_text, callout_style)]], colWidths=[530])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
            ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#DC2626")),
            ("PADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # Inundation Metrics Table
    story.append(Paragraph("1. Primary Inundation & Impact Metrics", heading_style))

    metrics_data = [
        ["Metric Description", "Measured Value", "Units / Details"],
        ["Newly Inundated Area (Flood)", f"{impact.newly_inundated_area_km2:.4f}", "Square Kilometres (km²)"],
        ["Permanent Baseline Water Area", f"{impact.pre_water_area_km2:.4f}", "Square Kilometres (km²)"],
        ["Total Post-Event Water Extent", f"{impact.post_water_area_km2:.4f}", "Square Kilometres (km²)"],
        ["Receded Water Area", f"{impact.receded_water_area_km2:.4f}", "Square Kilometres (km²)"],
        ["Total Analyzed Scene Area", f"{impact.total_scene_area_km2:.4f}", "Square Kilometres (km²)"],
        ["Scene Inundation Fraction", f"{impact.inundation_percentage:.2f}%", "Percentage of total area"],
        ["Number of Flood Polygons", f"{impact.num_flood_polygons:,}", "Discrete contiguous patches"],
        ["Mean Flood Patch Size", f"{impact.mean_flood_patch_km2:.4f}", "Square Kilometres (km²)"],
        ["Largest Flood Patch Size", f"{impact.max_flood_patch_km2:.4f}", "Square Kilometres (km²)"],
        ["Calculation CRS Projection", f"{impact.projection_used}", "Meter-accurate projected CRS"],
    ]

    metrics_table = Table(metrics_data, colWidths=[230, 130, 170])
    metrics_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284C7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9.5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FEE2E2")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(metrics_table)
    story.append(Spacer(1, 14))

    # Sensor & Methodology Table
    story.append(Paragraph("2. Sensor Specifications & Analytical Methodology", heading_style))

    sensor_data = [
        ["Parameter", "Operational Setting"],
        ["Satellite Sensor / Platform", satellite_source],
        ["Spectral Water Index", f"{index_name} (Xu, 2006 / McFeeters, 1996)"],
        ["Classification Method", method_name],
        ["Raster Dimensions", f"{metadata.width} × {metadata.height} px ({metadata.count} bands)"],
        ["Spatial Resolution", f"{metadata.res_x:.2f}m × {metadata.res_y:.2f}m"],
        ["Native Coordinate System", metadata.crs],
        ["WGS84 Bounding Coordinates", f"{metadata.bounds_wgs84[1]:.4f}N, {metadata.bounds_wgs84[0]:.4f}E to {metadata.bounds_wgs84[3]:.4f}N, {metadata.bounds_wgs84[2]:.4f}E"],
    ]

    sensor_table = Table(sensor_data, colWidths=[200, 330])
    sensor_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("PADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(sensor_table)
    story.append(Spacer(1, 14))

    # Administrative Breakdown Table (if present)
    if impact.district_breakdown:
        story.append(Paragraph("3. Inundation Extent by Administrative Boundary", heading_style))
        admin_data = [["District / Region", "Flooded Area (km²)", "Share of Total Flood"]]
        for d in impact.district_breakdown:
            share = (
                (d["flooded_area_km2"] / impact.newly_inundated_area_km2 * 100.0)
                if impact.newly_inundated_area_km2 > 0
                else 0.0
            )
            admin_data.append([
                d["district"],
                f"{d['flooded_area_km2']:.4f}",
                f"{share:.2f}%",
            ])

        admin_table = Table(admin_data, colWidths=[240, 150, 140])
        admin_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#047857")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(admin_table)
        story.append(Spacer(1, 14))

    # Footer note
    footer_text = (
        "<i>Notice: Generated automatically by GeoInundate Automated Satellite Water Inundation System. "
        "Area measurements derived using true-area projected coordinate transformations.</i>"
    )
    story.append(Paragraph(footer_text, subtitle_style))

    doc.build(story)
    return buffer.getvalue()


def generate_markdown_report(
    event_name: str,
    impact: FloodImpactSummary,
    metadata: RasterMetadata,
    index_name: str,
    method_name: str,
    analysis_date: Optional[str] = None,
) -> str:
    """Generate a clean Markdown disaster summary."""
    date_str = analysis_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    md = f"""# GeoInundate – Satellite Flood Inundation Assessment Report

**Event Name:** {event_name}  
**Analysis Timestamp:** {date_str}  
**Spectral Water Index:** {index_name}  
**Classification Method:** {method_name}  
**Calculation Projection:** {impact.projection_used}  

---

## 1. Executive Summary

- **Newly Inundated Area (Flood):** `{impact.newly_inundated_area_km2:.4f} km²`
- **Permanent Baseline Water Area:** `{impact.pre_water_area_km2:.4f} km²`
- **Total Post-Event Water Extent:** `{impact.post_water_area_km2:.4f} km²`
- **Receded Water Area:** `{impact.receded_water_area_km2:.4f} km²`
- **Total Scene Area:** `{impact.total_scene_area_km2:.4f} km²`
- **Inundation Percentage:** `{impact.inundation_percentage:.2f}%` of total scene
- **Flood Patches (Polygons):** `{impact.num_flood_polygons:,}` discrete patches
- **Mean Patch Size:** `{impact.mean_flood_patch_km2:.4f} km²`
- **Largest Single Patch:** `{impact.max_flood_patch_km2:.4f} km²`

---

## 2. Satellite & Sensor Metadata

- **Raster Size:** `{metadata.width} × {metadata.height} px` ({metadata.count} bands)
- **Spatial Resolution:** `{metadata.res_x:.2f}m × {metadata.res_y:.2f}m`
- **Coordinate Reference System:** `{metadata.crs}`
- **WGS84 Extents:** `[{metadata.bounds_wgs84[0]:.5f}, {metadata.bounds_wgs84[1]:.5f}, {metadata.bounds_wgs84[2]:.5f}, {metadata.bounds_wgs84[3]:.5f}]`

---
"""
    if impact.district_breakdown:
        md += "## 3. District Inundation Breakdown\n\n"
        md += "| District | Inundated Area (km²) | Proportion |\n"
        md += "|---|---|---|\n"
        for d in impact.district_breakdown:
            share = (
                (d["flooded_area_km2"] / impact.newly_inundated_area_km2 * 100.0)
                if impact.newly_inundated_area_km2 > 0
                else 0.0
            )
            md += f"| **{d['district']}** | `{d['flooded_area_km2']:.4f} km²` | `{share:.1f}%` |\n"

    md += "\n---\n*Report generated automatically by GeoInundate Satellite Water Inundation Pipeline.*"
    return md
