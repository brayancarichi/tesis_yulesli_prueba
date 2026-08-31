"""
SISTEMA PLATINUM DE INTELIGENCIA GEOESPACIAL Y MONITOREO AGROCLIMÁTICO
Integración Sentinel-2 (NDVI, SAVI, MNDWI, NDMI), CHIRPS, SQLite, Streamlit y ReportLab
"""

import os
import io
import json
import logging
import sqlite3
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium.plugins import DualMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import shape

import ee

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# ==============================================================================
# CONFIGURACIÓN Y ESTILOS HIGH-TECH (DARK MODE DASHBOARD)
# ==============================================================================
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Plataforma Geoespacial - Inteligencia Agroclimática & Sentinel",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        .stMetric {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        div[data-testid="stMetricValue"] {
            font-size: 28px !important;
            font-weight: 800 !important;
            color: #38bdf8 !important;
        }
        .status-card-green {
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid #10b981;
            color: #34d399;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 15px;
        }
        .status-card-yellow {
            background-color: rgba(245, 158, 11, 0.1);
            border: 1px solid #f59e0b;
            color: #fbbf24;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 15px;
        }
        .status-card-red {
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
            color: #f87171;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 15px;
        }
        .status-card-blue {
            background-color: rgba(2, 132, 199, 0.1);
            border: 1px solid #0284c7;
            color: #38bdf8;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 15px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
            color: #ffffff;
            border-radius: 8px;
            border: none;
            padding: 12px 20px;
            font-weight: 700;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            box-shadow: 0 0 15px rgba(37, 99, 235, 0.6);
            transform: translateY(-1px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DB_NAME = "predios.db"


# ==============================================================================
# AUTENTICACIÓN GOOGLE EARTH ENGINE
# ==============================================================================
class GEEAuthManager:

    @staticmethod
    def initialize_earth_engine() -> bool:
        if st.session_state.get("gee_initialized", False):
            return True

        gee_json_str = os.getenv("GEE_JSON_KEY")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        try:
            if gee_json_str:
                key_content = (
                    json.loads(gee_json_str)
                    if isinstance(gee_json_str, str)
                    else gee_json_str
                )
                credentials = ee.ServiceAccountCredentials(
                    key_content["client_email"], key_data=json.dumps(key_content)
                )
                ee.Initialize(credentials, project=key_content.get("project_id"))
            elif credentials_path and os.path.exists(credentials_path):
                with open(credentials_path, "r") as f:
                    key_content = json.load(f)
                credentials = ee.ServiceAccountCredentials(
                    key_content["client_email"], key_file=credentials_path
                )
                ee.Initialize(credentials, project=key_content.get("project_id"))
            else:
                ee.Initialize()

            st.session_state["gee_initialized"] = True
            return True

        except Exception as e:
            st.error(f"Error de autenticación en Google Earth Engine API: {str(e)}")
            return False


# ==============================================================================
# PROCESAMIENTO SATELITAL Y CLIMÁTICO (SENTINEL-2 + CHIRPS)
# ==============================================================================
class SatelliteProcessor:

    @staticmethod
    def obtener_predios_db() -> pd.DataFrame:
        if not os.path.exists(DB_NAME):
            return pd.DataFrame()
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM predios", conn)
        conn.close()
        return df

    @staticmethod
    def mask_s2_clouds(image: ee.Image) -> ee.Image:
        qa = image.select("QA60")
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        return image.updateMask(mask).divide(10000)

    @staticmethod
    def compute_indices(image: ee.Image) -> ee.Image:
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
        savi = (
            image.expression(
                "((NIR - RED) / (NIR + RED + 0.5)) * 1.5",
                {"NIR": image.select("B8"), "RED": image.select("B4")},
            )
            .rename("SAVI")
        )
        ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
        return image.addBands([ndvi, mndwi, savi, ndmi])

    @classmethod
    def process_region(
            cls, geometry: ee.Geometry, start_date: str, end_date: str, cloud_cover: int
    ) -> tuple[ee.Image, dict]:
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover))
            .map(cls.mask_s2_clouds)
            .map(cls.compute_indices)
        )

        size = collection.size().getInfo()
        if size == 0:
            raise ValueError(
                "No se encontraron imágenes satelitales en las fechas seleccionadas."
            )

        composite = collection.median().clip(geometry)

        stats = composite.select(["NDVI", "MNDWI", "SAVI", "NDMI"]).reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.minMax(), sharedInputs=True),
            geometry=geometry,
            scale=10,
            bestEffort=True,
            maxPixels=1e9,
        ).getInfo()

        metadata = {
            "scene_count": size,
            "sensor": "Sentinel-2 MSI (Level-2A)",
            "start_date": start_date,
            "end_date": end_date,
            "cloud_cover_threshold": cloud_cover,
        }

        return composite, {**metadata, **stats}

    @staticmethod
    def calcular_dias_secos_consecutivos(df: pd.DataFrame, umbral_mm: float = 1.0) -> int:
        max_dias = 0
        actual = 0
        for val in df["Lluvia (mm)"]:
            if val < umbral_mm:
                actual += 1
                if actual > max_dias:
                    max_dias = actual
            else:
                actual = 0
        return max_dias

    @classmethod
    def procesar_evaluacion_chirps(cls, geojson_geom: dict, fecha_inicio: date, fecha_fin: date) -> dict:
        coords_geojson = geojson_geom['coordinates']
        area_estudio = ee.Geometry.Polygon(coords_geojson)
        centroide_geom = area_estudio.centroid(maxError=1)

        coleccion_lluvia = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterDate(fecha_inicio.strftime("%Y-%m-%d"), fecha_fin.strftime("%Y-%m-%d"))

        lluvia_acumulada = coleccion_lluvia.sum().clip(area_estudio)

        estadisticas = lluvia_acumulada.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area_estudio,
            scale=5000,
            bestEffort=True,
            maxPixels=1e9
        ).getInfo()

        promedio_mm = estadisticas.get('precipitation')
        if promedio_mm is None:
            estadisticas = lluvia_acumulada.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=centroide_geom,
                scale=5000,
                bestEffort=True,
                maxPixels=1e9
            ).getInfo()
            promedio_mm = estadisticas.get('precipitation', 0.0) or 0.0

        dia_inicio_ano = fecha_inicio.timetuple().tm_yday
        dia_fin_ano = fecha_fin.timetuple().tm_yday

        coleccion_historica = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filter(ee.Filter.calendarRange(2003, 2023, 'year')) \
            .filter(ee.Filter.dayOfYear(dia_inicio_ano, dia_fin_ano))

        lluvia_promedio_historica = coleccion_historica.sum().divide(20).clip(area_estudio)

        stats_hist = lluvia_promedio_historica.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area_estudio,
            scale=5000,
            bestEffort=True,
            maxPixels=1e9
        ).getInfo()

        promedio_historico_mm = stats_hist.get('precipitation')
        if promedio_historico_mm is None:
            stats_hist = lluvia_promedio_historica.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=centroide_geom,
                scale=5000,
                bestEffort=True,
                maxPixels=1e9
            ).getInfo()
            promedio_historico_mm = stats_hist.get('precipitation', 1.0) or 1.0

        anomalia_porcentaje = ((promedio_mm - promedio_historico_mm) / promedio_historico_mm) * 100

        def extraer_diario_centroide(img):
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=centroide_geom,
                scale=5000,
                bestEffort=True,
                maxPixels=1e9
            )
            precip = stats.get('precipitation')
            val = ee.Algorithms.If(ee.Algorithms.ObjectType(precip).equals('Null'), 0.0, precip)
            return ee.Feature(None, {
                'fecha': img.date().format("YYYY-MM-dd"),
                'precipitation': val
            })

        fc_diarios = ee.FeatureCollection(coleccion_lluvia.map(extraer_diario_centroide))
        features_info = fc_diarios.getInfo()['features']

        datos_diarios = []
        for f in features_info:
            props = f['properties']
            val = props.get('precipitation', 0.0)
            datos_diarios.append({
                "Fecha": props.get('fecha'),
                "Lluvia (mm)": round(float(val), 2) if val is not None else 0.0
            })

        df_diario = pd.DataFrame(datos_diarios)
        if not df_diario.empty:
            df_diario["Fecha"] = pd.to_datetime(df_diario["Fecha"]).dt.strftime('%Y-%m-%d')
            df_diario["Lluvia (mm)"] = pd.to_numeric(df_diario["Lluvia (mm)"], errors='coerce').fillna(0.0)

        racha_seca = cls.calcular_dias_secos_consecutivos(df_diario, umbral_mm=1.0)

        if anomalia_porcentaje <= -30.0 or (anomalia_porcentaje < 0 and racha_seca >= 18):
            estado_siniestro = "Riesgo Alto por Sequía"
            css_class = "status-card-red"
            explicacion_sencilla = "Ha llovido menos de lo habitual y existe una racha seca prolongada."
            recomendacion = "Revisar la humedad del suelo a profundidad y programar riegos de auxilio."
        elif anomalia_porcentaje > 25.0:
            estado_siniestro = "Alerta por Exceso de Humedad"
            css_class = "status-card-blue"
            explicacion_sencilla = "Ha llovido considerablemente más de lo habitual para estas fechas."
            recomendacion = "Verificar el correcto drenaje de la parcela y monitorear hongos."
        elif -10.0 <= anomalia_porcentaje <= 25.0:
            if racha_seca >= 15:
                estado_siniestro = "Condición Normal con Distribución Irregular"
                css_class = "status-card-yellow"
                explicacion_sencilla = "Volumen adecuado acumulado pero concentrado en pocos días."
                recomendacion = "Monitorear la humedad superficial por días secos consecutivos."
            else:
                estado_siniestro = "Condición Normal de Lluvia"
                css_class = "status-card-green"
                explicacion_sencilla = "La humedad acumulada por lluvias se mantiene en valores normales."
                recomendacion = "Mantener las labores del cultivo de manera habitual."
        else:
            estado_siniestro = "Riesgo Moderado por Lluvia Baja"
            css_class = "status-card-yellow"
            explicacion_sencilla = "Las lluvias han sido ligeramente inferiores a lo normal."
            recomendacion = "Monitorear el estado del follaje y planificar riegos."

        return {
            "promedio_mm": promedio_mm,
            "promedio_historico_mm": promedio_historico_mm,
            "anomalia_porcentaje": anomalia_porcentaje,
            "racha_seca": racha_seca,
            "estado_siniestro": estado_siniestro,
            "css_class": css_class,
            "explicacion_sencilla": explicacion_sencilla,
            "recomendacion": recomendacion,
            "df_diario": df_diario,
        }

    @staticmethod
    def calculate_mann_kendall_trend(
            start_year: int, end_year: int, analysis_type: str, bbox: list = None
    ) -> tuple[ee.Image, ee.Geometry]:
        if bbox:
            geometry = ee.Geometry.Polygon(bbox)
        else:
            geometry = ee.Geometry.Polygon([[
                [-101.3, 23.5], [-98.4, 23.5], [-98.4, 27.8], [-101.3, 27.8], [-101.3, 23.5]
            ]])

        startDate = f"{start_year}-01-01"
        endDate = f"{end_year}-12-31"

        if analysis_type == "Precipitación (CHIRPS)":
            collection = (
                ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD")
                .filterBounds(geometry)
                .filterDate(startDate, endDate)
                .select("precipitation")
            )

            def annual_sum(y):
                y = ee.Number(y)
                start = ee.Date.fromYMD(y, 1, 1)
                end = start.advance(1, "year")
                img = collection.filterDate(start, end).sum()
                return img.set("year", y).set("system:time_start", start.millis())

            years = ee.List.sequence(start_year, end_year)
            annual_coll = ee.ImageCollection(years.map(annual_sum))
            band_name = "precipitation"

        elif analysis_type == "Temperaturas (ERA5-Land)":
            collection = (
                ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY")
                .filterBounds(geometry)
                .filterDate(startDate, endDate)
                .select("temperature_2m")
            )

            def annual_mean(y):
                y = ee.Number(y)
                start = ee.Date.fromYMD(y, 1, 1)
                end = start.advance(1, "year")
                img = collection.filterDate(start, end).mean().subtract(273.15)
                return img.set("year", y).set("system:time_start", start.millis())

            years = ee.List.sequence(start_year, end_year)
            annual_coll = ee.ImageCollection(years.map(annual_mean))
            band_name = "temperature_2m"

        else:
            collection = (
                ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
                .filterBounds(geometry)
                .filterDate(startDate, endDate)
                .select("pdsi")
            )

            def annual_pdsi(y):
                y = ee.Number(y)
                start = ee.Date.fromYMD(y, 1, 1)
                end = start.advance(1, "year")
                img = collection.filterDate(start, end).mean()
                return img.set("year", y).set("system:time_start", start.millis())

            years = ee.List.sequence(start_year, end_year)
            annual_coll = ee.ImageCollection(years.map(annual_pdsi))
            band_name = "pdsi"

        time_band = "time"

        def add_time(img):
            year = ee.Number(img.get("year"))
            t = year.subtract(start_year)
            return img.addBands(ee.Image.constant(t).rename(time_band)).float()

        with_time = annual_coll.map(add_time)
        trend = with_time.select([time_band, band_name]).reduce(ee.Reducer.linearFit())
        slope = trend.select("scale").rename("trend_slope")

        return slope.clip(geometry), geometry


# ==============================================================================
# GENERACIÓN DE REPORTE PDF (REPORTLAB PLATINUM CANVAS)
# ==============================================================================
class NumberedCanvas(canvas.Canvas):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1E3A8A"))
        self.drawString(54, 750, "EVALUACIÓN SATELITAL Y AGROCLIMÁTICA POR PREDIO")
        self.drawRightString(558, 750, "SISTEMA DE PROCESAMIENTO GEOESPACIAL")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.75)
        self.line(54, 742, 558, 742)
        self.line(54, 50, 558, 50)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        self.drawString(54, 38, f"Documento generado el {ts}")
        self.drawRightString(558, 38, f"Página {self._pageNumber} de {page_count}")
        self.restoreState()


class PDFReportGenerator:

    @staticmethod
    def generate_pdf(stats_data: dict, chirps_data: dict, predio_name: str, cultivo_name: str, coords: list) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=72,
            bottomMargin=72,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=18, leading=22,
            textColor=colors.HexColor("#1E3A8A"), spaceAfter=12
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=14,
            textColor=colors.HexColor("#4B5563"), spaceAfter=18
        )
        heading1_style = ParagraphStyle(
            "SectionH1", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, leading=16,
            textColor=colors.HexColor("#1E3A8A"), spaceBefore=14, spaceAfter=8, keepWithNext=True
        )
        body_style = ParagraphStyle(
            "BodyTechnical", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=13.5,
            textColor=colors.HexColor("#1F2937"), spaceAfter=8
        )
        table_header_style = ParagraphStyle(
            "TableHeader", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8.5, leading=10,
            textColor=colors.white, alignment=1
        )
        table_body_style = ParagraphStyle(
            "TableBody", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=11,
            textColor=colors.HexColor("#111827"), alignment=0
        )

        story = []
        story.append(Paragraph("INFORME DE EVALUACIÓN MULTIESPECTRAL Y CLIMÁTICA", title_style))
        story.append(Paragraph(f"PREDIO: {predio_name.upper()} | CULTIVO: {cultivo_name.upper()}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))

        story.append(Paragraph("1. Resumen Ejecutivo y Metadatos", heading1_style))
        exec_summary = (
            f"Informe técnico para el predio <b>{predio_name}</b> dedicado al cultivo de <b>{cultivo_name}</b>. "
            "Se evalúan índices multiespectrales biofísicos Sentinel-2 y parámetros de precipitación acumulada CHIRPS."
        )
        story.append(Paragraph(exec_summary, body_style))

        def gv(key: str) -> str:
            val = stats_data.get(key)
            return f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"

        metrics_table_data = [
            [Paragraph("Índice Espectral", table_header_style), Paragraph("Promedio", table_header_style),
             Paragraph("Desv. Est.", table_header_style), Paragraph("Mínimo", table_header_style),
             Paragraph("Máximo", table_header_style)],
            [Paragraph("NDVI (Vegetación)", table_body_style), Paragraph(gv("NDVI_mean"), table_body_style),
             Paragraph(gv("NDVI_stdDev"), table_body_style), Paragraph(gv("NDVI_min"), table_body_style),
             Paragraph(gv("NDVI_max"), table_body_style)],
            [Paragraph("SAVI (Suelo/Veg.)", table_body_style), Paragraph(gv("SAVI_mean"), table_body_style),
             Paragraph(gv("SAVI_stdDev"), table_body_style), Paragraph(gv("SAVI_min"), table_body_style),
             Paragraph(gv("SAVI_max"), table_body_style)],
            [Paragraph("MNDWI (Cuerpos Agua)", table_body_style), Paragraph(gv("MNDWI_mean"), table_body_style),
             Paragraph(gv("MNDWI_stdDev"), table_body_style), Paragraph(gv("MNDWI_min"), table_body_style),
             Paragraph(gv("MNDWI_max"), table_body_style)],
            [Paragraph("NDMI (Humedad)", table_body_style), Paragraph(gv("NDMI_mean"), table_body_style),
             Paragraph(gv("NDMI_stdDev"), table_body_style), Paragraph(gv("NDMI_min"), table_body_style),
             Paragraph(gv("NDMI_max"), table_body_style)],
        ]

        t_metrics = Table(metrics_table_data, colWidths=[180, 81, 81, 81, 81])
        t_metrics.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_metrics)
        story.append(Spacer(1, 14))

        if chirps_data:
            story.append(Paragraph("2. Métrica Climática (CHIRPS)", heading1_style))
            chirps_table_data = [
                [Paragraph("Indicador", table_header_style), Paragraph("Valor Registrado", table_header_style)],
                [Paragraph("Precipitación Acumulada", table_body_style),
                 Paragraph(f"{chirps_data['promedio_mm']:.1f} mm", table_body_style)],
                [Paragraph("Promedio Histórico (20 años)", table_body_style),
                 Paragraph(f"{chirps_data['promedio_historico_mm']:.1f} mm", table_body_style)],
                [Paragraph("Anomalía de Precipitación", table_body_style),
                 Paragraph(f"{chirps_data['anomalia_porcentaje']:+.1f}%", table_body_style)],
                [Paragraph("Racha Seca Consecutiva", table_body_style),
                 Paragraph(f"{chirps_data['racha_seca']} días", table_body_style)],
                [Paragraph("Diagnóstico Territorial", table_body_style),
                 Paragraph(chirps_data['estado_siniestro'], table_body_style)],
            ]

            t_chirps = Table(chirps_table_data, colWidths=[200, 304])
            t_chirps.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#1E3A8A")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(t_chirps)

        story.append(KeepTogether([
            Spacer(1, 25),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E0"), spaceAfter=15),
            Table([
                [Paragraph("<b>Firma del Especialista Geoespacial</b>", table_body_style),
                 Paragraph("<b>Firma de Validación Agronómica</b>", table_body_style)],
                [Paragraph("__________________________________________", table_body_style),
                 Paragraph("__________________________________________", table_body_style)]
            ], colWidths=[252, 252])
        ]))

        doc.build(story, canvasmaker=NumberedCanvas)
        buffer.seek(0)
        return buffer.getvalue()


# ==============================================================================
# INTERFAZ DE USUARIO CON STREAMLIT
# ==============================================================================
def main():
    st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🛰️ SISTEMA GEOESPACIAL Y MONITOREO AGROCLIMÁTICO</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #94a3b8;'>Evaluación Multiespectral por Predio (Sentinel-2, CHIRPS & SQLite)</p>",
        unsafe_allow_html=True)

    if not GEEAuthManager.initialize_earth_engine():
        st.stop()

    df_predios = SatelliteProcessor.obtener_predios_db()

    tab_predios, tab_comparador, tab_tendencias = st.tabs([
        "Monitoreo por Predio (NDVI & CHIRPS)",
        "Comparador Temporal Split-Map",
        "Tendencias Climáticas Mann-Kendall"
    ])

    # ==================== PESTAÑA 1: PREDIO MULTIESPECTRAL + CHIRPS ====================
    with tab_predios:
        st.sidebar.header("Panel de Control por Predio")

        if df_predios.empty:
            st.warning("⚠️ No se encontraron predios en 'predios.db'.")
            st.stop()

        opciones_predios = {
            f"{row['nombre_predio']} ({row['tipo_cultivo']}) - {row['superficie_ha']} Ha": row['id']
            for _, row in df_predios.iterrows()
        }
        predio_selected_label = st.sidebar.selectbox(
            "Seleccionar Predio Guardado:",
            options=list(opciones_predios.keys())
        )
        predio_id = opciones_predios[predio_selected_label]
        predio_data = df_predios[df_predios["id"] == predio_id].iloc[0]

        geojson_geom = json.loads(predio_data['geojson_geom'])
        geom_shapely = shape(geojson_geom)
        coords = [geom_shapely.centroid.y, geom_shapely.centroid.x]

        col_f1, col_f2 = st.sidebar.columns(2)
        with col_f1:
            f_inicio = st.sidebar.date_input("Fecha Inicio", value=date.today() - timedelta(days=60))
        with col_f2:
            f_fin = st.sidebar.date_input("Fecha Fin", value=date.today() - timedelta(days=5))

        layer_type = st.sidebar.radio(
            "Capa Visual Principal:",
            ["NDVI (Vegetación)", "SAVI (Suelo/Veg.)", "Color Real (RGB)", "MNDWI (Agua)", "NDMI (Humedad Foliar)"]
        )

        btn_analizar = st.sidebar.button("Ejecutar Análisis de Predio")

        if btn_analizar or "results" not in st.session_state:
            with st.spinner(f"Procesando telemetría satelital para predio {predio_data['nombre_predio']}..."):
                try:
                    area_estudio = ee.Geometry.Polygon(geojson_geom['coordinates'])

                    composite, stats = SatelliteProcessor.process_region(
                        geometry=area_estudio,
                        start_date=f_inicio.strftime("%Y-%m-%d"),
                        end_date=f_fin.strftime("%Y-%m-%d"),
                        cloud_cover=20,
                    )

                    chirps_eval = SatelliteProcessor.procesar_evaluacion_chirps(
                        geojson_geom=geojson_geom,
                        fecha_inicio=f_inicio,
                        fecha_fin=f_fin
                    )

                    st.session_state["results"] = {
                        "composite": composite,
                        "stats": stats,
                        "chirps": chirps_eval,
                        "predio_nombre": predio_data['nombre_predio'],
                        "tipo_cultivo": predio_data['tipo_cultivo'],
                        "coords": coords,
                        "geojson_geom": geojson_geom,
                    }
                except Exception as e:
                    st.error(f"Error procesando el predio: {str(e)}")
                    st.stop()

        if "results" in st.session_state:
            res = st.session_state["results"]
            stats = res["stats"]
            chirps = res["chirps"]

            # Tarjetas Métricas Espectrales y Climáticas
            col1, col2, col3, col4, col5 = st.columns(5)
            ndvi_val = stats.get('NDVI_mean', 0)
            col1.metric("NDVI (Vegetación)", f"{ndvi_val:.4f}")
            col2.metric("SAVI (Suelo/Veg.)", f"{stats.get('SAVI_mean', 0):.4f}")
            col3.metric("NDMI (Humedad)", f"{stats.get('NDMI_mean', 0):.4f}")
            col4.metric("Lluvia Acumulada", f"{chirps['promedio_mm']:.1f} mm")
            col5.metric("Racha Seca", f"{chirps['racha_seca']} Días")

            # Diagnósticos Combinados
            st.markdown("### Diagnóstico Agroclimático del Predio")
            col_diag1, col_diag2 = st.columns(2)

            with col_diag1:
                if ndvi_val < 0.2:
                    st.markdown(
                        "<div class='status-card-red'>🔴 SALUD VEGETAL CRÍTICA: Cobertura foliar muy baja o en desuso.</div>",
                        unsafe_allow_html=True
                    )
                elif ndvi_val < 0.35:
                    st.markdown(
                        "<div class='status-card-yellow'>🟡 SALUD VEGETAL MODERADA: Vegetación escasa o en fase temprana.</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div class='status-card-green'>🟢 SALUD VEGETAL ÓPTIMA: Follaje denso y vigoroso.</div>",
                        unsafe_allow_html=True
                    )

            with col_diag2:
                st.markdown(
                    f"""
                    <div class='{chirps['css_class']}'>
                        <b>Precipitación: {chirps['estado_siniestro']}</b><br/>
                        {chirps['recomendacion']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # Mapa Folium e Histogramas
            m = folium.Map(location=res["coords"], zoom_start=14, tiles="CartoDB dark_matter")

            if layer_type == "NDVI (Vegetación)":
                vis_params = {"min": 0.0, "max": 0.8,
                              "palette": ["FFFFFF", "CE7E45", "DF923D", "FCD163", "99B718", "397D02", "004C00"]}
                map_img = ee.Image(res["composite"].select("NDVI"))
            elif layer_type == "SAVI (Suelo/Veg.)":
                vis_params = {"min": 0.0, "max": 0.8, "palette": ["FFFFFF", "DF923D", "FCD163", "397D02"]}
                map_img = ee.Image(res["composite"].select("SAVI"))
            elif layer_type == "Color Real (RGB)":
                vis_params = {"min": 0.0, "max": 0.3, "bands": ["B4", "B3", "B2"]}
                map_img = res["composite"]
            elif layer_type == "MNDWI (Agua)":
                vis_params = {"min": -0.5, "max": 0.5, "palette": ["brown", "white", "blue"]}
                map_img = ee.Image(res["composite"].select("MNDWI"))
            else:
                vis_params = {"min": -0.4, "max": 0.4, "palette": ["blue", "white", "green"]}
                map_img = ee.Image(res["composite"].select("NDMI"))

            map_id = map_img.getMapId(vis_params)
            folium.TileLayer(
                tiles=map_id["tile_fetcher"].url_format,
                attr="Google Earth Engine - Sentinel-2",
                name=layer_type,
                overlay=True,
                control=True,
            ).add_to(m)

            folium.GeoJson(
                res["geojson_geom"],
                name="Predio",
                style_function=lambda x: {'fillColor': '#00000000', 'color': '#38bdf8', 'weight': 3}
            ).add_to(m)

            col_map, col_chart = st.columns([1.3, 1])

            with col_map:
                st.subheader("Visualización Satelital del Polígono")
                st_folium(m, width="100%", height=450)

            with col_chart:
                st.subheader("📊 Índices Biofísicos del Cultivo")
                df_chart = pd.DataFrame({
                    "Índice": ["NDVI", "SAVI", "NDMI", "MNDWI"],
                    "Valor Promedio": [stats.get('NDVI_mean', 0), stats.get('SAVI_mean', 0), stats.get('NDMI_mean', 0),
                                       stats.get('MNDWI_mean', 0)]
                })
                fig = px.bar(
                    df_chart,
                    x="Índice",
                    y="Valor Promedio",
                    color="Índice",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

            # Lluvia Diaria
            st.subheader("📉 Distribución Diaria de Precipitación (CHIRPS)")
            if not chirps["df_diario"].empty:
                fig_bar = px.bar(
                    chirps["df_diario"],
                    x="Fecha",
                    y="Lluvia (mm)",
                    template="plotly_dark",
                    color_discrete_sequence=["#38bdf8"]
                )
                fig_bar.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(type='category', tickangle=-45)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")

            pdf_bytes = PDFReportGenerator.generate_pdf(
                stats_data=stats,
                chirps_data=chirps,
                predio_name=res["predio_nombre"],
                cultivo_name=res["tipo_cultivo"],
                coords=res["coords"],
            )

            st.download_button(
                label=f"📄 Generar y Descargar Informe PDF ({res['predio_nombre']})",
                data=pdf_bytes,
                file_name=f"Reporte_Multiespectral_{res['predio_nombre']}.pdf",
                mime="application/pdf",
                key="pdf_download_btn_predio"
            )

    # ==================== PESTAÑA 2: COMPARADOR TEMPORAL (SPLIT-MAP) ====================
    with tab_comparador:
        st.subheader("Análisis Comparativo Multitemporal (Split-Map)")
        st.markdown("Compara visualmente la evolución vegetativa del predio entre dos periodos distintos.")

        col_cmp1, col_cmp2 = st.columns(2)
        with col_cmp1:
            st.markdown("#### 📅 Periodo A (Base)")
            year_a = st.slider("Año Base:", 2017, 2026, 2018)
        with col_cmp2:
            st.markdown("#### 📅 Periodo B (Comparativo)")
            year_b = st.slider("Año Comparativo:", 2017, 2026, 2025)

        if st.button("Ejecutar Comparación Espectacular"):
            with st.spinner("Procesando ambas series temporales en Google Earth Engine..."):
                try:
                    area_estudio = ee.Geometry.Polygon(geojson_geom['coordinates'])

                    comp_a, _ = SatelliteProcessor.process_region(area_estudio, f"{year_a}-01-01", f"{year_a}-12-31",
                                                                  20)
                    comp_b, _ = SatelliteProcessor.process_region(area_estudio, f"{year_b}-01-01", f"{year_b}-12-31",
                                                                  20)

                    dual_map = DualMap(location=coords, zoom_start=14, tiles="CartoDB dark_matter")

                    ndvi_vis = {"min": 0.0, "max": 0.8,
                                "palette": ["FFFFFF", "CE7E45", "DF923D", "FCD163", "99B718", "397D02", "004C00"]}

                    map_id_a = ee.Image(comp_a.select("NDVI")).getMapId(ndvi_vis)
                    map_id_b = ee.Image(comp_b.select("NDVI")).getMapId(ndvi_vis)

                    folium.TileLayer(tiles=map_id_a["tile_fetcher"].url_format, attr="GEE",
                                     name=f"NDVI {year_a}").add_to(dual_map.m1)
                    folium.TileLayer(tiles=map_id_b["tile_fetcher"].url_format, attr="GEE",
                                     name=f"NDVI {year_b}").add_to(dual_map.m2)

                    folium.GeoJson(geojson_geom, style_function=lambda x: {'fillColor': '#00000000', 'color': '#38bdf8',
                                                                           'weight': 2}).add_to(dual_map.m1)
                    folium.GeoJson(geojson_geom, style_function=lambda x: {'fillColor': '#00000000', 'color': '#38bdf8',
                                                                           'weight': 2}).add_to(dual_map.m2)

                    st_folium(dual_map, width=1100, height=500)
                    st.success(f"Comparación renderizada exitosamente entre {year_a} y {year_b}.")
                except Exception as e:
                    st.error(f"Error al generar mapa comparativo: {str(e)}")

    # ==================== PESTAÑA 3: MANN-KENDALL ====================
    with tab_tendencias:
        st.subheader("📈 Modelado Estadístico de Tendencia Histórica (Mann-Kendall / Sen's Slope)")
        st.markdown("Evalúa tendencias climáticas a largo plazo alrededor del predio seleccionado.")

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            analysis_var = st.selectbox(
                "Variable de Análisis:",
                ["Precipitación (CHIRPS)", "Temperaturas (ERA5-Land)", "Sequías / Estrés Hídrico (TerraClimate PDSI)"]
            )
        with col_t2:
            start_yr = st.number_input("Año Inicial", min_value=1981, max_value=2025, value=2005)
        with col_t3:
            end_yr = st.number_input("Año Final", min_value=1982, max_value=2026, value=2025)

        if st.button("Ejecutar Modelado en Región del Predio"):
            with st.spinner("Computando regresión espacial Mann-Kendall / Sen Slope..."):
                try:
                    area_estudio = ee.Geometry.Polygon(geojson_geom['coordinates'])
                    bbox = area_estudio.bounds().getInfo()['coordinates']

                    slope_img, trend_geom = SatelliteProcessor.calculate_mann_kendall_trend(
                        start_year=int(start_yr),
                        end_year=int(end_yr),
                        analysis_type=analysis_var,
                        bbox=bbox
                    )

                    st.session_state["trend_result"] = {
                        "slope_img": slope_img,
                        "var": analysis_var,
                        "start_yr": start_yr,
                        "end_yr": end_yr,
                        "geojson_geom": geojson_geom,
                        "coords": coords
                    }
                    st.success("¡Tendencia calculada correctamente!")
                except Exception as e:
                    st.error(f"Error en el modelado estadístico: {str(e)}")

        if "trend_result" in st.session_state:
            res_trend = st.session_state["trend_result"]

            st.markdown(
                f"#### Mapa de Tendencia Espacial ({res_trend['var']}: {res_trend['start_yr']} - {res_trend['end_yr']})")

            if "Precipitación" in res_trend['var']:
                vis_params = {"min": -10.0, "max": 10.0,
                              "palette": ["d73027", "f46d43", "fdae61", "fee08b", "d9ef8b", "a6d96a", "66bd63",
                                          "1a9850"]}
                unidad = "mm/año"
            elif "Temperaturas" in res_trend['var']:
                vis_params = {"min": -0.05, "max": 0.05,
                              "palette": ["313695", "4575b4", "74add1", "abd9e9", "fee090", "fdae61", "f46d43",
                                          "d73027"]}
                unidad = "°C/año"
            else:
                vis_params = {"min": -0.2, "max": 0.2,
                              "palette": ["8c510a", "bf812d", "dfc27d", "f6e8c3", "c7edd5", "80bfac", "35978f",
                                          "01665e"]}
                unidad = "puntos de índice/año"

            m_trend = folium.Map(location=res_trend["coords"], zoom_start=13, tiles="CartoDB dark_matter")

            map_id_slope = ee.Image(res_trend["slope_img"]).getMapId(vis_params)

            folium.TileLayer(
                tiles=map_id_slope["tile_fetcher"].url_format,
                attr="Google Earth Engine - Trend Slope",
                name="Pendiente de Sen (Sen's Slope)",
                overlay=True,
                control=True
            ).add_to(m_trend)

            folium.GeoJson(
                res_trend["geojson_geom"],
                name="Predio",
                style_function=lambda x: {'fillColor': '#00000000', 'color': '#ffffff', 'weight': 2.5,
                                          'dashArray': '4, 4'}
            ).add_to(m_trend)

            folium.LayerControl().add_to(m_trend)

            col_map_t, col_info_t = st.columns([2, 1])
            with col_map_t:
                st_folium(m_trend, width="100%", height=480)

            with col_info_t:
                st.markdown("##### Leyenda del Análisis")
                st.write(f"**Variable:** {res_trend['var']}")
                st.write(f"**Unidad de cambio:** {unidad}")
                st.markdown(
                    """
                    - 🟩 **Valores Positivos (Verde/Azul):** Tendencia al incremento acumulativo año con año.
                    - 🟥 **Valores Negativos (Rojo/Marrón):** Tendencia a la reducción acumulativa o aumento de degradación.
                    - 🟨 **Cercano a 0:** Sin cambio estacional o tendencia estadísticamente estable.
                    """
                )


if __name__ == "__main__":
    main()