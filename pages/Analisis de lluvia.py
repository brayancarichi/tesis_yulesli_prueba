"""
PÁGINA INSTITUCIONAL: EVALUACIÓN DE RIESGO Y ANOMALÍA AGROCLIMÁTICA POR PREDIO
Módulo de análisis satelital CHIRPS basado en los polígonos guardados en SQLite (predios.db)
con generación de informes ejecutivos en PDF en lenguaje accesible.
"""

import os
import json
import sqlite3
import io
import numpy as np
import pandas as pd
import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from shapely.geometry import shape

# Importaciones para generación de PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Evaluación Agroclimática por Predio | Plataforma Agrícola",
    layout="wide"
)

st.markdown(
    """
    <style>
        .texto-dinamico {
            font-size: 16px;
            background-color: #f8fafc;
            padding: 18px;
            border-left: 5px solid #0284c7;
            border-radius: 6px;
            color: #0f172a;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .explicacion-sencilla {
            font-size: 15px;
            color: #64748b;
            margin-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DB_NAME = "predios.db"


# ==============================================================================
# 1. AUTENTICACIÓN GOOGLE EARTH ENGINE
# ==============================================================================
def initialize_earth_engine() -> bool:
    if st.session_state.get("gee_initialized", False):
        return True

    gee_json_str = os.getenv("GEE_JSON_KEY")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    try:
        if gee_json_str:
            if isinstance(gee_json_str, str):
                cleaned_str = gee_json_str.strip()
                key_content = json.loads(cleaned_str, strict=False)
            else:
                key_content = gee_json_str

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
# 2. FUNCIONES DE BASE DE DATOS Y CÁLCULOS
# ==============================================================================
def obtener_predios_db():
    """Obtiene todos los predios almacenados en SQLite."""
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM predios", conn)
    conn.close()
    return df


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


def limpiar_cache_resultados():
    """Limpia el estado guardado al modificar parámetros en la barra lateral."""
    if "evaluacion_results" in st.session_state:
        del st.session_state["evaluacion_results"]


# ==============================================================================
# 3. GENERADOR DE PDF EN LENGUAJE ACCESIBLE
# ==============================================================================
def generar_pdf_reporte(res_dict, fecha_inicio, fecha_fin):
    """Genera un archivo PDF con formato limpio y etiquetas HTML válidas para ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Estilos personalizados
    estilo_titulo = ParagraphStyle(
        'TituloReporte',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0284c7"),
        spaceAfter=10
    )

    estilo_subtitulo = ParagraphStyle(
        'SubtituloReporte',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=6
    )

    estilo_texto = ParagraphStyle(
        'TextoBase',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # Encabezado
    story.append(Paragraph("Reporte de Diagnóstico Agroclimático", estilo_titulo))
    story.append(Paragraph(f"<b>Predio:</b> {res_dict['nombre_predio']} | <b>Cultivo:</b> {res_dict['tipo_cultivo']}",
                           estilo_texto))
    story.append(Paragraph(
        f"<b>Periodo analizado:</b> Del {fecha_inicio.strftime('%d/%m/%Y')} al {fecha_fin.strftime('%d/%m/%Y')}",
        estilo_texto))
    story.append(Spacer(1, 12))

    # Resumen Ejecutivo
    story.append(Paragraph("1. Diagnóstico del Terreno", estilo_subtitulo))

    explicacion_pdf = f"""
    Durante este periodo, en su parcela llovieron <b>{res_dict['promedio_mm']:.1f} milímetros</b> (litros por metro cuadrado). 
    En años normales, para estas mismas fechas acostumbran llover unos <b>{res_dict['promedio_historico_mm']:.1f} milímetros</b>.<br/><br/>
    Esto significa que ha llovido un <b>{res_dict['anomalia_porcentaje']:+.1f}%</b> en comparación con el promedio de los últimos 20 años.<br/>
    Además, se registró una racha de <b>{res_dict['racha_seca']} días consecutivos sin lluvias significativas</b>.<br/><br/>
    <b>Estado del predio:</b> <font color="{res_dict['color_alerta']}"><b>{res_dict['estado_siniestro']}</b></font>. {res_dict['explicacion_sencilla_texto']}
    """
    story.append(Paragraph(explicacion_pdf, estilo_texto))
    story.append(Spacer(1, 12))

    # Tabla de métricas principales
    story.append(Paragraph("2. Resumen Numérico", estilo_subtitulo))
    datos_tabla = [
        ["Indicador", "Valor Registrado", "Qué significa para su terreno"],
        ["Lluvia acumulada", f"{res_dict['promedio_mm']:.1f} mm", "Lluvia total recibida en el periodo"],
        ["Promedio histórico", f"{res_dict['promedio_historico_mm']:.1f} mm", "Lo que suele llover en 20 años"],
        ["Variación de lluvia", f"{res_dict['anomalia_porcentaje']:+.1f}%", "Diferencia respecto al histórico"],
        ["Días sin lluvia", f"{res_dict['racha_seca']} días", "Días seguidos sin lluvia útil (<1 mm)"]
    ]

    t = Table(datos_tabla, colWidths=[140, 110, 260])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    # Recomendación Final
    story.append(Paragraph("3. Recomendación para el Productor", estilo_subtitulo))
    story.append(Paragraph(res_dict['recomendacion'], estilo_texto))
    story.append(Spacer(1, 15))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ==============================================================================
# 4. INTERFAZ PRINCIPAL
# ==============================================================================
def main():
    st.title("🌧️ Evaluación Agroclimática de Predios Registrados")
    st.markdown(
        "<p class='explicacion-sencilla'>Monitoreo satelital de precipitación y análisis de humedad "
        "calculado directamente sobre el polígono de cada parcela, en un lenguaje sencillo y accesible.</p>",
        unsafe_allow_html=True
    )

    if not initialize_earth_engine():
        st.stop()

    df_predios = obtener_predios_db()

    if df_predios.empty:
        st.warning(
            "⚠️ No se encontraron predios guardados en la base de datos. Ve a la página 'Dibuja tu Predio' para registrar uno primero.")
        st.stop()

    # Selección de Predio desde la BD
    st.sidebar.header("Parámetros de Evaluación")

    opciones_predios = {
        f"{row['nombre_predio']} ({row['tipo_cultivo']}) - {row['superficie_ha']} Ha": row['id']
        for _, row in df_predios.iterrows()
    }
    predio_seleccionado_label = st.sidebar.selectbox(
        "Selecciona un Predio Guardado:",
        options=list(opciones_predios.keys()),
        on_change=limpiar_cache_resultados
    )
    predio_id = opciones_predios[predio_seleccionado_label]

    # Extraer fila seleccionada
    predio_data = df_predios[df_predios["id"] == predio_id].iloc[0]

    st.sidebar.markdown("---")
    st.sidebar.write("Periodo de Análisis")

    fecha_fin = st.sidebar.date_input(
        "Fecha final",
        value=date.today() - timedelta(days=5),
        on_change=limpiar_cache_resultados
    )
    fecha_inicio = st.sidebar.date_input(
        "Fecha inicial",
        value=fecha_fin - timedelta(days=30),
        on_change=limpiar_cache_resultados
    )

    if fecha_inicio >= fecha_fin:
        st.sidebar.error("La fecha inicial debe ser anterior a la fecha final.")
        st.stop()

    btn_consultar = st.sidebar.button("Generar Informe Técnico", type="primary", use_container_width=True)

    if btn_consultar:
        limpiar_cache_resultados()

        with st.spinner(f"Analizando terreno '{predio_data['nombre_predio']}' mediante satélite..."):
            try:
                # 1. Convertir GeoJSON de SQLite a ee.Geometry
                geojson_geom = json.loads(predio_data['geojson_geom'])
                coords_geojson = geojson_geom['coordinates']
                area_estudio = ee.Geometry.Polygon(coords_geojson)
                centroide_geom = area_estudio.centroid(maxError=1)

                # Obtener centroide en Shapely para centrar el mapa Folium
                geom_shapely = shape(geojson_geom)
                centroide_lat = geom_shapely.centroid.y
                centroide_lon = geom_shapely.centroid.x

                # 2. Precipitación del periodo actual (CHIRPS Daily)
                coleccion_lluvia = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
                    .filterDate(fecha_inicio.strftime("%Y-%m-%d"), fecha_fin.strftime("%Y-%m-%d"))

                lluvia_acumulada = coleccion_lluvia.sum().clip(area_estudio)

                # Reducción con fallback a centroide para predios pequeños
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

                # 3. Promedio Histórico (Referencia de 20 años)
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

                # 4. Serie de tiempo diaria adaptada a predios pequeños (Uso del Centroide + bestEffort)
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

                racha_seca = calcular_dias_secos_consecutivos(df_diario, umbral_mm=1.0)

                # ==============================================================
                # LÓGICA DE DIAGNÓSTICO COHERENTE Y REVISADA
                # ==============================================================
                if anomalia_porcentaje <= -30.0 or (anomalia_porcentaje < 0 and racha_seca >= 18):
                    estado_siniestro = "Riesgo Alto por Sequía"
                    color_alerta = "#dc2626"
                    explicacion_sencilla_texto = "Ha llovido menos de lo habitual y existe una racha seca prolongada. Su cultivo podría presentar estrés hídrico si no cuenta con riego de auxilio."
                    recomendacion = "Revisar la humedad del suelo a profundidad y programar riegos de auxilio si no hay disponibilidad de lluvia."

                elif anomalia_porcentaje > 25.0:
                    estado_siniestro = "Alerta por Exceso de Humedad"
                    color_alerta = "#0284c7"
                    explicacion_sencilla_texto = "Ha llovido considerablemente más de lo habitual para estas fechas."
                    recomendacion = "Verificar el correcto drenaje de los surcos/parcela y monitorear la posible aparición de enfermedades fúngicas."

                elif -10.0 <= anomalia_porcentaje <= 25.0:
                    if racha_seca >= 15:
                        estado_siniestro = "Condición Normal con Distribución Irregular"
                        color_alerta = "#d97706"
                        explicacion_sencilla_texto = "El volumen total de agua fue adecuado, pero la lluvia se concentró en pocos días seguidos de varios días secos."
                        recomendacion = "Monitorear la humedad superficial del suelo debido a los días consecutivos sin lluvia."
                    else:
                        estado_siniestro = "Condición Normal de Lluvia"
                        color_alerta = "#16a34a"
                        explicacion_sencilla_texto = "La humedad acumulada por lluvias se mantiene dentro o ligeramente por encima de los valores promedio normales."
                        recomendacion = "Mantener las labores del cultivo de manera habitual."

                else:  # Entre -30% y -10%
                    estado_siniestro = "Riesgo Moderado por Lluvia Baja"
                    color_alerta = "#d97706"
                    explicacion_sencilla_texto = "Las lluvias han sido ligeramente inferiores a lo normal."
                    recomendacion = "Monitorear el estado del follaje en las horas de mayor calor y planificar riegos estratégicos."

                texto_explicativo = f"""
                <div class="texto-dinamico">
                    <strong>Estado del Predio: {predio_data['nombre_predio']}</strong> ({predio_data['tipo_cultivo']})<br><br>
                    Durante los últimos días se registraron <b>{promedio_mm:.1f} mm</b> de lluvia en su parcela. 
                    El promedio de los últimos 20 años para estas fechas es de <b>{promedio_historico_mm:.1f} mm</b>.<br>
                    Variación: <b>{anomalia_porcentaje:+.1f}% respecto a un año normal</b>.<br><br>
                    Diagnóstico: <span style="color:{color_alerta}; font-weight:bold;">{estado_siniestro}</span>. {explicacion_sencilla_texto}
                </div>
                """

                paleta_colores = ['#ffffff', '#c6dbef', '#6baed6', '#2171b5', '#08306b']
                max_lluvia_mapa = max(promedio_mm * 1.5, 20)

                parametros_visuales = {'min': 0, 'max': max_lluvia_mapa, 'palette': paleta_colores}
                map_id = ee.Image(lluvia_acumulada).getMapId(parametros_visuales)

                st.session_state["evaluacion_results"] = {
                    "texto_html": texto_explicativo,
                    "centroide": [centroide_lat, centroide_lon],
                    "geojson_geom": geojson_geom,
                    "tile_url": map_id["tile_fetcher"].url_format,
                    "paleta": paleta_colores,
                    "max_val": max_lluvia_mapa,
                    "promedio_mm": promedio_mm,
                    "promedio_historico_mm": promedio_historico_mm,
                    "anomalia_porcentaje": anomalia_porcentaje,
                    "racha_seca": racha_seca,
                    "estado_siniestro": estado_siniestro,
                    "color_alerta": color_alerta,
                    "explicacion_sencilla_texto": explicacion_sencilla_texto,
                    "recomendacion": recomendacion,
                    "df_diario": df_diario,
                    "nombre_predio": predio_data['nombre_predio'],
                    "tipo_cultivo": predio_data['tipo_cultivo'],
                    "fecha_inicio": fecha_inicio,
                    "fecha_fin": fecha_fin
                }

            except Exception as e:
                st.error(f"Ocurrió un error al procesar el reporte satelital: {str(e)}")

    # RENDERIZADO DE RESULTADOS
    if "evaluacion_results" in st.session_state:
        res = st.session_state["evaluacion_results"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lluvia Registrada", f"{res['promedio_mm']:.1f} mm",
                    help="Total de litros de agua cayeron por metro cuadrado")
        col2.metric("Lluvia Normal (Histórica)", f"{res['promedio_historico_mm']:.1f} mm",
                    help="Promedio acumulado en los últimos 20 años")
        col3.metric("Diferencia", f"{res['anomalia_porcentaje']:+.1f}%",
                    help="Porcentaje de más o de menos en comparación con un año normal")
        col4.metric("Días Consecutivos Secos", f"{res['racha_seca']} Días",
                    help="Días seguidos sin lluvia aprovechable")

        st.markdown("---")
        st.markdown(res["texto_html"], unsafe_allow_html=True)

        col_mapa, col_grafico = st.columns([1.1, 1])

        with col_mapa:
            st.subheader("Mapa de Lluvia sobre el Predio")
            mapa = folium.Map(location=res["centroide"], zoom_start=14, tiles="Esri WorldImagery")

            folium.TileLayer(
                tiles=res["tile_url"],
                attr="Google Earth Engine - CHIRPS",
                name="Precipitación Acumulada",
                overlay=True,
                control=True,
                opacity=0.75
            ).add_to(mapa)

            folium.GeoJson(
                res["geojson_geom"],
                name="Polígono de Predio",
                style_function=lambda x: {
                    'fillColor': '#00000000',
                    'color': '#ff0033',
                    'weight': 3
                }
            ).add_to(mapa)

            colormap = cm.LinearColormap(
                colors=res["paleta"],
                vmin=0,
                vmax=res["max_val"],
                caption='Lluvia Acumulada (mm)'
            )
            colormap.add_to(mapa)

            st_folium(mapa, width="100%", height=380, key="mapa_predio_evaluacion")

        with col_grafico:
            st.subheader("Comparativa contra un Año Normal")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res['promedio_mm'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Lluvia Actual (mm) vs Normal (Línea Roja)"},
                gauge={
                    'axis': {'range': [None, max(res['promedio_historico_mm'] * 1.5, res['promedio_mm'] + 10)]},
                    'bar': {'color': "#0284c7"},
                    'steps': [
                        {'range': [0, res['promedio_historico_mm'] * 0.6], 'color': "#fee2e2"},
                        {'range': [res['promedio_historico_mm'] * 0.6, res['promedio_historico_mm']],
                         'color': "#fef3c7"},
                        {'range': [res['promedio_historico_mm'], res['promedio_historico_mm'] * 2], 'color': "#dcfce7"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': res['promedio_historico_mm']
                    }
                }
            ))
            fig_gauge.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Lluvia registrada día por día")

        if not res["df_diario"].empty:
            max_y = max(res["df_diario"]["Lluvia (mm)"].max() * 1.25, 2.0)
            fig_bar = px.bar(
                res["df_diario"],
                x="Fecha",
                y="Lluvia (mm)",
                title="Lluvia diaria (milímetros)",
                text_auto='.1f',
                template="plotly_white"
            )
            fig_bar.update_traces(marker_color='#0284c7', textposition='outside')
            fig_bar.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(type='category', tickangle=-45),
                yaxis=dict(title="Precipitación (mm)", range=[0, max_y])
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No se encontraron registros diarios para el rango de fechas seleccionado.")

        st.markdown("---")

        # Generar el PDF en memoria
        pdf_bytes = generar_pdf_reporte(res, res["fecha_inicio"], res["fecha_fin"])
        csv_data = res["df_diario"].to_csv(index=False).encode('utf-8')

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            st.download_button(
                label="📄 Descargar Informe Imprimible (PDF)",
                data=pdf_bytes,
                file_name=f"Informe_Agroclimatico_{res['nombre_predio']}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

        with col_btn2:
            st.download_button(
                label="📊 Descargar Tabla de Datos (CSV)",
                data=csv_data,
                file_name=f"datos_lluvia_{res['nombre_predio']}.csv",
                mime="text/csv",
                type="secondary",
                use_container_width=True
            )

    else:
        st.info("Selecciona un predio guardado en el menú lateral y presiona 'Generar Informe Técnico'.")


if __name__ == "__main__":
    main()