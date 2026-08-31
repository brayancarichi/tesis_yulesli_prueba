import sqlite3
import json
import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape, Polygon
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Dibuja tu Predio", layout="wide")

# -----------------------------------------------------------------------------
# 1. GESTIÓN DE BASE DE DATOS SQLITE (predios.db)
# -----------------------------------------------------------------------------
DB_NAME = "predios.db"


def init_db():
    """Inicializa la tabla de predios si no existe."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_predio TEXT NOT NULL,
            tipo_cultivo TEXT NOT NULL,
            variedad TEXT,
            sistema_riego TEXT,
            superficie_ha REAL NOT NULL,
            geojson_geom TEXT NOT NULL,
            fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def guardar_predio(nombre, cultivo, variedad, riego, superficie, geojson_str):
    """Guarda un nuevo predio en la base de datos SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predios (nombre_predio, tipo_cultivo, variedad, sistema_riego, superficie_ha, geojson_geom)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (nombre, cultivo, variedad, riego, superficie, geojson_str))
    conn.commit()
    conn.close()


def obtener_predios():
    """Recupera la lista de predios registrados."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(
        "SELECT id, nombre_predio, tipo_cultivo, variedad, sistema_riego, superficie_ha, fecha_registro FROM predios",
        conn)
    conn.close()
    return df


# Inicializar BD al cargar el módulo
init_db()


# -----------------------------------------------------------------------------
# 2. CÁLCULO GEODÉSICO DE ÁREA EN HECTÁREAS
# -----------------------------------------------------------------------------
def calcular_area_hectareas(geojson_geometry):
    """
    Calcula el área aproximada en hectáreas a partir del objeto GeoJSON Dibujado,
    corrigiendo la distorsión por latitud (Aproximación por Grado a Metros).
    """
    try:
        geom = shape(geojson_geometry)
        if not isinstance(geom, Polygon):
            return 0.0

        # Centroide para calcular factor de corrección por latitud
        lat_rad = np.radians(geom.centroid.y)

        # Conversión de grados a metros (aprox 111,320m por grado en latitud)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(lat_rad)

        # Transformación de coordenadas a metros cuadrados aproximados
        coords = list(geom.exterior.coords)
        coords_m = [(x * m_per_deg_lon, y * m_per_deg_lat) for x, y in coords]
        polygon_m = Polygon(coords_m)

        area_m2 = polygon_m.area
        area_ha = area_m2 / 10000.0
        return round(area_ha, 3)
    except Exception as e:
        return 0.0


import numpy as np

# -----------------------------------------------------------------------------
# 3. INTERFAZ DE USUARIO (STREAMLIT)
# -----------------------------------------------------------------------------
st.title("🗺️ Delimitación y Registro de Predio")
st.markdown("Utiliza las herramientas de dibujo a la izquierda del mapa para trazar el polígono de tu parcela.")

col_map, col_form = st.columns([2, 1])

# -----------------------------------------------------------------------------
# MAPA INTERACTIVO CON HERRAMIENTAS DE DIBUJO (Folium Draw)
# -----------------------------------------------------------------------------
with col_map:
    st.subheader("Dibujar Polígono")

    # Coordenadas por defecto (Centro de Chihuahua, Mex / Ajustable)
    m = folium.Map(location=[28.405, -106.865], zoom_start=14, tiles="Esri WorldImagery")

    # Agregar plugin de dibujo de Folium (Draw)
    folium.plugins.Draw(
        export=True,
        filename='predio.geojson',
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': True, 'remove': True}
    ).add_to(m)

    # Renderizar el mapa en Streamlit y capturar eventos de dibujo
    map_data = st_folium(m, width=750, height=500, key="mapa_dibujo")

# -----------------------------------------------------------------------------
# FORMULARIO Y CÁLCULOS
# -----------------------------------------------------------------------------
with col_form:
    st.subheader("Datos del Predio")

    area_ha = 0.0
    geometria_json = None

    # Verificar si el usuario ya dibujó un objeto en el mapa
    if map_data and map_data.get("all_drawings"):
        drawings = map_data["all_drawings"]
        if len(drawings) > 0:
            # Tomar el último polígono dibujado
            geometria_json = drawings[-1]["geometry"]
            area_ha = calcular_area_hectareas(geometria_json)

    st.metric("Superficie Calculada", f"{area_ha} Ha", delta="Calculado automáticamente" if area_ha > 0 else None)

    if area_ha == 0.0:
        st.info("👈 Dibuja un polígono o rectángulo en el mapa para calcular la superficie.")

    # Formulario de atributos agrícolas
    with st.form("form_registro_predio"):
        nombre_predio = st.text_input("Nombre del Predio / Parcela", placeholder="Ej. Lote San José 2")

        cultivo = st.selectbox(
            "Tipo de Cultivo",
            ["Manzano", "Nogal", "Maíz", "Frijol", "Alfalfa", "Chile", "Otro"]
        )

        variedad = st.text_input("Variedad / Híbrido", placeholder="Ej. Golden Delicious")

        riego = st.selectbox(
            "Sistema de Riego",
            ["Goteo", "Microaspersión", "Pivote Central", "Rodado / Gravedad", "Temporal"]
        )

        btn_guardar = st.form_submit_button("💾 Guardar Predio en BD", type="primary")

    if btn_guardar:
        if not nombre_predio:
            st.error("Por favor, ingresa un nombre para el predio.")
        elif area_ha == 0.0 or geometria_json is None:
            st.error("Debes dibujar el polígono de tu predio en el mapa antes de guardar.")
        else:
            geojson_str = json.dumps(geometria_json)
            guardar_predio(nombre_predio, cultivo, variedad, riego, area_ha, geojson_str)
            st.success(f"¡Predio **{nombre_predio}** ({area_ha} Ha) registrado correctamente en SQLite!")

st.divider()

# -----------------------------------------------------------------------------
# 4. CONSULTA Y REGISTRO DE PREDIOS ALMACENADOS
# -----------------------------------------------------------------------------
st.subheader("📋 Predios Registrados en la Base de Datos")
df_predios = obtener_predios()

if not df_predios.empty:
    st.dataframe(df_predios, use_container_width=True)
else:
    st.write("Aún no hay predios guardados en `predios.db`.")