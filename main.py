import os
import json
import sqlite3
import urllib.request
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape

# -----------------------------------------------------------------------------
# IMPORTACIÓN DE SAM 2 (Módulo oficial: sam2.build_sam)
# -----------------------------------------------------------------------------
try:
    from sam2.build_sam import build_sam2
except ModuleNotFoundError:
    try:
        from sam2.build_sam2 import build_sam2
    except ModuleNotFoundError:
        from sam2 import build_sam2

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Configuración inicial de la página en Streamlit
st.set_page_config(page_title="Plataforma Agrícola Demo", layout="wide")

DB_NAME = "predios.db"
CHECKPOINT_PATH = "sam2_hiera_tiny.pt"
MODEL_CFG = "sam2_hiera_t.yaml"
SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"


# -----------------------------------------------------------------------------
# 1. CARGA Y CACHÉ DEL MODELO SAM 2 TINY (RECUPERANDO SENSIBILIDAD Y PRECISIÓN)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_sam2_model():
    """
    Descarga y carga en memoria RAM la variante Tiny de SAM 2 (~75 MB),
    equilibrada para detectar copas pequeñas sin agotar la memoria en CPU.
    """
    device = "cpu"

    if not os.path.exists(CHECKPOINT_PATH):
        with st.spinner("📥 Descargando modelo ligero SAM 2 Tiny (~75 MB)..."):
            try:
                urllib.request.urlretrieve(SAM2_BASE_URL, CHECKPOINT_PATH)
                st.success("✅ Modelo SAM 2 Tiny descargado exitosamente.")
            except Exception as e:
                st.error(f"Error al descargar los pesos del modelo: {e}")
                return None

    try:
        sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=device)
        # Parámetros ajustados para alta sensibilidad en detección de copas
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=24,          # Aumentado a 24 para recuperar árboles/copas pequeñas
            pred_iou_thresh=0.70,        # Umbral más permisivo para capturar más candidatos
            stability_score_thresh=0.80, # Puntuación de estabilidad calibrada
            min_mask_region_area=15      # Permite regiones más pequeñas antes del filtro
        )
        return mask_generator
    except Exception as e:
        st.error(f"Error al instanciar el modelo SAM 2: {e}")
        return None


# -----------------------------------------------------------------------------
# 2. CONEXIÓN A BASE DE DATOS SQLITE (predios.db)
# -----------------------------------------------------------------------------
def obtener_predios_db():
    """Obtiene todos los predios almacenados en la base de datos SQLite predios.db."""
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM predios", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error al leer la base de datos {DB_NAME}: {e}")
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# 3. CONTROL DE SESIÓN Y LOGIN
# -----------------------------------------------------------------------------
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.title("🌾 Plataforma Agrícola Inteligente")
    st.subheader("Acceso a la demo de monitoreo y conteo de cultivos")

    col1, _ = st.columns([1, 2])

    with col1:
        usuario_input = st.text_input("Usuario", value="demo")
        pass_input = st.text_input("Contraseña", type="password", value="demo")

        if st.button("Ingresar a la Plataforma", type="primary"):
            if usuario_input == "demo" and pass_input == "demo":
                st.session_state.autenticado = True
                st.rerun()
            else:
                st.error("Credenciales incorrectas. Usa Usuario: demo y Contraseña: demo")


# -----------------------------------------------------------------------------
# 4. DASHBOARD PRINCIPAL
# -----------------------------------------------------------------------------
else:
    st.sidebar.title("Plataforma Agrícola")
    st.sidebar.markdown("**Módulo Principal de Consulta**")

    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.autenticado = False
        st.rerun()

    tab1, tab2 = st.tabs(["📌 Visión General de Parcela", "🤖 Conteo de Copas con SAM 2"])

    # -------------------------------------------------------------------------
    # TAB 1: Selección y Monitoreo Espacial de Predios desde SQLite (predios.db)
    # -------------------------------------------------------------------------
    with tab1:
        st.header("Monitoreo Espacial de Predios Registrados")

        df_predios = obtener_predios_db()

        if df_predios.empty:
            st.warning(f"⚠️ No se encontraron predios registrados en la base de datos `{DB_NAME}`.")
        else:
            opciones_predios = {
                f"{row['nombre_predio']} ({row.get('tipo_cultivo', 'N/A')}) - {row.get('superficie_ha', 0)} Ha": row['id']
                for _, row in df_predios.iterrows()
            }

            predio_seleccionado_lbl = st.selectbox(
                "🔎 **Selecciona un predio registrado en la base de datos:**",
                options=list(opciones_predios.keys())
            )

            predio_id_sel = opciones_predios[predio_seleccionado_lbl]
            predio_sel = df_predios[df_predios["id"] == predio_id_sel].iloc[0]

            centroide_lat, centroide_lon = 28.405, -106.865
            if "geojson_geom" in predio_sel and predio_sel["geojson_geom"]:
                try:
                    geojson_geom = json.loads(predio_sel["geojson_geom"])
                    geom_shapely = shape(geojson_geom)
                    centroide_lat = geom_shapely.centroid.y
                    centroide_lon = geom_shapely.centroid.x
                except Exception:
                    pass

            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            col_kpi1.metric("Nombre de Predio", predio_sel.get("nombre_predio", "N/A"))
            col_kpi2.metric("Tipo de Cultivo", predio_sel.get("tipo_cultivo", "N/A"))
            col_kpi3.metric("Superficie Registrada", f"{predio_sel.get('superficie_ha', 0)} Ha")

            st.divider()

            st.subheader(f"Vista Satelital: {predio_sel.get('nombre_predio', '')}")

            m = folium.Map(location=[centroide_lat, centroide_lon], zoom_start=15, tiles="Esri WorldImagery")

            for _, row in df_predios.iterrows():
                es_el_seleccionado = (row['id'] == predio_id_sel)
                icon_color = "blue" if es_el_seleccionado else "green"
                icon_name = "star" if es_el_seleccionado else "leaf"

                if "geojson_geom" in row and row["geojson_geom"]:
                    try:
                        geom = json.loads(row["geojson_geom"])
                        shapely_shape = shape(geom)
                        c_lat, c_lon = shapely_shape.centroid.y, shapely_shape.centroid.x

                        folium.GeoJson(
                            geom,
                            name=row.get("nombre_predio", "Predio"),
                            style_function=lambda x, active=es_el_seleccionado: {
                                'fillColor': '#0284c7' if active else '#16a34a',
                                'color': '#0284c7' if active else '#ffffff',
                                'weight': 3 if active else 1.5,
                                'fillOpacity': 0.4 if active else 0.2
                            }
                        ).add_to(m)

                        folium.Marker(
                            location=[c_lat, c_lon],
                            popup=folium.Popup(
                                f"<b>Predio:</b> {row.get('nombre_predio', 'N/A')}<br>"
                                f"<b>Cultivo:</b> {row.get('tipo_cultivo', 'N/A')}<br>"
                                f"<b>Superficie:</b> {row.get('superficie_ha', 0)} Ha",
                                max_width=250
                            ),
                            tooltip=f"{row.get('nombre_predio', 'Predio')} ({row.get('tipo_cultivo', 'N/A')})",
                            icon=folium.Icon(color=icon_color, icon=icon_name)
                        ).add_to(m)
                    except Exception:
                        continue

            st_folium(m, width=1100, height=480, key=f"mapa_predio_{predio_id_sel}")

    # -------------------------------------------------------------------------
    # TAB 2: Módulo de Conteo y Segmentación SAM 2 (Sensibilidad Restaurada)
    # -------------------------------------------------------------------------
    with tab2:
        st.header("Segmentación y Conteo Automatizado con SAM 2")
        st.write("Carga un recorte de ortomosaico o imagen aérea para procesar el inventario de árboles.")

        uploaded_file = st.file_uploader("Cargar imagen aérea (PNG, JPG, TIFF)", type=["jpg", "jpeg", "png", "tif"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            # Dimensión calibrada a 800 px para mayor definición de objetos sin agotar RAM
            max_dim = 800
            if max(image.size) > max_dim:
                image.thumbnail((max_dim, max_dim))

            image_np = np.array(image)

            col_img1, col_img2 = st.columns(2)

            with col_img1:
                st.subheader("Imagen de Entrada")
                st.image(image, use_container_width=True)

            st.divider()
            st.subheader("Filtros de Área para Copas")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                min_area = st.slider("Área mínima de copa (píxeles)", 5, 1000, 15)
            with col_p2:
                max_area = st.slider("Área máxima de copa (píxeles)", 500, 50000, 10000)

            if st.button("Ejecutar Conteo con SAM 2", type="primary"):
                mask_generator = load_sam2_model()

                if mask_generator is not None:
                    with st.spinner("Procesando segmentación en CPU (Alta Precisión)..."):
                        masks = mask_generator.generate(image_np)

                        filtered_masks = [m for m in masks if min_area <= m['area'] <= max_area]
                        count = len(filtered_masks)

                        overlay = image_np.copy()
                        for m in filtered_masks:
                            segmentation = m['segmentation']
                            color = np.array([0, 255, 0], dtype=np.uint8)
                            overlay[segmentation] = overlay[segmentation] * 0.5 + color * 0.5

                            y_indices, x_indices = np.where(segmentation)
                            if len(x_indices) > 0 and len(y_indices) > 0:
                                cx, cy = int(np.mean(x_indices)), int(np.mean(y_indices))
                                cv2.circle(overlay, (cx, cy), 2, (255, 0, 0), -1)

                        with col_img2:
                            st.subheader("Resultado de Segmentación")
                            st.image(overlay, use_container_width=True)

                        st.success(f"Detección finalizada: **{count} plantas/árboles detectados**.")
#export GOOGLE_APPLICATION_CREDENTIALS="credentials.json"
