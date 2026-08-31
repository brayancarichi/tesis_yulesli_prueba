import os
import urllib.request
import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import folium
from streamlit_folium import st_folium

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

# Configuración del modelo SAM 2 Base Plus
CHECKPOINT_PATH = "sam2_hiera_base_plus.pt"
MODEL_CFG = "sam2_hiera_b+.yaml"
SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"


# -----------------------------------------------------------------------------
# 1. CARGA Y CACHÉ DEL MODELO SAM 2 BASE PLUS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_sam2_model():
    """
    Verifica la existencia del archivo de pesos (.pt). Si no está localmente,
    lo descarga. Posteriormente inicializa el modelo ajustado para CPU.
    """
    device = "cpu"

    # Descarga automática del archivo de pesos si no existe en la carpeta raíz
    if not os.path.exists(CHECKPOINT_PATH):
        with st.spinner("📥 Descargando archivo de pesos SAM 2 Base Plus (~320 MB)..."):
            try:
                urllib.request.urlretrieve(SAM2_BASE_URL, CHECKPOINT_PATH)
                st.success("✅ Modelo SAM 2 Base Plus descargado exitosamente.")
            except Exception as e:
                st.error(f"Error al descargar el checkpoint de SAM 2: {e}")
                return None

    # Inicialización del modelo
    try:
        sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=device)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=16,  # Densidad de puntos en rejilla (óptima para CPU)
            pred_iou_thresh=0.80,  # Umbral de calidad de máscara
            stability_score_thresh=0.88,  # Umbral de estabilidad visual
            min_mask_region_area=100  # Filtro de ruido o artefactos muy pequeños
        )
        return mask_generator
    except Exception as e:
        st.error(f"Error al instanciar el modelo SAM 2: {e}")
        return None


# -----------------------------------------------------------------------------
# 2. BASE DE DATOS SIMULADA DE AGRICULTORES
# -----------------------------------------------------------------------------
AGRICULTORES = {
    "AGRO01": {
        "nombre": "Yulesli Guillén",
        "password": "demo",
        "ubicacion": [28.405, -106.865],  # Coordenadas de prueba [lat, lon]
        "cultivo": "Manzano",
        "superficie_ha": 12.5,
        "plantas_estimadas": 3500
    }
}

# -----------------------------------------------------------------------------
# 3. CONTROL DE SESIÓN
# -----------------------------------------------------------------------------
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False
if "usuario" not in st.session_state:
    st.session_state.usuario = None

# -----------------------------------------------------------------------------
# 4. PANTALLA DE ACCESO (LOGIN)
# -----------------------------------------------------------------------------
if not st.session_state.autenticado:
    st.title("🌾 Plataforma Agrícola Inteligente")
    st.subheader("Acceso a la demo de monitoreo y conteo de cultivos")

    col1, _ = st.columns([1, 2])

    with col1:
        id_input = st.text_input("ID de Agricultor", placeholder="Ej. AGRO01")
        pass_input = st.text_input("Contraseña", type="password", value="demo")

        if st.button("Ingresar a la Plataforma", type="primary"):
            if id_input in AGRICULTORES and AGRICULTORES[id_input]["password"] == pass_input:
                st.session_state.autenticado = True
                st.session_state.usuario = AGRICULTORES[id_input]
                st.rerun()
            else:
                st.error("Credenciales incorrectas. Usa ID: AGRO01 y Contraseña: demo")


# -----------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL DEL AGRICULTOR
# -----------------------------------------------------------------------------
else:
    user = st.session_state.usuario

    # Barra lateral
    st.sidebar.title(f"Bienvenido, {user['nombre']}")
    st.sidebar.markdown(f"**Cultivo:** {user['cultivo']}")
    st.sidebar.markdown(f"**Superficie:** {user['superficie_ha']} Ha")

    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.autenticado = False
        st.session_state.usuario = None
        st.rerun()

    # Pestañas principales
    tab1, tab2 = st.tabs(["📌 Visión General de Parcela", "🤖 Conteo de Copas con SAM 2"])

    # -------------------------------------------------------------------------
    # TAB 1: Mapa Satelital
    # -------------------------------------------------------------------------
    with tab1:
        st.header("Monitoreo Espacial de Parcela")

        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("Tipo de Cultivo", user["cultivo"])
        col_kpi2.metric("Superficie Registrada", f"{user['superficie_ha']} Ha")
        col_kpi3.metric("Plantas Teóricas", f"{user['plantas_estimadas']} unidades")

        st.divider()

        st.subheader("Ubicación Satelital de la Parcela")
        m = folium.Map(location=user["ubicacion"], zoom_start=16, tiles="Esri WorldImagery")
        folium.Marker(
            user["ubicacion"],
            popup=f"Parcela de {user['nombre']}",
            tooltip=user["cultivo"],
            icon=folium.Icon(color="green", icon="leaf")
        ).add_to(m)

        st_folium(m, width=1100, height=450)

    # -------------------------------------------------------------------------
    # TAB 2: Módulo de Conteo y Segmentación SAM 2
    # -------------------------------------------------------------------------
    with tab2:
        st.header("Segmentación y Conteo Automatizado con SAM 2")
        st.write("Carga un recorte de ortomosaico o imagen aérea para procesar el inventario de árboles.")

        uploaded_file = st.file_uploader("Cargar imagen aérea (PNG, JPG, TIFF)", type=["jpg", "jpeg", "png", "tif"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            # Redimensión defensiva para acelerar la procesación en CPU si la imagen es gigante
            max_dim = 1280
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
                min_area = st.slider("Área mínima de copa (píxeles)", 50, 5000, 200)
            with col_p2:
                max_area = st.slider("Área máxima de copa (píxeles)", 3000, 100000, 20000)

            if st.button("Ejecutar Conteo con SAM 2", type="primary"):
                mask_generator = load_sam2_model()

                if mask_generator is not None:
                    with st.spinner("Procesando segmentación en CPU con SAM 2..."):
                        masks = mask_generator.generate(image_np)

                        # Filtrado según los umbrales de área seleccionados
                        filtered_masks = [m for m in masks if min_area <= m['area'] <= max_area]
                        count = len(filtered_masks)

                        # Generar overlay visual (máscaras en verde semi-transparente y centroides en azul)
                        overlay = image_np.copy()
                        for m in filtered_masks:
                            segmentation = m['segmentation']

                            # Tinte verde
                            color = np.array([0, 255, 0], dtype=np.uint8)
                            overlay[segmentation] = overlay[segmentation] * 0.5 + color * 0.5

                            # Punto central
                            y_indices, x_indices = np.where(segmentation)
                            if len(x_indices) > 0 and len(y_indices) > 0:
                                cx, cy = int(np.mean(x_indices)), int(np.mean(y_indices))
                                cv2.circle(overlay, (cx, cy), 3, (255, 0, 0), -1)

                        # Mostrar resultado de la segmentación
                        with col_img2:
                            st.subheader("Resultado de Segmentación")
                            st.image(overlay, use_container_width=True)

                        st.success(f"Detección finalizada: **{count} plantas/árboles detectados**.")