import os
from pathlib import Path
from datetime import date, datetime
from getpass import getuser
import io

import streamlit as st
import numpy as np
import imutils
from PIL import Image
import cv2
import torch
import supervision as sv
from ultralytics import YOLO

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y MODELO PARA ZARZAMORAS
# ==============================================================================
st.set_page_config(
    page_title="Detección de Zarzamoras",
    layout="wide"
)

# Ruta donde se encuentra tu modelo entrenado para zarzamoras (.pt)
MODELO_ZARZAMORA_PATH = "bestZarzamora.pt"  # Cambia esta ruta si tu archivo tiene otro nombre o ubicación


# ==============================================================================
# FUNCIONES DE DETECCIÓN Y PROCESAMIENTO
# ==============================================================================
@st.cache_resource
def cargar_modelo_yolo(ruta_modelo):
    """Carga y almacena en caché el modelo YOLO para optimizar rendimiento."""
    if not os.path.exists(ruta_modelo):
        # Si no existe en la ruta específica, intenta buscar un .pt local genérico
        fallback = "bestZarzamora.pt"
        if os.path.exists(fallback):
            ruta_modelo = fallback
        else:
            return None
    return YOLO(ruta_modelo)


def deteccion_zarzamora_imagen(image, model):
    """
    Procesa la imagen para la detección de anomalías/frutos en cultivos de zarzamora
    y devuelve la imagen anotada redimensionada.
    """
    # Convertir PIL Image a formato RGB/NumPy si es necesario para YOLO
    imagen_np = np.array(image)

    # Inferencia con YOLO ajustada para el cultivo
    result = model(imagen_np, imgsz=640, conf=0.1, show_labels=False, show_conf=False)[0]
    resultados = model.predict(imagen_np, imgsz=640, conf=0.1)

    anotaciones = resultados[0].plot()
    imagen_redimensionada = imutils.resize(anotaciones, width=1024)

    return imagen_redimensionada


def contar_detecciones_zarzamora(image, model):
    """
    Calcula la cantidad total de detecciones válidas encontradas en la zarzamora.
    """
    imagen_np = np.array(image)
    result = model(imagen_np, imgsz=640, conf=0.1, show_labels=False, show_conf=False)[0]
    resultados = model.predict(imagen_np, imgsz=640, conf=0.1)

    detections = sv.Detections.from_ultralytics(result)
    alta = detections[detections.confidence > 0.1]

    cantidad_detectada = len(alta)
    return str(cantidad_detectada)


# ==============================================================================
# APLICACIÓN PRINCIPAL (STREAMLIT)
# ==============================================================================
def main():
    st.header('Detección de Anomalías y Conteo en Zarzamoras')
    st.markdown(
        'Esta aplicación utiliza un modelo de red neuronal entrenado específicamente para la detección '
        'de anomalías, desarrollo o conteo en cultivos de **zarzamora**. Sube una imagen aérea o de campo '
        'para realizar el análisis automatizado.'
    )

    # Cargar el modelo adaptado
    model = cargar_modelo_yolo(MODELO_ZARZAMORA_PATH)

    if model is None:
        st.error(
            f"⚠️ No se encontró el archivo de pesos del modelo en la ruta: `{MODELO_ZARZAMORA_PATH}`. Asegúrate de colocar tu archivo `.pt` entrenado.")
        return

    file_uploader = st.file_uploader('Sube tu imagen en formato:', type=['jpg', 'jpeg', 'png'])

    if file_uploader is not None:
        image = Image.open(file_uploader).convert("RGB")

        st.subheader("Imagen Original Cargada")
        st.image(image, use_container_width=True)

        with st.spinner("Procesando imagen con el modelo de zarzamoras..."):
            imagen_procesada = deteccion_zarzamora_imagen(image, model)
            total_anomalias = contar_detecciones_zarzamora(image, model)

        st.markdown('### Resultados del Análisis')
        st.markdown('Las zonas detectadas se marcan sobre la imagen analizada para su inspección visual.')
        st.image(imagen_procesada, use_container_width=True)

        st.success(f"📊 **{total_anomalias}** elementos en el cultivo de zarzamora.")


if __name__ == "__main__":
    main()