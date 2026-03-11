import streamlit as st
import pandas as pd
from google import genai
import os
from core.procesador_datos import procesar_df_a_json

st.set_page_config(page_title="Archego | Auditor B2B", page_icon="⚡", layout="wide")
API_KEY = os.environ.get("GEMINI_API_KEY", "TU_API_KEY_AQUI")

def cargar_prompt(nombre_archivo):
    # 1. Obtenemos la ruta absoluta de la carpeta donde vive app.py
    directorio_base = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construimos la ruta dinámica a la carpeta prompts
    ruta = os.path.join(directorio_base, "prompts", f"{nombre_archivo}.txt")
    
    with open(ruta, "r", encoding="utf-8") as archivo:
        return archivo.read()

AGENTES = {
    "Arquitecto ML (Datos Médicos)": "experto_medicina",
    "Ingeniero NLP (Text Mining)": "experto_argumentacion"
}

st.title("⚡ Archego | Auditor de Datasets")
st.markdown("Auditor de IA estricto para modelos centralizados y descentralizados.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Configuración")
    agente_seleccionado = st.selectbox("Especialidad del Agente:", list(AGENTES.keys()))
    
    archivos_csv = st.file_uploader("Dataset(s) (.csv)", type=["csv"], accept_multiple_files=True)
    columna_objetivo = st.text_input("Columna Objetivo (Obligatorio):").strip()
    columna_nodo_manual = st.text_input("Columna Nodo (Opcional - Para 1 archivo):").strip()
    
    boton_auditar = st.button("Ejecutar Auditoría", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Reporte y Visualización")
    if boton_auditar:
        if not archivos_csv:
            st.warning("Falta subir al menos un archivo CSV.")
            st.stop()
        if not columna_objetivo:
            st.warning("Falta indicar la columna objetivo.")
            st.stop()
            
        try:
            lista_dfs = []
            for archivo in archivos_csv:
                df_temp = pd.read_csv(archivo)
                if len(archivos_csv) > 1:
                    df_temp['origen_archivo_automatico'] = archivo.name
                lista_dfs.append(df_temp)
                
            df = pd.concat(lista_dfs, ignore_index=True)
            
            columna_nodo_final = 'origen_archivo_automatico' if len(archivos_csv) > 1 else columna_nodo_manual

            if columna_objetivo not in df.columns:
                st.error(f"❌ La columna '{columna_objetivo}' no existe en los datos.")
                st.stop()
                
            st.markdown("### 📊 Distribución Global de la Variable Objetivo")
            st.bar_chart(df[columna_objetivo].value_counts(), color="#FF4B4B")
            
            with st.spinner("🤖 Analizando topología y consultando al Agente..."):
                json_estadisticas = procesar_df_a_json(df, columna_objetivo, columna_nodo_final)
                prompt_maestro = cargar_prompt(AGENTES[agente_seleccionado])
                prompt_final = f"{prompt_maestro}\n\nDATOS DEL DATASET:\n{json_estadisticas}"
                
                client = genai.Client(api_key=API_KEY)
                respuesta = client.models.generate_content(
                    model='models/gemini-2.5-flash',
                    contents=prompt_final
                )
                
                st.markdown("---")
                st.markdown(respuesta.text)
                st.markdown("---")
                
                with st.expander("Ver JSON estadístico crudo (Enviado a la IA)"):
                    st.code(json_estadisticas, language="json")
                    
        except Exception as e:

            st.error(f"Falla crítica en el procesamiento: {e}")

