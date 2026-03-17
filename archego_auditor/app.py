import streamlit as st
import pandas as pd
from google import genai
import os
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

def renderizar_graficos_auditoria(json_string, df, columna_objetivo):
    """Función modular para renderizar topología visual sin recargar a Gemini"""
    try:
        resumen = json.loads(json_string)
        analisis = resumen.get("analisis_predictivo", {})
    except json.JSONDecodeError:
        st.error("Error al decodificar el JSON estadístico para los gráficos.")
        return

    st.markdown("---")
    st.header("📊 Topología Visual del Dataset")

    # Usamos columnas anidadas dentro de la columna 2 principal
    col_izq, col_der = st.columns(2)

    # 1. RADAR DE RELEVANCIA (Información Mutua)
    with col_izq:
        mi_data = analisis.get("informacion_mutua_relevante", {})
        if mi_data:
            st.subheader("🎯 Radar de Relevancia Predictiva")
            df_mi = pd.DataFrame(list(mi_data.items()), columns=["Variable", "Información Mutua"]).sort_values(by="Información Mutua", ascending=False)
            st.bar_chart(df_mi.set_index("Variable"), color="#2ecc71")
        else:
            st.info("No se detectó Información Mutua significativa (> 0.05).")

    # 2. SEMÁFORO DE TOXICIDAD (Multicolinealidad VIF)
    with col_der:
        vif_data = analisis.get("multicolinealidad_vif_severa", {})
        if vif_data:
            st.subheader("⚠️ Toxicidad Estructural (VIF > 10)")
            df_vif = pd.DataFrame(list(vif_data.items()), columns=["Variable", "Valor VIF"]).sort_values(by="Valor VIF", ascending=False)
            st.bar_chart(df_vif.set_index("Variable"), color="#e74c3c")
        else:
            st.success("No se detectó multicolinealidad severa (VIF limpio).")

    # 3. MAPA DE RUIDO CLÍNICO (Outliers: IQR vs Z-Score)
    outliers_data = analisis.get("valores_atipicos_severos", {})
    if outliers_data:
        st.markdown("---")
        st.subheader("🕵️ Mapa de Ruido y Anomalías")
        filas_outliers = []
        for col, stats in outliers_data.items():
            filas_outliers.append({
                "Variable": col,
                "Asimetría Biológica (IQR %)": stats.get("porcentaje_outliers_iqr_biologico", 0.0),
                "Errores Extremos (Z-Score %)": stats.get("porcentaje_outliers_zscore_extremo", 0.0)
            })
        df_outliers = pd.DataFrame(filas_outliers).set_index("Variable")
        st.bar_chart(df_outliers)

    # 4. RADIOGRAFÍA DEL ESPACIO (PCA Scatter Plot)
    sep_data = analisis.get("separabilidad_clases", {})
    if sep_data:
        st.markdown("---")
        diagnostico = sep_data.get('diagnostico_topologico', 'Desconocido')
        silueta = sep_data.get('score_silueta_pca', 0)
        
        st.subheader(f"🌌 Radiografía Espacial: {diagnostico}")
        st.caption(f"**Score de Silueta:** {silueta} (Mide qué tan separadas están las clases)")
        
        cols_num = df.select_dtypes(include=['number']).columns
        cols_pred = [c for c in cols_num if c != columna_objetivo]
        
        if len(cols_pred) >= 2:
            df_pca = df[cols_pred + [columna_objetivo]].dropna()
            if len(df_pca) > 10 and df_pca[columna_objetivo].nunique() > 1:
                if len(df_pca) > 1000:
                    df_pca = df_pca.sample(1000, random_state=42)
                    st.caption("*(Mostrando una muestra aleatoria de 1000 pacientes para optimizar rendimiento)*")
                
                try:
                    X_pca_vis = StandardScaler().fit_transform(df_pca[cols_pred])
                    componentes = PCA(n_components=2, random_state=42).fit_transform(X_pca_vis)
                    
                    df_plot = pd.DataFrame(componentes, columns=["Componente Principal 1", "Componente Principal 2"])
                    df_plot["Clase"] = df_pca[columna_objetivo].astype(str).values
                    
                    st.scatter_chart(df_plot, x="Componente Principal 1", y="Componente Principal 2", color="Clase")
                except Exception:
                    st.warning("No se pudo renderizar la proyección 2D debido a la topología de los datos.")


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
                
            # Renderizado del reporte LLM
            st.markdown("---")
            st.markdown(respuesta.text)
            
            # Renderizado de la capa visual matemática
            renderizar_graficos_auditoria(json_estadisticas, df, columna_objetivo)
            
            st.markdown("---")
            with st.expander("Ver JSON estadístico crudo (Enviado a la IA)"):
                st.code(json_estadisticas, language="json")
                
        except Exception as e:
            st.error(f"Falla crítica en el procesamiento: {e}")
