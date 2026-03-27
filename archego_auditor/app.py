import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
from google import genai
from core.procesador_datos import procesar_df_a_json

st.set_page_config(page_title="Archego | Auditor B2B", page_icon="⚡", layout="wide")
API_KEY = os.environ.get("GEMINI_API_KEY", "TU_API_KEY_AQUI")

# ==========================================
# 1. CARGA DE PROMPTS (BLINDADA)
# ==========================================
AGENTES = {
    "Arquitecto ML (Datos Médicos)": "experto_medicina",
    "Ingeniero NLP (Text Mining)": "experto_argumentacion"
}

def cargar_prompt(nombre_archivo):
    """Carga el prompt protegiendo contra archivos inexistentes."""
    try:
        directorio_base = os.path.dirname(os.path.abspath(__file__))
        ruta = os.path.join(directorio_base, "prompts", f"{nombre_archivo}.txt")
        with open(ruta, "r", encoding="utf-8") as archivo:
            return archivo.read()
    except FileNotFoundError:
        return f"ERROR_PROMPT_FALTANTE: No se encontró el archivo '{nombre_archivo}.txt' en la carpeta 'prompts'."
    except Exception as e:
        return f"ERROR_SISTEMA: {str(e)}"

# ==========================================
# 2. MOTOR VISUAL: RENDERIZADO ADAPTATIVO (Plotly)
# ==========================================
def renderizar_graficos_auditoria(json_string, df, columna_objetivo):
    """Renderiza gráficos dinámicos protegiendo la memoria y adaptándose a la tarea."""
    try:
        resumen = json.loads(json_string)
    except Exception as e:
        st.error(f"Error al leer el JSON interno para gráficos: {e}")
        return

    st.markdown("---")
    st.header("📊 Radiografía Visual del Dataset")

    tipo_tarea = resumen.get("analisis_predictivo", {}).get("objetivo", {}).get("tipo_tarea", "Clasificación")
    analisis_pred = resumen.get("analisis_predictivo", {})

    # --- A. DISTRIBUCIÓN DEL OBJETIVO ---
    st.subheader(f"Distribución del Objetivo: '{columna_objetivo}'")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if "Regresión" in tipo_tarea:
            fig_obj = px.histogram(df, x=columna_objetivo, nbins=50, marginal="box", color_discrete_sequence=['#2ecc71'])
            fig_obj.update_layout(yaxis_title="Frecuencia (Filas)", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_obj, use_container_width=True)
        else:
            dist_pct = analisis_pred.get("objetivo", {}).get("distribucion_pct", {})
            if dist_pct:
                df_dist = pd.DataFrame(list(dist_pct.items()), columns=['Clase', 'Porcentaje'])
                fig_obj = px.bar(df_dist, x='Clase', y='Porcentaje', text='Porcentaje', color='Clase')
                fig_obj.update_traces(texttemplate='%{text}%', textposition='outside')
                fig_obj.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_obj, use_container_width=True)
            else:
                st.warning("No se pudo extraer la distribución de clases.")
                
    with col2:
        st.info(f"**Tarea Inferida:**\n\n{tipo_tarea}")
        if "Regresión" in tipo_tarea:
            stats = analisis_pred.get("objetivo", {}).get("estadisticas_objetivo", {})
            st.write(f"**Media:** {stats.get('media')}")
            st.write(f"**Desviación Estd:** {stats.get('desviacion_estandar')}")
            st.write(f"**Rango:** {stats.get('minimo')} a {stats.get('maximo')}")

    # --- B. RELEVANCIA PREDICTIVA ---
    st.markdown("---")
    st.subheader("🎯 Relevancia Predictiva de Variables")
    
    dict_relevancia = {}
    es_correlacion = False

    if "Regresión" in tipo_tarea:
        dict_relevancia = analisis_pred.get("pearson_fuertes", {})
        titulo_eje_x = "Fuerza de Correlación Lineal (Pearson)"
        if not dict_relevancia:
            dict_relevancia = analisis_pred.get("spearman_fuertes", {})
            titulo_eje_x = "Fuerza de Correlación Monotónica (Spearman)" if dict_relevancia else ""
        es_correlacion = True
    else:
        dict_relevancia = analisis_pred.get("informacion_mutua_relevante", {})
        titulo_eje_x = "Ganancia de Información (Mutual Information)"

    if dict_relevancia:
        df_imp = pd.DataFrame(list(dict_relevancia.items()), columns=['Predictor', 'Score'])
        df_imp['Predictor_Label'] = df_imp['Predictor'].astype(str).apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
        df_imp['Magnitud'] = df_imp['Score'].abs() if es_correlacion else df_imp['Score']
        df_imp = df_imp.sort_values(by='Magnitud', ascending=True).tail(15)

        fig_imp = px.bar(
            df_imp, x='Score', y='Predictor_Label', orientation='h', text_auto='.3f', color='Magnitud',
            color_continuous_scale="Blues" if not es_correlacion else "RdBu",
            hover_data={"Predictor": True, "Predictor_Label": False}
        )
        if es_correlacion: fig_imp.update_layout(xaxis=dict(range=[-1.1, 1.1]))
        fig_imp.update_layout(xaxis_title=titulo_eje_x, yaxis_title="", coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("No se detectaron predictores numéricos tabulares con señal matemática fuerte.")

    # --- C. MAPA DE RUIDO Y ANOMALÍAS ---
    st.markdown("---")
    st.subheader("🌪️ Mapa de Ruido y Topología de Tensores")
    
    outliers_data = analisis_pred.get("valores_atipicos_severos", {})
    hay_numericas = bool(analisis_pred.get("pearson_fuertes") or analisis_pred.get("spearman_fuertes") or analisis_pred.get("informacion_mutua_relevante") or outliers_data)

    if hay_numericas:
        if outliers_data:
            lista_ruido = []
            for col, stats in outliers_data.items():
                pct_iqr, pct_z = float(stats.get("porcentaje_outliers_iqr", 0)), float(stats.get("porcentaje_outliers_zscore", 0))
                pct_z_real = min(100.0, pct_z)
                pct_iqr_solo = min(100.0 - pct_z_real, max(0.0, pct_iqr - pct_z))
                pct_limpio = max(0.0, 100.0 - pct_iqr_solo - pct_z_real)
                col_label = col[:30] + "..." if len(col) > 30 else col

                lista_ruido.extend([
                    {"Variable_Label": col_label, "Variable": col, "Tipo": "Datos Limpios", "Porcentaje": pct_limpio},
                    {"Variable_Label": col_label, "Variable": col, "Tipo": "Asimetría (IQR)", "Porcentaje": pct_iqr_solo},
                    {"Variable_Label": col_label, "Variable": col, "Tipo": "Corrupción (Z-Score)", "Porcentaje": pct_z_real}
                ])

            df_ruido = pd.DataFrame(lista_ruido)
            vars_con_ruido = df_ruido[df_ruido["Tipo"] != "Datos Limpios"].groupby("Variable_Label")["Porcentaje"].sum()
            vars_a_graficar = vars_con_ruido[vars_con_ruido > 0].index.tolist()
            
            if vars_a_graficar:
                df_ruido = df_ruido[df_ruido["Variable_Label"].isin(vars_a_graficar)]
                fig_ruido = px.bar(
                    df_ruido, x="Porcentaje", y="Variable_Label", color="Tipo", orientation="h",
                    hover_data={"Variable": True, "Variable_Label": False},
                    color_discrete_map={"Datos Limpios": "#2ecc71", "Asimetría (IQR)": "#f1c40f", "Corrupción (Z-Score)": "#e74c3c"},
                    category_orders={"Tipo": ["Datos Limpios", "Asimetría (IQR)", "Corrupción (Z-Score)"]}
                )
                fig_ruido.update_layout(xaxis_title="Porcentaje de Filas (%)", yaxis_title="", barmode='stack', xaxis=dict(range=[0, 100]), margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_ruido, use_container_width=True)
            else: st.success("Variables numéricas libres de valores atípicos severos.")
        else: st.success("Variables tabulares 100% libres de valores atípicos severos.")
    else:
        # RUTA NLP
        cols_texto = []
        for c in df.select_dtypes(include=['object', 'string']).columns:
            if c != columna_objetivo:
                unicos_pct = df[c].nunique() / len(df)
                muestra = df[c].dropna().astype(str).head(100)
                if not muestra.empty:
                    longitud_promedio_chars = muestra.apply(len).mean()
                    longitud_promedio_palabra = muestra.apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0).mean()
                    if ((unicos_pct > 0.3 and longitud_promedio_chars > 15) or (longitud_promedio_chars > 50)) and (longitud_promedio_palabra < 15):
                        cols_texto.append(c)
        
        if cols_texto:
            df_noise_nlp = df[cols_texto].dropna(how='all').copy()
            if len(df_noise_nlp) > 10000: df_noise_nlp = df_noise_nlp.sample(10000, random_state=42)

            col_calc = '__sys_texto_analisis__'
            if len(cols_texto) > 1:
                df_noise_nlp[col_calc] = df_noise_nlp[cols_texto].fillna('').astype(str).agg(' '.join, axis=1)
                titulo_grafico = f"Longitud Combinada ({len(cols_texto)} cols)"
            else:
                df_noise_nlp[col_calc] = df_noise_nlp[cols_texto[0]].astype(str)
                titulo_grafico = f"Longitud en '{cols_texto[0]}'"

            if not df_noise_nlp.empty:
                col_tokens, col_vocab, col_ttr = '__sys_tokens__', '__sys_vocab__', '__sys_ttr__'
                df_noise_nlp[col_tokens] = df_noise_nlp[col_calc].apply(lambda x: len(x.split()))
                
                fig_nlp_noise = px.histogram(df_noise_nlp, x=col_tokens, nbins=50, marginal="box", title=f"Riesgo de Truncamiento: {titulo_grafico}", color_discrete_sequence=['#9b59b6'])
                fig_nlp_noise.add_vline(x=512, line_dash="dash", line_color="red")
                fig_nlp_noise.update_layout(xaxis_title="Tokens Aprox.", yaxis_title="Documentos", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_nlp_noise, use_container_width=True)
                
                st.markdown("---")
                st.subheader("🕵️ Huella Léxica (Type-Token Ratio)")
                df_noise_nlp[col_vocab] = df_noise_nlp[col_calc].apply(lambda x: len(set(x.split())))
                df_noise_nlp[col_ttr] = np.where(df_noise_nlp[col_tokens] > 0, df_noise_nlp[col_vocab] / df_noise_nlp[col_tokens], 0.0)
                
                fig_ttr = px.scatter(df_noise_nlp, x=col_tokens, y=col_ttr, opacity=0.4, color_discrete_sequence=['#e67e22'], title="Huella Léxica (TTR)", hover_data={col_calc: False, col_vocab: False})
                fig_ttr.add_hline(y=0.25, line_dash="dot", line_color="red")
                fig_ttr.update_layout(xaxis_title="Tokens", yaxis_title="TTR", yaxis=dict(range=[-0.05, 1.05]), margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_ttr, use_container_width=True)
            else: st.info("Textos vacíos o ilegibles.")
        else: st.info("No hay columnas tabulares o de texto válidas para mapear ruido.")


# ==========================================
# 3. ESTRUCTURA PRINCIPAL DE LA APP (UI BLINDADA)
# ==========================================
def main():
    st.title("⚡ Archego | Auditor de Datasets")
    st.markdown("Auditor de IA estricto para modelos centralizados y descentralizados.")

    # Memoria de Sesión
    if 'estado_auditoria' not in st.session_state:
        st.session_state.estado_auditoria = False

    # Verificación temprana de la API Key (Fail Fast)
    if API_KEY == "TU_API_KEY_AQUI" or not API_KEY:
        st.error("🚨 **ALERTA DE SISTEMA:** No se ha configurado la clave 'GEMINI_API_KEY'. Por favor, configúrala en las variables de entorno.")
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Configuración")
        agente_seleccionado = st.selectbox("Especialidad del Agente:", list(AGENTES.keys()))
        
        # Leemos los archivos primero para poder extraer las columnas reales
        archivos_csv = st.file_uploader("Dataset(s) (.csv)", type=["csv"], accept_multiple_files=True)
        
        # Variables por defecto
        columna_objetivo = None
        columna_nodo_final = ""
        df_global = None

        if archivos_csv:
            # Consolidamos los dataframes al vuelo para alimentar los dropdowns
            try:
                lista_dfs = []
                for archivo in archivos_csv:
                    try:
                        df_temp = pd.read_csv(archivo)
                    except UnicodeDecodeError:
                        archivo.seek(0)
                        df_temp = pd.read_csv(archivo, encoding='latin1', sep=None, engine='python')
                    
                    if len(archivos_csv) > 1:
                        df_temp['origen_archivo_automatico'] = archivo.name
                    lista_dfs.append(df_temp)
                    
                df_global = pd.concat(lista_dfs, ignore_index=True)
                
                # [BLINDAJE UI] Usamos selectbox en lugar de text_input para evitar errores humanos
                columnas_disponibles = list(df_global.columns)
                columna_objetivo = st.selectbox("🎯 Columna Objetivo (Obligatorio):", columnas_disponibles)
                
                # Dinámica del Nodo
                if len(archivos_csv) > 1:
                    st.info("🌐 Nodo Multicéntrico detectado automáticamente por origen de archivo.")
                    columna_nodo_final = 'origen_archivo_automatico'
                else:
                    opciones_nodo = ["Ninguno"] + columnas_disponibles
                    columna_nodo_manual = st.selectbox("🌐 Columna Nodo (Opcional):", opciones_nodo)
                    columna_nodo_final = columna_nodo_manual if columna_nodo_manual != "Ninguno" else ""

            except Exception as e:
                st.error(f"Error al leer los archivos: {e}")
                st.stop()
        
        # Botón bloqueado si no hay archivos
        if st.button("🚀 Ejecutar Auditoría", type="primary", use_container_width=True, disabled=not archivos_csv):
            st.session_state.estado_auditoria = True

    with col2:
        st.subheader("2. Reporte y Visualización")
        
        if st.session_state.estado_auditoria and df_global is not None and columna_objetivo:
            try:
                # --- NÚCLEO DE LA APLICACIÓN ---
                with st.spinner("🤖 Analizando topología y consultando a Gemini..."):
                    # 1. Extracción Matemática
                    json_estadisticas = procesar_df_a_json(df_global, columna_objetivo, columna_nodo_final)
                    
                    # 2. Llamada a Gemini (Protegida)
                    prompt_maestro = cargar_prompt(AGENTES[agente_seleccionado])
                    
                    if prompt_maestro.startswith("ERROR"):
                        st.error(prompt_maestro)
                        st.session_state.estado_auditoria = False
                        st.stop()

                    prompt_final = f"{prompt_maestro}\n\nDATOS DEL DATASET:\n{json_estadisticas}"
                    
                    client = genai.Client(api_key=API_KEY)
                    respuesta = client.models.generate_content(
                        model='models/gemini-2.5-flash',
                        contents=prompt_final
                    )
                
                # Renderizado del reporte LLM
                st.markdown("### 📄 Reporte Técnico (IA)")
                st.markdown(respuesta.text)
                
                # Renderizado de la capa visual (Plotly)
                renderizar_graficos_auditoria(json_estadisticas, df_global, columna_objetivo)
                
                st.markdown("---")
                with st.expander("Ver JSON matemático crudo (Enviado a Gemini)"):
                    st.code(json_estadisticas, language="json")
                    
            except Exception as e:
                st.error(f"Falla crítica en el procesamiento del modelo o visualización: {str(e)}")
                st.session_state.estado_auditoria = False
        else:
            if not st.session_state.estado_auditoria:
                st.info("👈 Sube un dataset y configura los parámetros para comenzar.")

if __name__ == "__main__":
    main()
