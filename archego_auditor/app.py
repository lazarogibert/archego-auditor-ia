import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
from google import genai
from core.procesador_datos import procesar_df_a_json

st.set_page_config(page_title="Archego | Auditor ML", page_icon="⚡", layout="wide")
API_KEY = os.environ.get("GEMINI_API_KEY", "TU_API_KEY_AQUI")

# ==========================================
# 0. CONTROLADOR DE ESTADO Y DICCIONARIO MAESTRO
# ==========================================
if 'estado_auditoria' not in st.session_state:
    st.session_state.estado_auditoria = False

def resetear_estado():
    """Apaga la auditoría si el usuario cambia cualquier parámetro para evitar llamadas accidentales a la API."""
    st.session_state.estado_auditoria = False

MOTORES_CONFIG = {
    "🏥 Clínico / Médico": {
        "titulo": "🏥 Auditor Clínico de Machine Learning",
        "descripcion": "Sube tu dataset médico para evaluar riesgo de fuga de datos, viabilidad biométrica y sesgos en salud.",
        "agentes": {
            "Arquitecto ML (Clínico)": "experto_medicina",
            "Ingeniero NLP (Text Mining)": "experto_argumentacion"
        }
    },
    "🏢 General / Negocios": {
        "titulo": "🧠 Auditor Arquitectónico de Machine Learning",
        "descripcion": "Sube tu dataset corporativo para evaluar integridad predictiva, distribuciones y viabilidad del modelo de negocio.",
        "agentes": {
            "Arquitecto ML (General)": "experto_general",
            "Ingeniero NLP (Text Mining)": "experto_argumentacion"
        }
    }
}

# ==========================================
# 1. CARGA DE PROMPTS Y DATOS (CACHÉ)
# ==========================================
@st.cache_data
def cargar_prompt(nombre_archivo):
    try:
        directorio_base = os.path.dirname(os.path.abspath(__file__))
        ruta = os.path.join(directorio_base, "prompts", f"{nombre_archivo}.txt")
        with open(ruta, "r", encoding="utf-8") as archivo:
            return archivo.read()
    except FileNotFoundError:
        return f"ERROR_PROMPT_FALTANTE: No se encontró el archivo '{nombre_archivo}.txt'."
    except Exception as e:
        return f"ERROR_SISTEMA: {str(e)}"

@st.cache_data
def extraer_json_seguro(json_string):
    """Parsea el JSON una sola vez por ejecución para ahorrar memoria."""
    try:
        return json.loads(json_string)
    except Exception:
        return None

# ==========================================
# 2. MOTOR VISUAL: RENDERIZADO ADAPTATIVO (Plotly)
# ==========================================
def renderizar_graficos_auditoria(json_string, df, columna_objetivo):
    """Renderiza gráficos dinámicos protegiendo la memoria y adaptándose a la tarea."""
    resumen = extraer_json_seguro(json_string)
    if not resumen:
        st.error("Error al leer el JSON interno para gráficos.")
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
            fig_obj = px.histogram(df, x=columna_objetivo, nbins=100, marginal="box", color_discrete_sequence=['#2ecc71'])
            fig_obj.update_layout(yaxis_title="Frecuencia (Filas)", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_obj, use_container_width=True)
            
            stats = analisis_pred.get("objetivo", {}).get("estadisticas_objetivo", {})
            if stats.get("distribucion_bimodal_en_U"):
                st.error("🚨 **ALERTA CRÍTICA DE TOPOLOGÍA:** Distribución bimodal severa (forma de 'U'). La mayoría de los datos están polarizados en los extremos. Un modelo de regresión lineal clásico fracasará aquí porque intentará predecir valores medios donde no hay datos. Considera binarizar el objetivo o usar regresión logística/beta.")
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
    st.subheader("🎯 Relevancia Predictiva de Variables Tabulares")
    
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

        if es_correlacion:
            df_imp['Tipo_Impacto'] = df_imp['Score'].apply(lambda x: 'Correlación Positiva' if x >= 0 else 'Correlación Negativa')
            mapa_colores = {'Correlación Positiva': '#3498db', 'Correlación Negativa': '#e74c3c'}
        else:
            df_imp['Tipo_Impacto'] = 'Ganancia de Información'
            mapa_colores = {'Ganancia de Información': '#2ecc71'}

        altura_dinamica_imp = max(150, len(df_imp) * 45)

        fig_imp = px.bar(
            df_imp, x='Score', y='Predictor_Label', orientation='h', text_auto='.3f', 
            color='Tipo_Impacto', color_discrete_map=mapa_colores,
            hover_data={"Predictor": True, "Predictor_Label": False, "Magnitud": False}
        )
        
        if es_correlacion: 
            fig_imp.update_layout(xaxis=dict(range=[-1.1, 1.1]))
            
        fig_imp.update_layout(
            height=altura_dinamica_imp,
            xaxis_title=titulo_eje_x, yaxis_title="", 
            showlegend=True if es_correlacion else False,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        if len(df_imp) == 1:
            st.caption(f"⚠️ **Nota de Sistema:** Solo se graficó la variable `{df_imp['Predictor'].iloc[0]}`. Este panel audita características tabulares numéricas. Para ver la relevancia interna de palabras (tokens), se requerirá un análisis posterior con TF-IDF o SHAP.")
            
    else:
        st.info("No se detectaron predictores numéricos tabulares con señal matemática fuerte.")

    # --- B.5 ACUERDO ENTRE ANOTADORES (FIABILIDAD HUMANA) ---
    acuerdo_data = resumen.get("analisis_acuerdo_anotadores", {})
    pares_evaluados = acuerdo_data.get("pares_evaluados", {})
    
    if pares_evaluados:
        st.markdown("---")
        st.subheader("🤝 Fiabilidad Humana (Inter-Annotator Agreement)")
        
        df_acuerdo = pd.DataFrame([
            {"Par de Jueces": par, "Score": datos["score"], "Métrica": datos["metrica_usada"], "Diagnóstico": datos["diagnostico_fiabilidad"]}
            for par, datos in pares_evaluados.items()
        ]).sort_values(by="Score", ascending=True)
        
        rango_x = [-1.0, 1.0] if any("Pearson" in str(m) for m in df_acuerdo["Métrica"]) else [0.0, 1.0]
        altura_dinamica_acuerdo = max(150, len(df_acuerdo) * 45)

        fig_acuerdo = px.bar(
            df_acuerdo, x="Score", y="Par de Jueces", orientation="h",
            color="Score", color_continuous_scale="RdYlGn", text_auto=".2f",
            hover_data={"Diagnóstico": True, "Métrica": True}
        )
        fig_acuerdo.add_vline(x=0.60, line_dash="dash", line_color="black", annotation_text="Umbral Aceptable (>0.60)")
        fig_acuerdo.update_layout(height=altura_dinamica_acuerdo, xaxis=dict(range=rango_x), coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_acuerdo, use_container_width=True)

    # --- C. MAPA DE RUIDO Y ANOMALÍAS ---
    st.markdown("---")
    st.subheader("🕵️ Auditoría de Calidad de Datos (Outliers Tabulares)")
    
    outliers_data = analisis_pred.get("valores_atipicos_severos", {})
    hay_numericas = bool(analisis_pred.get("pearson_fuertes") or analisis_pred.get("spearman_fuertes") or analisis_pred.get("informacion_mutua_relevante") or outliers_data)

    if hay_numericas:
        if outliers_data:
            lista_ruido = []
            for col, stats in outliers_data.items():
                pct_iqr, pct_z = float(stats.get("porcentaje_outliers_iqr", 0)), float(stats.get("porcentaje_outliers_zscore", 0))
                
                base = stats.get("estadisticas_base", {})
                skew = base.get("asimetria_skewness", 0.0)
                kurt = base.get("curtosis_kurtosis", 0.0)
                escasez = base.get("escasez_ceros_pct", 0.0)

                pct_z_real = min(100.0, pct_z)
                pct_iqr_solo = min(100.0 - pct_z_real, max(0.0, pct_iqr - pct_z))
                pct_limpio = max(0.0, 100.0 - pct_iqr_solo - pct_z_real)
                col_label = col[:30] + "..." if len(col) > 30 else col

                for tipo, pct in [("Datos Limpios", pct_limpio), ("Asimetría (IQR)", pct_iqr_solo), ("Corrupción (Z-Score)", pct_z_real)]:
                    lista_ruido.append({
                        "Variable_Label": col_label, "Variable": col, "Tipo": tipo, "Porcentaje": pct,
                        "Skewness": skew, "Curtosis": kurt, "Ceros (%)": escasez
                    })

            df_ruido = pd.DataFrame(lista_ruido)
            vars_con_ruido = df_ruido[df_ruido["Tipo"] != "Datos Limpios"].groupby("Variable_Label")["Porcentaje"].sum()
            vars_a_graficar = vars_con_ruido[vars_con_ruido > 0].index.tolist()
            
            if vars_a_graficar:
                df_ruido = df_ruido[df_ruido["Variable_Label"].isin(vars_a_graficar)]
                altura_dinamica_ruido = max(150, len(vars_a_graficar) * 45)
                
                fig_ruido = px.bar(
                    df_ruido, x="Porcentaje", y="Variable_Label", color="Tipo", orientation="h",
                    hover_data={"Variable": True, "Variable_Label": False, "Skewness": True, "Curtosis": True, "Ceros (%)": True},
                    color_discrete_map={"Datos Limpios": "#2ecc71", "Asimetría (IQR)": "#f1c40f", "Corrupción (Z-Score)": "#e74c3c"},
                    category_orders={"Tipo": ["Datos Limpios", "Asimetría (IQR)", "Corrupción (Z-Score)"]}
                )
                fig_ruido.update_layout(
                    height=altura_dinamica_ruido, 
                    xaxis_title="Porcentaje de Filas (%)", yaxis_title="", barmode='stack', 
                    xaxis=dict(range=[0, 100]), margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_ruido, use_container_width=True)
            else: st.success("Variables numéricas libres de valores atípicos severos.")
        else: st.success("Variables tabulares libres de valores atípicos severos.")
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
            if len(df_noise_nlp) > 10000: 
                df_noise_nlp = df_noise_nlp.sample(10000, random_state=42)

            nombres_pestanas = [c[:20] + "..." if len(c) > 20 else c for c in cols_texto]
            
            st.info(f"Se detectaron {len(cols_texto)} campos de texto libres.")
            pestañas = st.tabs(nombres_pestanas)

            for idx, col_txt in enumerate(cols_texto):
                with pestañas[idx]:
                    st.markdown(f"**Analizando campo:** `{col_txt}`")
                    stats_nlp = resumen.get("analisis_nlp", {}).get("columnas_analizadas", {}).get(col_txt, {})
                    ttr_val = stats_nlp.get("riqueza_lexica_ttr", 0)
                    lon_palabra = stats_nlp.get("longitud_promedio_palabra", 0)
                    
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("Riqueza Léxica (TTR)", f"{ttr_val:.2f}", "Spam / Bots" if ttr_val < 0.2 else "Saludable", delta_color="inverse" if ttr_val < 0.2 else "normal")
                    kpi2.metric("Complejidad Morfológica", f"{lon_palabra:.1f} lts/pal", "Técnico" if lon_palabra > 6 else "Coloquial", delta_color="off")
                    
                    if "correlacion_longitud_vs_score" in stats_nlp:
                        corr_len = stats_nlp.get("correlacion_longitud_vs_score", 0)
                        kpi3.metric("Sesgo Longitud (Corr)", f"{corr_len:.2f}", "Peligro" if abs(corr_len) > 0.3 else "Neutral", delta_color="inverse" if abs(corr_len) > 0.3 else "normal")
                    
                    st.markdown("---")
                    df_col = df_noise_nlp[[col_txt]].dropna().copy()
                    
                    if not df_col.empty:
                        col_tokens, col_vocab, col_ttr = f'__tks_{idx}__', f'__voc_{idx}__', f'__ttr_{idx}__'
                        
                        df_col[col_tokens] = df_col[col_txt].astype(str).apply(lambda x: len(x.split()))
                        
                        if df_col[col_tokens].sum() > 0:
                            fig_nlp_noise = px.histogram(
                                df_col, x=col_tokens, nbins=50, marginal="box", 
                                title=f"Riesgo de Truncamiento de Secuencias", 
                                color_discrete_sequence=['#9b59b6']
                            )
                            fig_nlp_noise.add_vline(x=512, line_dash="dash", line_color="red", annotation_text="Límite BERT (512)", annotation_position="top right")
                            fig_nlp_noise.update_layout(xaxis_title="Tokens Aprox.", yaxis_title="Documentos", margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig_nlp_noise, use_container_width=True)
                            
                            st.markdown("---")
                            df_col[col_vocab] = df_col[col_txt].astype(str).apply(lambda x: len(set(x.split())))
                            df_col[col_ttr] = np.where(df_col[col_tokens] > 0, df_col[col_vocab] / df_col[col_tokens], 0.0)
                            
                            fig_ttr = px.scatter(
                                df_col, x=col_tokens, y=col_ttr, opacity=0.4, 
                                color_discrete_sequence=['#e67e22'], 
                                title=f"Dispersión Semántica", 
                                hover_data={col_txt: False, col_vocab: False}
                            )
                            fig_ttr.add_hline(y=0.25, line_dash="dot", line_color="red", annotation_text="Peligro: Spam / Bots / Plantillas (TTR < 0.25)", annotation_position="bottom right")
                            fig_ttr.update_layout(xaxis_title="Tokens", yaxis_title="TTR", yaxis=dict(range=[-0.05, 1.05]), margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig_ttr, use_container_width=True)
                        else:
                            st.warning("Los textos procesados no contienen palabras válidas medibles.")
                    else: 
                        st.warning(f"La columna quedó vacía tras la limpieza de nulos.")
        else: st.info("No hay columnas tabulares o de texto válidas para mapear ruido.")

# ==========================================
# 3. ESTRUCTURA PRINCIPAL DE LA APP (UI)
# ==========================================
def main():
    st.sidebar.title("⚙️ Motor de Auditoría")
    modo_auditoria = st.sidebar.radio(
        "Selecciona el dominio del dataset:",
        list(MOTORES_CONFIG.keys()),
        on_change=resetear_estado 
    )
    st.sidebar.markdown("---")

    config_actual = MOTORES_CONFIG[modo_auditoria]
    
    st.title(config_actual["titulo"])
    st.markdown(config_actual["descripcion"])

    if API_KEY == "TU_API_KEY_AQUI" or not API_KEY:
        st.error("🚨 **ALERTA DE SISTEMA:** No se ha configurado la clave 'GEMINI_API_KEY'. Por favor, configúrala en las variables de entorno.")
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Configuración")
        
        agente_seleccionado = st.selectbox(
            "Especialidad del Agente:", 
            list(config_actual["agentes"].keys()), 
            on_change=resetear_estado 
        )
        
        archivos_csv = st.file_uploader(
            "Dataset(s) (.csv)", 
            type=["csv"], 
            accept_multiple_files=True, 
            on_change=resetear_estado 
        )
        
        columna_objetivo = None
        columna_nodo_final = ""
        df_global = None

        if archivos_csv:
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
                
                columnas_disponibles = list(df_global.columns)
                columna_objetivo = st.selectbox(
                    "🎯 Columna Objetivo (Obligatorio):", 
                    columnas_disponibles, 
                    on_change=resetear_estado 
                )
                
                if len(archivos_csv) > 1:
                    st.info("🌐 Nodo Multicéntrico detectado automáticamente por origen de archivo.")
                    columna_nodo_final = 'origen_archivo_automatico'
                else:
                    opciones_nodo = ["Ninguno"] + columnas_disponibles
                    columna_nodo_manual = st.selectbox(
                        "🌐 Columna Nodo / Segmento (Opcional):", 
                        opciones_nodo, 
                        on_change=resetear_estado 
                    )
                    columna_nodo_final = columna_nodo_manual if columna_nodo_manual != "Ninguno" else ""

            except Exception as e:
                st.error(f"Error al leer los archivos: {e}")
                st.stop()
        else:
            resetear_estado() # Purga de memoria si el usuario borra los archivos
        
        if st.button("🚀 Ejecutar Auditoría", type="primary", use_container_width=True, disabled=not archivos_csv):
            st.session_state.estado_auditoria = True

    with col2:
        st.subheader("2. Reporte y Visualización")
        
        if st.session_state.estado_auditoria and df_global is not None and columna_objetivo:
            try:
                with st.spinner(f"🤖 Analizando topología y consultando al agente..."):
                    json_estadisticas = procesar_df_a_json(df_global, columna_objetivo, columna_nodo_final)
                    
                    nombre_archivo_prompt = config_actual["agentes"][agente_seleccionado]
                    prompt_maestro = cargar_prompt(nombre_archivo_prompt)
                    
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
                
                st.markdown("### 📄 Reporte Técnico (IA)")
                st.markdown(respuesta.text)
                
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
