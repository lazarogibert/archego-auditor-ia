import pandas as pd
import numpy as np
import json
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def procesar_df_a_json(df, columna_objetivo, columna_nodo=""):
    resumen = {
        "metadatos": {
            "filas_totales": len(df),
            "columnas_totales": len(df.columns)
        },
        "alertas_criticas": [],
        "variables_categoricas": {},
        "analisis_predictivo": {}
    }

    try:
        # Regla de purga estructural: Identificadores inútiles para la fase 1
        if 'pt_id' in df.columns:
            resumen["alertas_criticas"].append("La columna 'pt_id' ha sido detectada. Esta columna es un identificador, carece de valor predictivo para la fase 1 y debe ser excluida del modelo.")

        # 1. ANÁLISIS DE CATEGORÍAS Y TEXTO LIBRE (NLP)
        cols_categoricas = df.select_dtypes(include=['object', 'category']).columns
        for col in cols_categoricas:
            nulos = int(df[col].isnull().sum())
            pct_nulos = round((nulos / len(df)) * 100, 2)
            unicos = df[col].nunique()
            
            info_col = {
                "valores_unicos": unicos,
                "porcentaje_nulos_real": pct_nulos
            }
            
            # Heurística: Si más del 50% es único, es texto libre (ensayos) o un ID
            if unicos > (len(df) * 0.5) and unicos > 1: 
                info_col["tipo_inferido"] = "Texto Libre (NLP) o Identificador"
                textos_limpios = df[col].dropna().astype(str)
                if not textos_limpios.empty:
                    info_col["longitud_promedio_caracteres"] = round(textos_limpios.apply(len).mean(), 2)
            else:
                info_col["tipo_inferido"] = "Categoría Estándar"
                top_valores = df[col].value_counts(normalize=True).head(3).to_dict()
                info_col["top_3_frecuencias_pct"] = {str(k): round(v * 100, 2) for k, v in top_valores.items()}
                
            resumen["variables_categoricas"][col] = info_col
            
            if unicos == 1:
                resumen["alertas_criticas"].append(f"Varianza cero en '{col}'. Variable inútil para predicción.")

        # 2. ANÁLISIS PREDICTIVO NUMÉRICO
        if columna_objetivo and columna_objetivo in df.columns:
            dist = df[columna_objetivo].value_counts(normalize=True).to_dict()
            resumen["analisis_predictivo"]["objetivo"] = {
                "columna": columna_objetivo,
                "distribucion_pct": {str(k): round(v * 100, 2) for k, v in dist.items()}
            }
            
            cols_num = df.select_dtypes(include=[np.number]).columns
            if columna_objetivo in cols_num:
                corr_pearson = df[cols_num].corr(method='pearson')[columna_objetivo].drop(columna_objetivo)
                resumen["analisis_predictivo"]["pearson_fuertes"] = corr_pearson[abs(corr_pearson) > 0.3].round(3).to_dict()
                
                corr_spearman = df[cols_num].corr(method='spearman')[columna_objetivo].drop(columna_objetivo)
                resumen["analisis_predictivo"]["spearman_fuertes"] = corr_spearman[abs(corr_spearman) > 0.3].round(3).to_dict()
                
                df_limpio = df[cols_num].dropna()
                if not df_limpio.empty:
                    X = df_limpio.drop(columns=[columna_objetivo])
                    y = df_limpio[columna_objetivo]
                    mi_scores = mutual_info_classif(X, y, random_state=42) if len(y.unique()) < 20 else mutual_info_regression(X, y, random_state=42)
                    mi_series = pd.Series(mi_scores, index=X.columns)
                    resumen["analisis_predictivo"]["informacion_mutua_relevante"] = mi_series[mi_series > 0.05].round(3).to_dict()

        # 3. ANÁLISIS FEDERADO (Silos)
        if columna_nodo and columna_nodo in df.columns:
            resumen["analisis_federado_nodos"] = {}
            agrupado = df.groupby(columna_nodo)
            for nodo, datos_nodo in agrupado:
                dist_nodo = datos_nodo[columna_objetivo].value_counts(normalize=True).to_dict() if columna_objetivo in datos_nodo.columns else {}
                resumen["analisis_federado_nodos"][str(nodo)] = {
                    "volumen_filas": len(datos_nodo),
                    "distribucion_pct": {str(k): round(v * 100, 2) for k, v in dist_nodo.items()}
                }

        return json.dumps(resumen, indent=4, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error_procesamiento": str(e)})