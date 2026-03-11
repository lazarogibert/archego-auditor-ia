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
            
            # A. Correlaciones y Dependencias (Solo si el objetivo es numérico/binario)
            if columna_objetivo in cols_num:
                corr_pearson = df[cols_num].corr(method='pearson')[columna_objetivo].drop(columna_objetivo)
                resumen["analisis_predictivo"]["pearson_fuertes"] = corr_pearson[abs(corr_pearson) > 0.3].round(3).to_dict()
                
                corr_spearman = df[cols_num].corr(method='spearman')[columna_objetivo].drop(columna_objetivo)
                resumen["analisis_predictivo"]["spearman_fuertes"] = corr_spearman[abs(corr_spearman) > 0.3].round(3).to_dict()
                
                df_limpio = df[cols_num].dropna()
                if not df_limpio.empty:
                    X = df_limpio.drop(columns=[columna_objetivo])
                    y = df_limpio[columna_objetivo]
                    
                    # Información Mutua (Captura no linealidad)
                    mi_scores = mutual_info_classif(X, y, random_state=42) if len(y.unique()) < 20 else mutual_info_regression(X, y, random_state=42)
                    mi_series = pd.Series(mi_scores, index=X.columns)
                    resumen["analisis_predictivo"]["informacion_mutua_relevante"] = mi_series[mi_series > 0.05].round(3).to_dict()

                    # ==========================================
                    # B. MULTICOLINEALIDAD SEVERA (VIF)
                    # ==========================================
                    resumen["analisis_predictivo"]["multicolinealidad_vif_severa"] = {}
                    try:
                        # Filtro de seguridad: Variables con varianza > 0
                        X_vif = X.loc[:, X.var() > 0]
                        # Control dimensional: Necesitamos más filas que columnas para invertir la matriz
                        if len(X_vif.columns) >= 2 and len(X_vif) > len(X_vif.columns):
                            corr_matrix = X_vif.corr().values
                            inv_corr = np.linalg.inv(corr_matrix) # Inversión matricial
                            vifs = np.diag(inv_corr)
                            
                            for col, vif in zip(X_vif.columns, vifs):
                                # Solo extraemos ruido tóxico (VIF > 10)
                                if vif > 10 and not np.isinf(vif):
                                    resumen["analisis_predictivo"]["multicolinealidad_vif_severa"][col] = round(float(vif), 2)
                    except np.linalg.LinAlgError:
                        resumen["alertas_criticas"].append("Matriz singular detectada. Existe correlación perfecta (1.0) entre predictores numéricos. Requiere limpieza manual.")
                    except Exception:
                        pass # Silenciamos otros errores menores de numpy para mantener el pipeline vivo

            # ==========================================
            # C. DETECCIÓN EXHAUSTIVA DE VALORES ATÍPICOS (HÍBRIDO: IQR + Z-SCORE)
            # ==========================================
            resumen["analisis_predictivo"]["valores_atipicos_severos"] = {}
            # Excluimos la variable objetivo para no sesgar el análisis
            cols_pred_num = [c for c in cols_num if c != columna_objetivo]
            
            for col in cols_pred_num:
                s = df[col].dropna()
                
                # 1. Filtro de Seguridad: Mínimo 10 muestras y exclusión de banderas binarias (0/1)
                if len(s) > 10 and s.nunique() > 2: 
                    # Extraemos métricas base forzando float nativo para evitar colapsos en el JSON
                    Q1 = float(s.quantile(0.25))
                    Q3 = float(s.quantile(0.75))
                    IQR = Q3 - Q1
                    media = float(s.mean())
                    std = float(s.std())
                    
                    pct_iqr = 0.0
                    pct_z = 0.0
                    
                    # 2. Motor 1: Rango Intercuartílico (Asimetría Biológica)
                    if IQR > 0:
                        lim_inf_iqr = Q1 - 1.5 * IQR
                        lim_sup_iqr = Q3 + 1.5 * IQR
                        outliers_iqr = s[(s < lim_inf_iqr) | (s > lim_sup_iqr)]
                        pct_iqr = round(float((len(outliers_iqr) / len(s)) * 100), 2)
                        
                    # 3. Motor 2: Z-Score (Anomalías y Errores Extremos)
                    if std > 0:
                        lim_inf_z = media - 3 * std
                        lim_sup_z = media + 3 * std
                        outliers_z = s[(s < lim_inf_z) | (s > lim_sup_z)]
                        pct_z = round(float((len(outliers_z) / len(s)) * 100), 2)
                        
                    # 4. Extracción: Solo reportamos si cruza los umbrales de peligro
                    if pct_iqr > 5.0 or pct_z > 0.0:
                        resumen["analisis_predictivo"]["valores_atipicos_severos"][col] = {
                            "porcentaje_outliers_iqr_biologico": pct_iqr,
                            "porcentaje_outliers_zscore_extremo": pct_z,
                            "estadisticas_base": {
                                "media": round(media, 2),
                                "desviacion_estandar": round(std, 2),
                                "valor_minimo_real": round(float(s.min()), 2),
                                "valor_maximo_real": round(float(s.max()), 2)
                            }
                        }
