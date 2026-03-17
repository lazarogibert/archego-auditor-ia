import pandas as pd
import numpy as np
import json
import datetime
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def procesar_df_a_json(df, columna_objetivo, columna_nodo=""):
    resumen = {
        "metadatos": {
            "filas_totales": len(df),
            "fecha_auditoria": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # <--- [NUEVO] Sello de tiempo exacto
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

        # ==========================================
        # 1. ANÁLISIS DE CATEGORÍAS Y TEXTO LIBRE (NLP EXTENDIDO)
        # ==========================================
        cols_categoricas = df.select_dtypes(include=['object', 'category']).columns
        for col in cols_categoricas:
            nulos = int(df[col].isnull().sum())
            pct_nulos = round((nulos / len(df)) * 100, 2)
            unicos = df[col].nunique()
            
            info_col = {
                "valores_unicos": unicos,
                "porcentaje_nulos_real": pct_nulos
            }
            
            # Heurística: Si más del 50% es único, es texto libre (ensayos, notas) o un ID
            if unicos > (len(df) * 0.5) and unicos > 1: 
                info_col["tipo_inferido"] = "Texto Libre (NLP) o Identificador"
                textos_brutos = df[col].dropna().astype(str)
                
                # Filtro de seguridad: Purgar strings compuestos solo de espacios (" ", "")
                textos_validos = textos_brutos[textos_brutos.str.strip().astype(bool)]
                
                if not textos_validos.empty:
                    # Cálculo de topología de tokens (Aproximación por espacios)
                    conteo_palabras = textos_validos.apply(lambda x: len(x.split()))
                    
                    info_col["estadisticas_texto"] = {
                        "longitud_promedio_caracteres": round(float(textos_validos.apply(len).mean()), 2),
                        "palabras_promedio": round(float(conteo_palabras.mean()), 2),
                        "percentil_50_palabras": int(conteo_palabras.quantile(0.50)),
                        "percentil_90_palabras": int(conteo_palabras.quantile(0.90)),
                        "percentil_99_palabras": int(conteo_palabras.quantile(0.99))
                    }
                    
                    # Detección de Ruido NLP (Textos inútiles para modelado)
                    textos_muy_cortos = int((conteo_palabras < 3).sum())
                    info_col["alertas_nlp"] = {
                        "textos_menores_a_3_palabras_pct": round(float(textos_muy_cortos / len(textos_validos)) * 100, 2)
                    }
                    
                    # Diversidad Léxica (Type-Token Ratio - TTR)
                    # Submuestreo extremo para no colapsar la RAM concatenando 50,000 ensayos
                    if len(textos_validos) > 1000:
                        muestra_ttr = textos_validos.sample(1000, random_state=42)
                    else:
                        muestra_ttr = textos_validos
                        
                    todas_las_palabras = " ".join(muestra_ttr.tolist()).lower().split()
                    if len(todas_las_palabras) > 0:
                        ttr = len(set(todas_las_palabras)) / len(todas_las_palabras)
                        info_col["riqueza_lexica_ttr"] = round(float(ttr), 3)
            else:
                info_col["tipo_inferido"] = "Categoría Estándar"
                top_valores = df[col].value_counts(normalize=True).head(3).to_dict()
                info_col["top_3_frecuencias_pct"] = {str(k): round(float(v) * 100, 2) for k, v in top_valores.items()}
                
            resumen["variables_categoricas"][col] = info_col
            
            if unicos == 1:
                resumen["alertas_criticas"].append(f"Varianza cero en '{col}'. Variable inútil para predicción.")
       
        
        # ==========================================
        # 2. ANÁLISIS PREDICTIVO NUMÉRICO
        # ==========================================
        if columna_objetivo and columna_objetivo in df.columns:
            dist = df[columna_objetivo].value_counts(normalize=True).to_dict()
            resumen["analisis_predictivo"]["objetivo"] = {
                "columna": columna_objetivo,
                "distribucion_pct": {str(k): round(float(v) * 100, 2) for k, v in dist.items()}
            }
            
            # Separar predictores numéricos del objetivo (Aislamiento seguro)
            cols_num = df.select_dtypes(include=[np.number]).columns
            cols_pred_num = [c for c in cols_num if c != columna_objetivo]
            
            # A. Correlaciones Lineales (Solo si el objetivo es intrínsecamente numérico)
            if columna_objetivo in cols_num:
                corr_pearson = df[cols_num].corr(method='pearson')[columna_objetivo].drop(columna_objetivo)
                resumen["analisis_predictivo"]["pearson_fuertes"] = corr_pearson[abs(corr_pearson) > 0.3].round(3).to_dict()
                
                corr_spearman = df[cols_num].corr(method='spearman')[columna_objetivo].drop(columna_objetivo)
                resumen["analisis_predictivo"]["spearman_fuertes"] = corr_spearman[abs(corr_spearman) > 0.3].round(3).to_dict()

            if len(cols_pred_num) > 0:
                # B. INFORMACIÓN MUTUA (Seguridad JSON)
                df_limpio_mi = df[cols_pred_num + [columna_objetivo]].dropna()
                if not df_limpio_mi.empty:
                    X_mi = df_limpio_mi[cols_pred_num]
                    y_mi = df_limpio_mi[columna_objetivo]
                    
                    es_categorico = y_mi.dtype == 'object' or y_mi.dtype.name == 'category' or len(y_mi.unique()) < 20
                    try:
                        if es_categorico:
                            mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
                        else:
                            mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
                        
                        mi_series = pd.Series(mi_scores, index=X_mi.columns)
                        mi_filtrado = mi_series[mi_series > 0.05].round(3).to_dict()
                        resumen["analisis_predictivo"]["informacion_mutua_relevante"] = {k: float(v) for k, v in mi_filtrado.items()}
                    except Exception:
                        pass

                # C. MULTICOLINEALIDAD SEVERA (VIF)
                resumen["analisis_predictivo"]["multicolinealidad_vif_severa"] = {}
                df_preds = df[cols_pred_num].dropna()
                if not df_preds.empty:
                    try:
                        X_vif = df_preds.loc[:, df_preds.var() > 0]
                        if len(X_vif.columns) >= 2 and len(X_vif) > len(X_vif.columns):
                            corr_matrix = X_vif.corr().values
                            inv_corr = np.linalg.inv(corr_matrix)
                            vifs = np.diag(inv_corr)
                            for col, vif in zip(X_vif.columns, vifs):
                                if vif > 10 and not np.isinf(vif):
                                    resumen["analisis_predictivo"]["multicolinealidad_vif_severa"][col] = round(float(vif), 2)
                    except np.linalg.LinAlgError:
                        resumen["alertas_criticas"].append("Matriz singular detectada. Existe correlación perfecta (1.0) entre predictores numéricos. Requiere limpieza manual.")
                    except Exception:
                        pass

                # D. DETECCIÓN EXHAUSTIVA DE VALORES ATÍPICOS (IQR + Z-SCORE)
                resumen["analisis_predictivo"]["valores_atipicos_severos"] = {}
                for col in cols_pred_num:
                    s = df[col].dropna()
                    if len(s) > 10 and s.nunique() > 2:
                        Q1 = float(s.quantile(0.25))
                        Q3 = float(s.quantile(0.75))
                        IQR = Q3 - Q1
                        media = float(s.mean())
                        std = float(s.std())
                        
                        pct_iqr = 0.0
                        pct_z = 0.0
                        
                        if IQR > 0:
                            lim_inf_iqr = Q1 - 1.5 * IQR
                            lim_sup_iqr = Q3 + 1.5 * IQR
                            outliers_iqr = s[(s < lim_inf_iqr) | (s > lim_sup_iqr)]
                            pct_iqr = round(float((len(outliers_iqr) / len(s)) * 100), 2)
                            
                        if std > 0:
                            lim_inf_z = media - 3 * std
                            lim_sup_z = media + 3 * std
                            outliers_z = s[(s < lim_inf_z) | (s > lim_sup_z)]
                            pct_z = round(float((len(outliers_z) / len(s)) * 100), 2)
                            
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

            # E. SEPARABILIDAD DE CLASES (PCA + SILHOUETTE)
            resumen["analisis_predictivo"]["separabilidad_clases"] = {}
            y_completo = df[columna_objetivo].dropna()
            n_clases = y_completo.nunique()
            
            if 1 < n_clases <= 15 and len(cols_pred_num) >= 2:
                df_sep = df[cols_pred_num + [columna_objetivo]].dropna()
                if len(df_sep) > 50:
                    y_sep = df_sep[columna_objetivo]
                    clases_validas = y_sep.value_counts()[y_sep.value_counts() > 1].index
                    df_sep = df_sep[df_sep[columna_objetivo].isin(clases_validas)]
                    
                    if df_sep[columna_objetivo].nunique() > 1:
                        if len(df_sep) > 3000:
                            df_sample = df_sep.sample(n=3000, random_state=42)
                            y_samp_verif = df_sample[columna_objetivo]
                            clases_seguras = y_samp_verif.value_counts()[y_samp_verif.value_counts() > 1].index
                            df_sample = df_sample[df_sample[columna_objetivo].isin(clases_seguras)]
                        else:
                            df_sample = df_sep
                            
                        if df_sample[columna_objetivo].nunique() > 1:
                            X_sample = df_sample[cols_pred_num]
                            y_sample = df_sample[columna_objetivo]
                            
                            try:
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X_sample)
                                n_comps = min(3, len(cols_pred_num))
                                pca = PCA(n_components=n_comps, random_state=42)
                                X_pca = pca.fit_transform(X_scaled)
                                varianza_explicada = float(sum(pca.explained_variance_ratio_))
                                score_silueta = float(silhouette_score(X_pca, y_sample, random_state=42))
                                
                                if score_silueta > 0.40:
                                    estado_manifold = "Altamente Separable (Clústeres limpios)"
                                elif score_silueta > 0.15:
                                    estado_manifold = "Superposición Moderada (Requiere no-linealidad)"
                                else:
                                    estado_manifold = "Caos Espacial (Ruido severo / Clases superpuestas)"
                                    
                                resumen["analisis_predictivo"]["separabilidad_clases"] = {
                                    "score_silueta_pca": round(score_silueta, 3),
                                    "varianza_retenida_pct": round(varianza_explicada * 100, 2),
                                    "diagnostico_topologico": estado_manifold
                                }
                            except Exception:
                                pass

        # ==========================================
        # 3. ANÁLISIS DE SILOS (APRENDIZAJE FEDERADO)
        # ==========================================
        if columna_nodo and columna_nodo in df.columns:
            resumen["analisis_federado_nodos"] = {}
            dist_nodos = df[columna_nodo].value_counts(normalize=True).to_dict()
            resumen["analisis_federado_nodos"]["distribucion_muestras"] = {str(k): round(float(v) * 100, 2) for k, v in dist_nodos.items()}
            
            if columna_objetivo and columna_objetivo in df.columns:
                resumen["analisis_federado_nodos"]["divergencia_objetivo_por_nodo"] = {}
                for nodo in df[columna_nodo].unique():
                    df_nodo = df[df[columna_nodo] == nodo]
                    dist_obj_nodo = df_nodo[columna_objetivo].value_counts(normalize=True).to_dict()
                    resumen["analisis_federado_nodos"]["divergencia_objetivo_por_nodo"][str(nodo)] = {str(k): round(float(v) * 100, 2) for k, v in dist_obj_nodo.items()}

    except Exception as e:
        resumen["alertas_criticas"].append(f"Error general en el procesamiento del DataFrame: {str(e)}")

    # Retorno seguro forzando la compatibilidad UTF-8
    return json.dumps(resumen, indent=4, ensure_ascii=False)

