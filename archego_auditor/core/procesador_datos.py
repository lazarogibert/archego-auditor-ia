import pandas as pd
import numpy as np
import json
import datetime
import itertools
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, cohen_kappa_score

def procesar_df_a_json(df, columna_objetivo, columna_nodo=""):
    resumen = {
        "metadatos": {
            "filas_totales": len(df),
            "fecha_auditoria": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "columnas_totales": len(df.columns)
        },
        "alertas_criticas": [],
        "variables_categoricas": {},
        "analisis_predictivo": {}
    }

    try:
        # Regla de purga estructural
        if 'pt_id' in df.columns:
            resumen["alertas_criticas"].append("La columna 'pt_id' ha sido detectada y excluida por ser identificador.")

        # ==========================================
        # 0. DETECCIÓN GLOBAL DE TAREA Y PERFILADO DEL OBJETIVO
        # ==========================================
        tarea_global = "Clasificación / Discreta"
        es_regresion_tabular = False
        
        if columna_objetivo and columna_objetivo in df.columns:
            objetivo_serie = df[columna_objetivo].dropna()
            
            if pd.api.types.is_numeric_dtype(objetivo_serie) and objetivo_serie.nunique() > 15:
                es_regresion_tabular = True
                tarea_global = "Regresión Continua"
            elif objetivo_serie.dtype == 'object' and not objetivo_serie.empty:
                obj_str = objetivo_serie.astype(str)
                es_lista_estandar = obj_str.str.contains(r'\[|\,', regex=True).mean() > 0.5
                muestra_100 = obj_str.head(100)
                palabras_por_fila = muestra_100.apply(lambda x: len(x.split()))
                es_secuencia_espacios = palabras_por_fila.mean() > 3 if not palabras_por_fila.empty else False
                vocab_objetivo = len(set(" ".join(muestra_100.tolist()).split()))
                
                if es_lista_estandar or (es_secuencia_espacios and vocab_objetivo < 50):
                    tarea_global = "Segmentación de Tokens"

            resumen["analisis_predictivo"]["objetivo"] = {
                "columna": columna_objetivo,
                "tipo_tarea": tarea_global
            }
            
            if es_regresion_tabular:
                media = float(objetivo_serie.mean())
                std = float(objetivo_serie.std())
                min_val = float(objetivo_serie.min())
                max_val = float(objetivo_serie.max())
                
                # [NUEVO BLINDAJE] Detección Matemática de Distribución Bimodal (Forma de U)
                rango = max_val - min_val
                es_bimodal = False
                if rango > 0:
                    # Si más del 60% de los datos están concentrados en el 15% inferior y superior...
                    extremo_inf = objetivo_serie[objetivo_serie <= min_val + 0.15 * rango].count()
                    extremo_sup = objetivo_serie[objetivo_serie >= max_val - 0.15 * rango].count()
                    if (extremo_inf + extremo_sup) / len(objetivo_serie) > 0.60:
                        es_bimodal = True

                resumen["analisis_predictivo"]["objetivo"]["estadisticas_objetivo"] = {
                    "media": round(media, 3),
                    "desviacion_estandar": round(std, 3),
                    "minimo": round(min_val, 3),
                    "maximo": round(max_val, 3),
                    "coeficiente_variacion_cv": round(std / media, 3) if media != 0 else 0.0,
                    "distribucion_bimodal_en_U": es_bimodal # Inyectamos la alerta al JSON
                }
            else:
                dist = objetivo_serie.value_counts(normalize=True)
                if len(dist) > 15:
                    dist_dict = dist.head(10).to_dict()
                    dist_dict["[OTRAS_CLASES_AGRUPADAS]"] = float(dist.iloc[10:].sum())
                    resumen["alertas_criticas"].append(f"Alta cardinalidad en el objetivo '{columna_objetivo}' ({len(dist)} clases). El JSON fue truncado.")
                else:
                    dist_dict = dist.to_dict()
                
                # Entropía de Shannon (Balance de Clases)
                probabilidades = np.array(list(dist_dict.values()))
                entropia = -np.sum(probabilidades * np.log2(probabilidades + 1e-9))
                
                resumen["analisis_predictivo"]["objetivo"]["distribucion_pct"] = {str(k): round(float(v) * 100, 2) for k, v in dist_dict.items()}
                resumen["analisis_predictivo"]["objetivo"]["entropia_shannon"] = round(float(entropia), 3)

        # ==========================================
        # 1. ENRUTADOR NLP Y ANÁLISIS DE TEXTO (GRANULAR)
        # ==========================================
        cols_text_detectadas = []
        for col in df.select_dtypes(include=['object', 'string']).columns:
            if col == columna_objetivo:
                continue
            
            unicos_pct = df[col].nunique() / len(df)
            muestra = df[col].dropna().astype(str).head(100)
            
            if not muestra.empty:
                lon_chars = muestra.apply(len).mean()
                # [BLINDAJE 1] Filtro Semántico para evitar procesar URLs o hashes
                lon_words = muestra.apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0).mean()
                
                if ((unicos_pct > 0.3 and lon_chars > 15) or (lon_chars > 50)) and (lon_words < 15):
                    cols_text_detectadas.append(col)
        
        if cols_text_detectadas:
            resumen["analisis_nlp"] = {
                "tarea_inferida": tarea_global,
                "columnas_analizadas": {}
            }
            
            for col_txt in cols_text_detectadas:
                textos_validos = df[col_txt].dropna().astype(str)
                textos_validos = textos_validos[textos_validos.str.strip().astype(bool)]
                
                if not textos_validos.empty:
                    conteo_palabras = textos_validos.apply(lambda x: len(x.split()))
                    conteo_caracteres = textos_validos.apply(len)
                    
                    info_columna = {
                        "topologia_tensores": {
                            "palabras_promedio": round(float(conteo_palabras.mean()), 2),
                            "caracteres_promedio": round(float(conteo_caracteres.mean()), 2),
                            "percentil_50_palabras": int(conteo_palabras.quantile(0.50)),
                            "percentil_90_palabras": int(conteo_palabras.quantile(0.90)),
                            "percentil_99_palabras": int(conteo_palabras.quantile(0.99))
                        }
                    }
                    
                    # [BLINDAJE 3] Alineación de índices segura para correlación NLP
                    if tarea_global == "Regresión Continua":
                        idx_comunes = conteo_palabras.index.intersection(objetivo_serie.index)
                        if len(idx_comunes) > 5:
                            corr_longitud = conteo_palabras.loc[idx_comunes].corr(objetivo_serie.loc[idx_comunes])
                            info_columna["correlacion_longitud_vs_score"] = round(float(corr_longitud), 3) if not pd.isna(corr_longitud) else 0.0
                    
                    elif tarea_global == "Segmentación de Tokens":
                        obj_str = objetivo_serie.astype(str)
                        todos_los_tags = obj_str.str.replace(r'\[|\]|\'|\"|\,', ' ', regex=True).str.split(expand=True).stack()
                        dist_tags = todos_los_tags.value_counts(normalize=True).head(6).to_dict()
                        info_columna["distribucion_tags_individuales"] = {str(k): round(float(v)*100, 2) for k, v in dist_tags.items()}
                        
                    # Riqueza Léxica (TTR) y Complejidad Morfológica
                    muestra_str = textos_validos.sample(min(len(textos_validos), 1000), random_state=42)
                    palabras_totales = " ".join(muestra_str.tolist()).lower().split()
                    
                    if palabras_totales:
                        ttr = len(set(palabras_totales)) / len(palabras_totales)
                        len_promedio_palabra = np.mean([len(p) for p in palabras_totales])
                        info_columna["riqueza_lexica_ttr"] = round(float(ttr), 3)
                        info_columna["longitud_promedio_palabra"] = round(float(len_promedio_palabra), 2)
                        
                    resumen["analisis_nlp"]["columnas_analizadas"][col_txt] = info_columna

        # C. Variables Categóricas Tradicionales
        cols_categoricas = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in cols_text_detectadas and c != columna_objetivo]
        for col in cols_categoricas:
            unicos = df[col].nunique()
            info_col = {"valores_unicos": unicos, "porcentaje_nulos_real": round((int(df[col].isnull().sum()) / len(df)) * 100, 2)}
            top_valores = df[col].value_counts(normalize=True).head(3).to_dict()
            info_col["top_3_frecuencias_pct"] = {str(k): round(float(v) * 100, 2) for k, v in top_valores.items()}
            resumen["variables_categoricas"][col] = info_col

        # ==========================================
        # 1.5 RADAR DE ACUERDO ENTRE ANOTADORES
        # ==========================================
        resumen["analisis_acuerdo_anotadores"] = {}
        keywords_anotadores = ['annotator', 'rater', 'juez', 'anotador', 'coder', 'labeler']
        cols_anotadores = [col for col in df.columns if any(kw in col.lower() for kw in keywords_anotadores) and col != columna_objetivo]
        
        if len(cols_anotadores) >= 2:
            resumen["analisis_acuerdo_anotadores"]["columnas_detectadas"] = cols_anotadores
            resumen["analisis_acuerdo_anotadores"]["pares_evaluados"] = {}
            for par in itertools.combinations(cols_anotadores, 2):
                col_A, col_B = par
                df_pares = df[[col_A, col_B]].dropna()
                if len(df_pares) >= 10: 
                    try:
                        nombre_par = f"{col_A} vs {col_B}"
                        if tarea_global == "Regresión Continua":
                            s_A = pd.to_numeric(df_pares[col_A], errors='coerce').dropna()
                            s_B = pd.to_numeric(df_pares[col_B], errors='coerce').dropna()
                            idx_comunes = s_A.index.intersection(s_B.index)
                            if len(idx_comunes) >= 10:
                                val_A, val_B = s_A[idx_comunes], s_B[idx_comunes]
                                corr = 1.0 if val_A.std() == 0 or val_B.std() == 0 else val_A.corr(val_B)
                                corr = 0.0 if pd.isna(corr) else corr
                                fiabilidad = "Acuerdo Fuerte" if corr > 0.7 else ("Acuerdo Moderado" if corr > 0.4 else "Acuerdo Pobre / Ruido")
                                resumen["analisis_acuerdo_anotadores"]["pares_evaluados"][nombre_par] = {"metrica_usada": "Correlacion_Pearson", "score": round(float(corr), 3), "diagnostico_fiabilidad": fiabilidad}
                        elif tarea_global == "Segmentación de Tokens":
                            all_t_A, all_t_B = [], []
                            for idx, row in df_pares.iterrows():
                                t_A = str(row[col_A]).replace('[', '').replace(']', '').replace("'", '').replace('"', '').replace(',', ' ').split()
                                t_B = str(row[col_B]).replace('[', '').replace(']', '').replace("'", '').replace('"', '').replace(',', ' ').split()
                                min_t = min(len(t_A), len(t_B))
                                all_t_A.extend(t_A[:min_t])
                                all_t_B.extend(t_B[:min_t])
                            if len(all_t_A) > 50:
                                kappa = cohen_kappa_score(all_t_A, all_t_B)
                                resumen["analisis_acuerdo_anotadores"]["pares_evaluados"][nombre_par] = {"metrica_usada": "Kappa_Cohen_por_Token", "score": round(float(kappa), 3), "tokens_evaluados": len(all_t_A), "diagnostico_fiabilidad": "Confiable" if kappa > 0.60 else "Ambigüedad Severa"}
                        else:
                            kappa = cohen_kappa_score(df_pares[col_A].astype(str), df_pares[col_B].astype(str))
                            fiabilidad = "Desacuerdo Sistemático" if kappa < 0 else ("Acuerdo Pobre" if kappa <= 0.20 else ("Acuerdo Sustancial" if kappa <= 0.80 else "Acuerdo Casi Perfecto"))
                            resumen["analisis_acuerdo_anotadores"]["pares_evaluados"][nombre_par] = {"metrica_usada": "Kappa_Cohen_Documento", "score": round(float(kappa), 3), "muestras_evaluadas": len(df_pares), "diagnostico_fiabilidad": fiabilidad}
                    except Exception: pass

        # ==========================================
        # 2. ANÁLISIS PREDICTIVO NUMÉRICO
        # ==========================================
        cols_num = df.select_dtypes(include=[np.number]).columns
        cols_pred_num = [c for c in cols_num if c != columna_objetivo]
        
        if es_regresion_tabular:
            corr_pearson = df[cols_num].corr(method='pearson')[columna_objetivo].drop(columna_objetivo)
            resumen["analisis_predictivo"]["pearson_fuertes"] = corr_pearson[abs(corr_pearson) > 0.3].round(3).to_dict()
            corr_spearman = df[cols_num].corr(method='spearman')[columna_objetivo].drop(columna_objetivo)
            resumen["analisis_predictivo"]["spearman_fuertes"] = corr_spearman[abs(corr_spearman) > 0.3].round(3).to_dict()

        if len(cols_pred_num) > 0:
            df_limpio_mi = df[cols_pred_num + [columna_objetivo]].dropna()
            if not df_limpio_mi.empty:
                X_mi, y_mi = df_limpio_mi[cols_pred_num], df_limpio_mi[columna_objetivo]
                try:
                    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42) if es_regresion_tabular else mutual_info_classif(X_mi, y_mi, random_state=42)
                    mi_filtrado = pd.Series(mi_scores, index=X_mi.columns)[pd.Series(mi_scores, index=X_mi.columns) > 0.05].round(3).to_dict()
                    resumen["analisis_predictivo"]["informacion_mutua_relevante"] = {k: float(v) for k, v in mi_filtrado.items()}
                except Exception: pass

            df_preds = df[cols_pred_num].dropna()
            if not df_preds.empty:
                try:
                    X_vif = df_preds.loc[:, df_preds.var() > 0]
                    if len(X_vif.columns) >= 2 and len(X_vif) > len(X_vif.columns):
                        vifs = np.diag(np.linalg.inv(X_vif.corr().values))
                        resumen["analisis_predictivo"]["multicolinealidad_vif_severa"] = {col: round(float(vif), 2) for col, vif in zip(X_vif.columns, vifs) if vif > 10 and not np.isinf(vif)}
                except np.linalg.LinAlgError:
                    resumen["alertas_criticas"].append("Matriz singular detectada en VIF. Existe correlación perfecta (1.0).")
                except Exception: pass

            resumen["analisis_predictivo"]["valores_atipicos_severos"] = {}
            for col in cols_pred_num:
                s = df[col].dropna()
                if len(s) > 10 and s.nunique() > 2:
                    Q1, Q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                    IQR = Q3 - Q1
                    media, std = float(s.mean()), float(s.std())
                    
                    pct_iqr, pct_z = 0.0, 0.0
                    if IQR > 0: pct_iqr = round(float((len(s[(s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)]) / len(s)) * 100), 2)
                    if std > 0: pct_z = round(float((len(s[(s < media - 3 * std) | (s > media + 3 * std)]) / len(s)) * 100), 2)
                    
                    if pct_iqr > 5.0 or pct_z > 0.0:
                        # [BLINDAJE 2] Protección anti-NaN para Skewness y Kurtosis
                        skew_val = float(s.skew())
                        kurt_val = float(s.kurtosis())
                        
                        resumen["analisis_predictivo"]["valores_atipicos_severos"][col] = {
                            "porcentaje_outliers_iqr": pct_iqr,
                            "porcentaje_outliers_zscore": pct_z,
                            "estadisticas_base": {
                                "media": round(media, 2), "desviacion_estandar": round(std, 2),
                                "valor_minimo_real": round(float(s.min()), 2), "valor_maximo_real": round(float(s.max()), 2),
                                "asimetria_skewness": round(skew_val, 2) if not pd.isna(skew_val) else 0.0, 
                                "curtosis_kurtosis": round(kurt_val, 2) if not pd.isna(kurt_val) else 0.0, 
                                "escasez_ceros_pct": round(float((s == 0).mean() * 100), 2)
                            }
                        }

        if not es_regresion_tabular:
            resumen["analisis_predictivo"]["separabilidad_clases"] = {}
            y_completo = df[columna_objetivo].dropna()
            if 1 < y_completo.nunique() <= 15 and len(cols_pred_num) >= 2:
                df_sep = df[cols_pred_num + [columna_objetivo]].dropna()
                if len(df_sep) > 50:
                    clases_validas = df_sep[columna_objetivo].value_counts()[df_sep[columna_objetivo].value_counts() > 1].index
                    df_sep = df_sep[df_sep[columna_objetivo].isin(clases_validas)]
                    if df_sep[columna_objetivo].nunique() > 1:
                        df_sample = df_sep.sample(n=min(len(df_sep), 3000), random_state=42)
                        clases_seguras = df_sample[columna_objetivo].value_counts()[df_sample[columna_objetivo].value_counts() > 1].index
                        df_sample = df_sample[df_sample[columna_objetivo].isin(clases_seguras)]
                        if df_sample[columna_objetivo].nunique() > 1:
                            try:
                                X_pca = PCA(n_components=min(3, len(cols_pred_num)), random_state=42).fit_transform(StandardScaler().fit_transform(df_sample[cols_pred_num]))
                                score_silueta = float(silhouette_score(X_pca, df_sample[columna_objetivo], random_state=42))
                                estado_manifold = "Altamente Separable" if score_silueta > 0.40 else ("Superposición Moderada" if score_silueta > 0.15 else "Caos Espacial")
                                resumen["analisis_predictivo"]["separabilidad_clases"] = {
                                    "score_silueta_pca": round(score_silueta, 3),
                                    "diagnostico_topologico": estado_manifold
                                }
                            except Exception: pass

        # ==========================================
        # 3. ANÁLISIS DE SILOS (APRENDIZAJE FEDERADO)
        # ==========================================
        if columna_nodo and columna_nodo in df.columns:
            nodos_validos = df[columna_nodo].dropna()
            if nodos_validos.nunique() > 50:
                resumen["alertas_criticas"].append(f"La columna de nodo '{columna_nodo}' tiene {nodos_validos.nunique()} valores únicos. Se abortó el análisis federado por riesgo de OOM.")
            elif nodos_validos.nunique() < 2:
                resumen["alertas_criticas"].append(f"La columna '{columna_nodo}' tiene un solo nodo válido. No se puede realizar análisis federado multicéntrico.")
            else:
                resumen["analisis_federado_nodos"] = {
                    "distribucion_muestras": {str(k): round(float(v) * 100, 2) for k, v in nodos_validos.value_counts(normalize=True).to_dict().items()},
                    "divergencia_objetivo_por_nodo": {}
                }
                if columna_objetivo and columna_objetivo in df.columns:
                    for nodo in nodos_validos.unique():
                        df_nodo = df[df[columna_nodo] == nodo]
                        if es_regresion_tabular:
                            serie_obj_nodo = df_nodo[columna_objetivo].dropna()
                            resumen["analisis_federado_nodos"]["divergencia_objetivo_por_nodo"][str(nodo)] = {"media_objetivo": round(float(serie_obj_nodo.mean()), 2) if not serie_obj_nodo.empty else 0.0}
                        else:
                            resumen["analisis_federado_nodos"]["divergencia_objetivo_por_nodo"][str(nodo)] = {str(k): round(float(v) * 100, 2) for k, v in df_nodo[columna_objetivo].value_counts(normalize=True).head(10).to_dict().items()}

    except Exception as e:
        resumen["alertas_criticas"].append(f"Error general en el procesamiento del DataFrame: {str(e)}")

    return json.dumps(resumen, indent=4, ensure_ascii=False)
