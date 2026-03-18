import pandas as pd
import numpy as np
import json
import datetime
import itertools
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score, cohen_kappa_score

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
        # 1. ENRUTADOR UNIVERSAL NLP Y ANÁLISIS DE TEXTO
        # ==========================================
        cols_text_detectadas = []
        for col in df.select_dtypes(include=['object', 'string']).columns:
            if col == columna_objetivo:
                continue
            
            # BLINDAJE 1: Detección Bifásica (Varianza o Longitud)
            unicos_pct = df[col].nunique() / len(df)
            muestra_textos = df[col].dropna().astype(str).head(100)
            longitud_promedio = muestra_textos.apply(len).mean() if not muestra_textos.empty else 0
            
            # Es texto libre si: Tiene mucha varianza (>30%) O son párrafos/oraciones largas (>50 chars)
            if (unicos_pct > 0.3 and longitud_promedio > 15) or (longitud_promedio > 50):
                cols_text_detectadas.append(col)
        
        if cols_text_detectadas:
            resumen["analisis_nlp"] = {"columnas_procesadas": cols_text_detectadas}
            
            if len(cols_text_detectadas) > 1:
                df['texto_fusionado_nlp'] = df[cols_text_detectadas].fillna('').astype(str).agg(' '.join, axis=1)
                col_analisis_texto = 'texto_fusionado_nlp'
                resumen["analisis_nlp"]["tipo_entrada"] = "Multi-Texto (Cross-Encoder / Relaciones)"
            else:
                col_analisis_texto = cols_text_detectadas[0]
                resumen["analisis_nlp"]["tipo_entrada"] = "Texto Único (Clasificación / Regresión)"

            textos_validos = df[col_analisis_texto].dropna().astype(str)
            textos_validos = textos_validos[textos_validos.str.strip().astype(bool)]
            
            if not textos_validos.empty:
                conteo_palabras = textos_validos.apply(lambda x: len(x.split()))
                resumen["analisis_nlp"]["topologia_tensores"] = {
                    "palabras_promedio": round(float(conteo_palabras.mean()), 2),
                    "percentil_50_palabras": int(conteo_palabras.quantile(0.50)),
                    "percentil_90_palabras": int(conteo_palabras.quantile(0.90)),
                    "percentil_99_palabras": int(conteo_palabras.quantile(0.99))
                }
                
                objetivo_serie = df[columna_objetivo].dropna()
                
                # Caso 1: REGRESIÓN (Ej. ibm-qrank)
                if pd.api.types.is_numeric_dtype(objetivo_serie) and objetivo_serie.nunique() > 15:
                    resumen["analisis_nlp"]["tarea_inferida"] = "Regresión (Score Continuo)"
                    corr_longitud = conteo_palabras.corr(objetivo_serie)
                    resumen["analisis_nlp"]["correlacion_longitud_vs_score"] = round(float(corr_longitud), 3)
                
                # BLINDAJE 2: Detección Avanzada de SEQUENCE TAGGING (Ej. webis)
                elif objetivo_serie.dtype == 'object':
                    obj_str = objetivo_serie.astype(str)
                    es_lista_estandar = obj_str.str.contains(r'\[|\,', regex=True).mean() > 0.5
                    
                    # Detección de tags separados por espacio (BIO format)
                    palabras_por_fila = obj_str.head(100).apply(lambda x: len(x.split()))
                    es_secuencia_espacios = palabras_por_fila.mean() > 3
                    vocab_objetivo = len(set(" ".join(obj_str.head(100).tolist()).split()))
                    
                    if es_lista_estandar or (es_secuencia_espacios and vocab_objetivo < 50):
                        resumen["analisis_nlp"]["tarea_inferida"] = "Segmentación de Tokens (Sequence Tagging)"
                        todos_los_tags = obj_str.str.replace(r'\[|\]|\'|\"|\,', ' ', regex=True).str.split(expand=True).stack()
                        dist_tags = todos_los_tags.value_counts(normalize=True).head(6).to_dict()
                        resumen["analisis_nlp"]["distribucion_tags_individuales"] = {str(k): round(float(v)*100, 2) for k, v in dist_tags.items()}
                    else:
                        # Caso 3: CLASIFICACIÓN ESTÁNDAR (Ej. ukp, fs150t)
                        resumen["analisis_nlp"]["tarea_inferida"] = "Clasificación de Textos"
                else:
                    resumen["analisis_nlp"]["tarea_inferida"] = "Clasificación de Textos"
                    
                # Riqueza Léxica (TTR)
                muestra_ttr = textos_validos.sample(min(len(textos_validos), 1000), random_state=42)
                todas_las_palabras = " ".join(muestra_ttr.tolist()).lower().split()
                if len(todas_las_palabras) > 0:
                    ttr = len(set(todas_las_palabras)) / len(todas_las_palabras)
                    resumen["analisis_nlp"]["riqueza_lexica_ttr"] = round(float(ttr), 3)

        # C. Variables Categóricas Tradicionales (Tópicos, Metadatos)
        cols_categoricas = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in cols_text_detectadas and c != columna_objetivo]
        for col in cols_categoricas:
            unicos = df[col].nunique()
            info_col = {"valores_unicos": unicos, "porcentaje_nulos_real": round((int(df[col].isnull().sum()) / len(df)) * 100, 2)}
            top_valores = df[col].value_counts(normalize=True).head(3).to_dict()
            info_col["top_3_frecuencias_pct"] = {str(k): round(float(v) * 100, 2) for k, v in top_valores.items()}
            resumen["variables_categoricas"][col] = info_col

        # ==========================================
        # 1.5 RADAR DE ACUERDO ENTRE ANOTADORES (ADAPTATIVO)
        # ==========================================
        resumen["analisis_acuerdo_anotadores"] = {}
        
        keywords_anotadores = ['annotator', 'rater', 'juez', 'anotador', 'coder', 'labeler']
        cols_anotadores = [col for col in df.columns if any(kw in col.lower() for kw in keywords_anotadores) and col != columna_objetivo]
        
        if len(cols_anotadores) >= 2:
            resumen["analisis_acuerdo_anotadores"]["columnas_detectadas"] = cols_anotadores
            resumen["analisis_acuerdo_anotadores"]["pares_evaluados"] = {}
            
            tarea_inferida = resumen.get("analisis_nlp", {}).get("tarea_inferida", "Clasificación")
            
            for par in itertools.combinations(cols_anotadores, 2):
                col_A, col_B = par
                df_pares = df[[col_A, col_B]].dropna()
                
                if len(df_pares) >= 10: 
                    try:
                        nombre_par = f"{col_A} vs {col_B}"
                        
                        # ------------------------------------------------
                        # RUTA 1: REGRESIÓN (Correlación con Blindaje NaN)
                        # ------------------------------------------------
                        if "Regresión" in tarea_inferida:
                            s_A = pd.to_numeric(df_pares[col_A], errors='coerce').dropna()
                            s_B = pd.to_numeric(df_pares[col_B], errors='coerce').dropna()
                            idx_comunes = s_A.index.intersection(s_B.index)
                            
                            if len(idx_comunes) >= 10:
                                val_A, val_B = s_A[idx_comunes], s_B[idx_comunes]
                                
                                # Prevención de División por Cero (Varianza 0)
                                if val_A.std() == 0 or val_B.std() == 0:
                                    corr = 1.0 if val_A.equals(val_B) else 0.0
                                else:
                                    corr = val_A.corr(val_B)
                                    if pd.isna(corr): corr = 0.0
                                    
                                fiabilidad = "Acuerdo Fuerte" if corr > 0.7 else ("Acuerdo Moderado" if corr > 0.4 else "Acuerdo Pobre / Ruido")
                                resumen["analisis_acuerdo_anotadores"]["pares_evaluados"][nombre_par] = {
                                    "metrica_usada": "Correlacion_Pearson",
                                    "score": round(float(corr), 3),
                                    "diagnostico_fiabilidad": fiabilidad
                                }
                                
                        # ------------------------------------------------
                        # RUTA 2: SEGMENTACIÓN DE TOKENS (Aislamiento por Fila)
                        # ------------------------------------------------
                        elif "Segmentación" in tarea_inferida:
                            all_t_A, all_t_B = [], []
                            
                            for idx, row in df_pares.iterrows():
                                # Limpieza severa de caracteres de listas/tuplas
                                t_A = str(row[col_A]).replace('[', '').replace(']', '').replace("'", '').replace('"', '').replace(',', ' ').split()
                                t_B = str(row[col_B]).replace('[', '').replace(']', '').replace("'", '').replace('"', '').replace(',', ' ').split()
                                
                                # Aislamiento de error: emparejamos hasta donde ambas listas coincidan en ESA oración
                                min_t = min(len(t_A), len(t_B))
                                all_t_A.extend(t_A[:min_t])
                                all_t_B.extend(t_B[:min_t])
                                
                            if len(all_t_A) > 50:
                                kappa = cohen_kappa_score(all_t_A, all_t_B)
                                resumen["analisis_acuerdo_anotadores"]["pares_evaluados"][nombre_par] = {
                                    "metrica_usada": "Kappa_Cohen_por_Token",
                                    "score": round(float(kappa), 3),
                                    "tokens_evaluados": len(all_t_A),
                                    "diagnostico_fiabilidad": "Confiable" if kappa > 0.60 else "Ambigüedad Severa"
                                }
                                
                        # ------------------------------------------------
                        # RUTA 3: CLASIFICACIÓN ESTÁNDAR
                        # ------------------------------------------------
                        else:
                            kappa = cohen_kappa_score(df_pares[col_A].astype(str), df_pares[col_B].astype(str))
                            
                            if kappa < 0: fiabilidad = "Desacuerdo Sistemático"
                            elif kappa <= 0.20: fiabilidad = "Acuerdo Pobre (Inviable)"
                            elif kappa <= 0.40: fiabilidad = "Acuerdo Justo (Ruido Severo)"
                            elif kappa <= 0.60: fiabilidad = "Acuerdo Moderado"
                            elif kappa <= 0.80: fiabilidad = "Acuerdo Sustancial (Dataset Confiable)"
                            else: fiabilidad = "Acuerdo Casi Perfecto"
                                
                            resumen["analisis_acuerdo_anotadores"]["pares_evaluados"][nombre_par] = {
                                "metrica_usada": "Kappa_Cohen_Documento",
                                "score": round(float(kappa), 3),
                                "muestras_evaluadas": len(df_pares),
                                "diagnostico_fiabilidad": fiabilidad
                            }
                    except Exception:
                        pass
        
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

