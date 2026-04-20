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
        # ==========================================
        # 0. INTEGRIDAD ESTRUCTURAL (HÍBRIDA)
        # ==========================================
        posibles_ids = [c for c in df.columns if str(c).lower() in ['id', 'uuid', 'index', 'pt_id', 'paciente_id', 'user_id'] or str(c).lower().endswith('_id')]
        if posibles_ids:
            resumen["alertas_criticas"].append(f"Se detectaron posibles identificadores directos: {posibles_ids}. Revisa si deben excluirse para evitar sobreajuste o fugas de privacidad.")

        resumen["dimensiones_dataset"] = {
            "total_filas": int(len(df)),
            "total_columnas": int(len(df.columns)),
            "ratio_filas_columnas": round(float(len(df) / len(df.columns)), 2) if len(df.columns) > 0 else 0.0,
            "pct_duplicados": round(float((df.duplicated().sum() / len(df)) * 100), 2) if len(df) > 0 else 0.0
        }

        # ==========================================
        # 1. PERFILADO DEL OBJETIVO Y ENRUTADOR DE TAREA
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
                
                rango = max_val - min_val
                es_bimodal = False
                if rango > 0:
                    extremo_inf = objetivo_serie[objetivo_serie <= min_val + 0.15 * rango].count()
                    extremo_sup = objetivo_serie[objetivo_serie >= max_val - 0.15 * rango].count()
                    if (extremo_inf + extremo_sup) / len(objetivo_serie) > 0.60:
                        es_bimodal = True

                resumen["analisis_predictivo"]["objetivo"]["estadisticas_objetivo"] = {
                    "media": round(media, 3), "desviacion_estandar": round(std, 3),
                    "minimo": round(min_val, 3), "maximo": round(max_val, 3),
                    "coeficiente_variacion_cv": round(std / media, 3) if media != 0 else 0.0,
                    "distribucion_bimodal_en_U": es_bimodal
                }
            else:
                dist = objetivo_serie.value_counts(normalize=True)
                if len(dist) > 15:
                    dist_dict = dist.head(10).to_dict()
                    dist_dict["[OTRAS_CLASES_AGRUPADAS]"] = float(dist.iloc[10:].sum())
                    resumen["alertas_criticas"].append(f"Alta cardinalidad en el objetivo '{columna_objetivo}' ({len(dist)} clases).")
                else:
                    dist_dict = dist.to_dict()
                
                probabilidades = np.array(list(dist_dict.values()))
                entropia = -np.sum(probabilidades * np.log2(probabilidades + 1e-9))
                
                resumen["analisis_predictivo"]["objetivo"]["distribucion_pct"] = {str(k): round(float(v) * 100, 2) for k, v in dist_dict.items()}
                resumen["analisis_predictivo"]["objetivo"]["entropia_shannon"] = round(float(entropia), 3)

            # ==========================================
            # 1.5 CÁLCULO DE LÍNEA BASE (DUMMY MODELS)
            # ==========================================
            linea_base = {}
            if tarea_global == "Regresión Continua":
                y_num = pd.to_numeric(objetivo_serie, errors='coerce').dropna()
                if len(y_num) > 0:
                    media_y = float(y_num.mean())
                    mediana_y = float(y_num.median())
                    mae_base = float((y_num - mediana_y).abs().mean())
                    rmse_base = float(np.sqrt(((y_num - media_y)**2).mean()))
                    linea_base = {
                        "estrategia_prediccion_mediana": {"mae_esperado": round(mae_base, 4)},
                        "estrategia_prediccion_media": {"rmse_esperado": round(rmse_base, 4)}
                    }
            elif tarea_global == "Clasificación / Discreta":
                if not objetivo_serie.empty:
                    frecuencias = objetivo_serie.value_counts(normalize=True)
                    prev_mayoritaria = float(frecuencias.max())
                    prev_minoritaria = float(frecuencias.min())
                    linea_base = {
                        "estrategia_mayoritaria": {
                            "accuracy_esperado": round(prev_mayoritaria, 4),
                            "recall_esperado": 0.0,
                            "f2_score_esperado": 0.0
                        },
                        "estrategia_aleatoria_estratificada": {
                            "pr_auc_esperado_clase_minoritaria": round(prev_minoritaria, 4)
                        }
                    }
            
            if linea_base:
                resumen["analisis_predictivo"]["linea_base_predictiva"] = linea_base

        # ==========================================
        # 2. ENRUTADOR NLP Y RIQUEZA LÉXICA
        # ==========================================
        cols_text_detectadas = []
        for col in df.select_dtypes(include=['object', 'string']).columns:
            if col == columna_objetivo: continue
            
            unicos_pct = df[col].nunique() / len(df)
            muestra = df[col].dropna().astype(str).head(100)
            
            if not muestra.empty:
                lon_chars = muestra.apply(len).mean()
                lon_words = muestra.apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0).mean()
                if ((unicos_pct > 0.3 and lon_chars > 15) or (lon_chars > 50)) and (lon_words < 15):
                    cols_text_detectadas.append(col)
        
        if cols_text_detectadas:
            resumen["analisis_nlp"] = {"tarea_inferida": tarea_global, "columnas_analizadas": {}}
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
                        
                    muestra_str = textos_validos.sample(min(len(textos_validos), 1000), random_state=42)
                    palabras_totales = " ".join(muestra_str.tolist()).lower().split()
                    
                    if palabras_totales:
                        ttr = len(set(palabras_totales)) / len(palabras_totales)
                        info_columna["riqueza_lexica_ttr"] = round(float(ttr), 3)
                        
                    resumen["analisis_nlp"]["columnas_analizadas"][col_txt] = info_columna

        # ==========================================
        # 3. CATEGÓRICAS Y ALTA CARDINALIDAD
        # ==========================================
        cols_categoricas = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in cols_text_detectadas and c != columna_objetivo]
        resumen["alta_cardinalidad"] = {} 
        
        for col in cols_categoricas:
            unicos = df[col].nunique()
            info_col = {"valores_unicos": int(unicos), "porcentaje_nulos_real": round((int(df[col].isnull().sum()) / len(df)) * 100, 2)}
            top_valores = df[col].value_counts(normalize=True).head(3).to_dict()
            info_col["top_3_frecuencias_pct"] = {str(k): round(float(v) * 100, 2) for k, v in top_valores.items()}
            resumen["variables_categoricas"][col] = info_col
            
            if unicos > 50:
                resumen["alta_cardinalidad"][col] = int(unicos)

        # ==========================================
        # 4. AUDITORÍA DE EQUIDAD (TOPOLOGÍA ABSTRACTA)
        # ==========================================
        resumen["auditoria_equidad"] = {}
        if tarea_global == "Clasificación / Discreta" and columna_objetivo in df.columns:
            try:
                clase_positiva = str(df[columna_objetivo].value_counts().idxmin())
            except Exception:
                clase_positiva = None
                
            if clase_positiva is not None:
                segmentos_potenciales = []
                for c in df.columns:
                    if c == columna_objetivo: continue
                    unicos = df[c].nunique()
                    es_cat_texto = pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(df[c])
                    es_num_discreto = pd.api.types.is_numeric_dtype(df[c]) and 1 < unicos <= 5
                    
                    if (es_cat_texto and 1 < unicos <= 12) or es_num_discreto:
                        segmentos_potenciales.append((c, unicos))
                
                segmentos_potenciales.sort(key=lambda x: x[1])
                top_5_segmentos = [x[0] for x in segmentos_potenciales[:5]]
                
                for col_seg in top_5_segmentos:
                    ct = pd.crosstab(df[col_seg], df[columna_objetivo].astype(str), normalize='index')
                    if clase_positiva in ct.columns:
                        tasas = ct[clase_positiva].to_dict()
                        resumen["auditoria_equidad"][f"Tasa de '{clase_positiva}' por {col_seg}"] = {
                            str(k): round(float(v) * 100, 2) for k, v in tasas.items()
                        }

        # ==========================================
        # 5. FIABILIDAD HUMANA (ANOTADORES)
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
                        kappa = cohen_kappa_score(df_pares[col_A].astype(str), df_pares[col_B].astype(str))
                        fiabilidad = "Desacuerdo Sistemático" if kappa < 0 else ("Acuerdo Pobre" if kappa <= 0.20 else ("Acuerdo Sustancial" if kappa <= 0.80 else "Acuerdo Casi Perfecto"))
                        resumen["analisis_acuerdo_anotadores"]["pares_evaluados"][nombre_par] = {"metrica_usada": "Kappa_Cohen_Documento", "score": round(float(kappa), 3), "diagnostico_fiabilidad": fiabilidad}
                    except Exception: pass

        # ==========================================
        # 6. ANÁLISIS PREDICTIVO NUMÉRICO (BLINDADO)
        # ==========================================
        cols_num = df.select_dtypes(include=[np.number]).columns
        cols_pred_num = [c for c in cols_num if c != columna_objetivo]

        if len(cols_pred_num) > 0:
            # BLINDAJE: Imputación rápida para Matemáticas (Evita Listwise Deletion)
            df_math_safe = df[cols_pred_num + [columna_objetivo]].copy()
            df_math_safe = df_math_safe.dropna(subset=[columna_objetivo])
            
            # ELIMINAR COLUMNAS FANTASMA PARA EVITAR NaNs POST-IMPUTACIÓN
            df_math_safe = df_math_safe.dropna(axis=1, how='all')
            cols_pred_num_safe = [c for c in cols_pred_num if c in df_math_safe.columns]
            
            if not df_math_safe.empty and len(cols_pred_num_safe) > 0:
                df_math_safe[cols_pred_num_safe] = df_math_safe[cols_pred_num_safe].fillna(df_math_safe[cols_pred_num_safe].median())

                X_mi, y_mi = df_math_safe[cols_pred_num_safe], df_math_safe[columna_objetivo]
                try:
                    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42) if es_regresion_tabular else mutual_info_classif(X_mi, y_mi, random_state=42)
                    mi_filtrado = pd.Series(mi_scores, index=X_mi.columns)[pd.Series(mi_scores, index=X_mi.columns) > 0.05].round(3).to_dict()
                    resumen["analisis_predictivo"]["informacion_mutua_relevante"] = {k: float(v) for k, v in mi_filtrado.items()}
                except Exception: pass

                try:
                    X_vif = df_math_safe[cols_pred_num_safe]
                    X_vif = X_vif.loc[:, X_vif.var() > 0]
                    if len(X_vif.columns) >= 2 and len(X_vif) > len(X_vif.columns):
                        # USO DE PSEUDO-INVERSA PARA TOLERAR COLINEALIDAD PERFECTA
                        vifs = np.diag(np.linalg.pinv(X_vif.corr().values))
                        resumen["analisis_predictivo"]["multicolinealidad_vif_severa"] = {col: round(float(vif), 2) for col, vif in zip(X_vif.columns, vifs) if vif > 10 and not np.isinf(vif)}
                except Exception: pass

            # El análisis de Outliers se hace en los datos reales (con NaNs permitidos localmente)
            resumen["analisis_predictivo"]["valores_atipicos_severos"] = {}
            for col in cols_pred_num:
                s = df[col].dropna()
                if len(s) > 10 and s.nunique() > 2:
                    Q1, Q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                    IQR = Q3 - Q1
                    media, std = float(s.mean()), float(s.std())
                    
                    pct_iqr = round(float((len(s[(s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)]) / len(s)) * 100), 2) if IQR > 0 else 0.0
                    pct_z = round(float((len(s[(s < media - 3 * std) | (s > media + 3 * std)]) / len(s)) * 100), 2) if std > 0 else 0.0
                    
                    if pct_iqr > 5.0 or pct_z > 0.0:
                        skew_val = float(s.skew())
                        kurt_val = float(s.kurtosis())
                        resumen["analisis_predictivo"]["valores_atipicos_severos"][col] = {
                            "porcentaje_outliers_iqr": pct_iqr, "porcentaje_outliers_zscore": pct_z,
                            "estadisticas_base": {
                                "media": round(media, 2), "desviacion_estandar": round(std, 2),
                                "asimetria_skewness": round(skew_val, 2) if not pd.isna(skew_val) else 0.0, 
                                "curtosis_kurtosis": round(kurt_val, 2) if not pd.isna(kurt_val) else 0.0, 
                                "escasez_ceros_pct": round(float((s == 0).mean() * 100), 2)
                            }
                        }

        # ==========================================
        # 7. SEPARABILIDAD (PCA sobre datos imputados)
        # ==========================================
        if not es_regresion_tabular:
            resumen["analisis_predictivo"]["separabilidad_clases"] = {}
            y_completo = df[columna_objetivo].dropna()
            if 1 < y_completo.nunique() <= 15 and len(cols_pred_num) >= 2:
                # Volvemos a generar df_math_safe local por si falló la sección anterior
                df_pca_safe = df[cols_pred_num + [columna_objetivo]].dropna()
                
                if not df_pca_safe.empty and len(df_pca_safe) > 50:
                    clases_validas = df_pca_safe[columna_objetivo].value_counts()[df_pca_safe[columna_objetivo].value_counts() > 1].index
                    df_sep = df_pca_safe[df_pca_safe[columna_objetivo].isin(clases_validas)]
                    
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
                                    "score_silueta_pca": round(score_silueta, 3), "diagnostico_topologico": estado_manifold
                                }
                            except Exception: pass

    except Exception as e:
        resumen["alertas_criticas"].append(f"Error general en el procesamiento del DataFrame: {str(e)}")

    return json.dumps(resumen, indent=4, ensure_ascii=False)
