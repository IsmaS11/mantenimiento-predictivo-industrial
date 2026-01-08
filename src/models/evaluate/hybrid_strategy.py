import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_hybrid_metrics(
    models_dict: Dict[str, BaseEstimator],
    thresholds_dict: Dict[str, float],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    costs: Dict[str, float],
    rule_exceptions: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calcula métricas financieras aplicando una estrategia híbrida robusta.
    Corrige posibles errores de dimensión en matriz de confusión.
    """
    if rule_exceptions is None:
        rule_exceptions = {}

    reporte_hibrido = []
    
    acc_ahorro = 0.0
    acc_costo_rtf = 0.0
    acc_costo_estrategia = 0.0

    # Iteramos sobre las columnas presentes en y_test
    target_cols = list(y_test.columns)
    
    logging.info("--- Iniciando cálculo de Estrategia Híbrida ---")

    for falla in target_cols:
        try:
            # --- 1. SELECCIÓN DE ESTRATEGIA ---
            y_pred_final = None
            desc_estrategia = ""

            # CASO A: Regla Manual (Prioridad Alta)
            if falla in rule_exceptions:
                rule = rule_exceptions[falla]
                col_name = rule.get('col')
                limit = rule.get('threshold')
                
                if col_name not in X_test.columns:
                    logging.error(f"Columna '{col_name}' no encontrada en X_test para la regla de {falla}.")
                    continue

                # Aplicar regla
                y_pred_final = (X_test[col_name] > limit).astype(int)
                desc_estrategia = f"Regla Fija (>{limit} {col_name})"
            
            # CASO B: Modelo de IA
            elif falla in models_dict and falla in thresholds_dict:
                modelo = models_dict[falla]
                umbral = thresholds_dict[falla]
                
                # Manejo robusto de predict_proba vs predict
                if hasattr(modelo, "predict_proba"):
                    # Tomamos la columna 1 (probabilidad positiva)
                    probs = modelo.predict_proba(X_test)[:, 1]
                    y_pred_final = (probs >= umbral).astype(int)
                else:
                    y_pred_final = modelo.predict(X_test)
                
                desc_estrategia = f"XGBoost (Umbral {umbral:.2f})"
            
            else:
                logging.warning(f"Se omite '{falla}': No tiene regla ni modelo+umbral definidos.")
                continue

            # --- 2. CÁLCULO DE COSTOS (CORREGIDO) ---
            # labels=[0, 1] asegura que siempre devuelva 2x2 (TN, FP, FN, TP)
            # incluso si falta alguna clase en los datos de test.
            cm = confusion_matrix(y_test[falla], y_pred_final, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            costo_estrategia = (tp * costs['preventivo']) + \
                               (fp * costs['inspeccion']) + \
                               (fn * costs['falla']) + \
                               (tn * costs['normal'])

            # Costo Base Run-to-Failure
            fallas_reales = fn + tp
            costo_rtf = fallas_reales * costs['falla']

            ahorro = costo_rtf - costo_estrategia

            # --- 3. GUARDAR RESULTADOS ---
            reporte_hibrido.append({
                'Tipo de Falla': falla,
                'Estrategia Usada': desc_estrategia,
                'Costo "Dejar Romper"': costo_rtf,
                'Costo Estrategia': costo_estrategia,
                'Ahorro Neto': ahorro,
                'Fallas Evitadas (TP)': tp,
                'Fallas No Detectadas (FN)': fn,
                'Inspecciones Extra (FP)': fp
            })

            acc_ahorro += ahorro
            acc_costo_rtf += costo_rtf
            acc_costo_estrategia += costo_estrategia

        except Exception as e:
            logging.error(f"Error crítico procesando '{falla}': {e}")
            continue

    # Resultados Finales
    df_hibrido = pd.DataFrame(reporte_hibrido)
    
    # KPIs Globales (Evitando división por cero)
    roi_hibrido = (acc_ahorro / acc_costo_estrategia * 100) if acc_costo_estrategia > 0 else 0.0
    reduccion_paradas = ((acc_costo_rtf - acc_costo_estrategia) / acc_costo_rtf * 100) if acc_costo_rtf > 0 else 0.0

    metrics = {
        'costo_rtf_total': acc_costo_rtf,
        'costo_estrategia_total': acc_costo_estrategia,
        'ahorro_total': acc_ahorro,
        'roi': roi_hibrido,
        'reduccion_paradas': reduccion_paradas
    }

    return df_hibrido, metrics

def print_hybrid_report(df_hibrido: pd.DataFrame, metrics: Dict[str, float]) -> None:
    """
    Imprime el reporte ejecutivo con el formato solicitado.
    Compatible con Jupyter (visual) y Terminal (texto).
    """
    print(f"--- REPORTE FINANCIERO: ESTRATEGIA HÍBRIDA (REGLA + IA) ---")
    
    # Intentamos usar display de Jupyter, si falla, usamos print normal
    try:
        from IPython.display import display
        # Aplicamos formato de moneda sin decimales para limpieza visual
        styler = df_hibrido.style.format({
            'Costo "Dejar Romper"': '${:,.0f}',
            'Costo Estrategia': '${:,.0f}',
            'Ahorro Neto': '${:,.0f}'
        }).background_gradient(subset=['Ahorro Neto'], cmap='Greens')
        display(styler)
    except (ImportError, ModuleNotFoundError):
        # Fallback para scripts ejecutados en terminal
        print(df_hibrido.to_string())

    print(f"\n==========================================================")
    print(f"RESUMEN EJECUTIVO DE PLANTA (Estrategia Mixta)")
    print(f"==========================================================")
    print(f"Costo Actual (Reactivo/RTF):         ${metrics['costo_rtf_total']:,.2f}")
    print(f"Costo Propuesto (Híbrido):           ${metrics['costo_estrategia_total']:,.2f}")
    print(f"----------------------------------------------------------")
    print(f"AHORRO TOTAL ANUAL ESTIMADO:         ${metrics['ahorro_total']:,.2f}")
    print(f"RETORNO DE INVERSIÓN (ROI):          {metrics['roi']:.1f}%")
    print(f"REDUCCIÓN DE PARADAS NO PLANIFICADAS: {metrics['reduccion_paradas']:.1f}%")
    print(f"==========================================================")

if __name__ == "__main__":
    # --- PRUEBA DE ROBUSTEZ ---
    print("🔧 Ejecutando prueba de robustez de hybrid_strategy.py...")
    from sklearn.dummy import DummyClassifier
    
    # 1. Creamos datos donde SOLO hay ceros (caso borde común)
    X_mock = pd.DataFrame({'temp': [100, 100, 100], 'tool_wear_min': [10, 20, 30]})
    y_mock = pd.DataFrame({'falla_rara': [0, 0, 0]}) # Nadie falla
    
    # 2. Configuración
    models = {'falla_rara': DummyClassifier(strategy='constant', constant=0)}
    thresholds = {'falla_rara': 0.5}
    costs = {'preventivo': 500, 'inspeccion': 150, 'falla': 10000, 'normal': 0}
    
    try:
        df, mets = calculate_hybrid_metrics(models, thresholds, X_mock, y_mock, costs)
        print("✅ Prueba superada: El código no falló con clases faltantes.")
        print_hybrid_report(df, mets)
    except Exception as e:
        print(f"❌ La prueba falló: {e}")