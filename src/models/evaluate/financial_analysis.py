import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_financial_metrics(
    models_dict: Dict[str, BaseEstimator],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    costs: Dict[str, float]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calcula los costos, ahorros y métricas financieras de los modelos predictivos
    comparados contra una política de 'Run-to-Failure'.

    Args:
        models_dict (Dict[str, BaseEstimator]): Diccionario {nombre_falla: modelo}.
        X_test (pd.DataFrame): Datos de prueba.
        y_test (pd.DataFrame): Etiquetas reales.
        costs (Dict[str, float]): Diccionario con claves: 'mantenimiento_preventivo', 
                                  'inspeccion_innecesaria', 'falla_no_planificada', 'operacion_normal'.

    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]: 
            1. DataFrame con el desglose detallado por falla.
            2. Diccionario con los totales acumulados (ahorro, costos, ROI).
    """
    # Validar claves de costos
    required_keys = {'mantenimiento_preventivo', 'inspeccion_innecesaria', 'falla_no_planificada', 'operacion_normal'}
    if not required_keys.issubset(costs.keys()):
        raise ValueError(f"Faltan claves en el diccionario de costos. Requeridas: {required_keys}")

    reporte_financiero = []
    accumulators = {
        'ahorro': 0.0,
        'costo_actual': 0.0,
        'costo_ia': 0.0
    }

    logging.info("--- Iniciando cálculo de métricas financieras ---")

    for falla, modelo in models_dict.items():
        try:
            if falla not in y_test.columns:
                logging.warning(f"La columna '{falla}' no está en y_test. Se omite.")
                continue

            # A. Predicción
            y_pred = modelo.predict(X_test)
            
            # B. Matriz de Confusión
            # ravel() devuelve (TN, FP, FN, TP)
            tn, fp, fn, tp = confusion_matrix(y_test[falla], y_pred).ravel()

            # C. Costo IA (Escenario Predictivo)
            costo_ia = (tp * costs['mantenimiento_preventivo']) + \
                       (fp * costs['inspeccion_innecesaria']) + \
                       (fn * costs['falla_no_planificada']) + \
                       (tn * costs['operacion_normal'])

            # D. Costo Base (Run-to-Failure)
            # Asumimos que todas las fallas reales (TP + FN) ocurren y cuestan el máximo
            total_fallas_reales = fn + tp
            costo_rtf = total_fallas_reales * costs['falla_no_planificada']

            # E. Ahorro
            ahorro = costo_rtf - costo_ia

            # Guardar desglose exacto
            reporte_financiero.append({
                'Falla': falla,
                'Costo "Dejar Romper"': costo_rtf,
                'Costo con IA': costo_ia,
                'Ahorro Generado': ahorro,
                'Fallas Detectadas (TP)': tp,
                'Fallas Perdidas (FN)': fn,
                'Falsas Alarmas (FP)': fp
            })

            # Sumar acumulados
            accumulators['ahorro'] += ahorro
            accumulators['costo_actual'] += costo_rtf
            accumulators['costo_ia'] += costo_ia

        except Exception as e:
            logging.error(f"Error procesando la falla '{falla}': {e}")
            continue

    # Crear DataFrame y calcular ROI global
    df_reporte = pd.DataFrame(reporte_financiero)
    
    # Manejo de división por cero para el ROI
    roi_global = (accumulators['ahorro'] / accumulators['costo_ia'] * 100) if accumulators['costo_ia'] > 0 else 0.0
    
    accumulators['roi_global'] = roi_global
    
    return df_reporte, accumulators

def print_executive_report(df_reporte: pd.DataFrame, metrics: Dict[str, float]) -> None:
    """
    Imprime el reporte ejecutivo con el formato EXACTO solicitado por el usuario.
    Nota: En un script .py estándar, 'display()' no existe. Si se usa en Notebook,
    esta función detectará si puede usar display, si no, usará print para el DF.
    """
    print(f"--- ANÁLISIS FINANCIERO POR TIPO DE FALLA ---")
    print("\nDesglose por Falla:")
    
    # Intento de usar display() si estamos en Jupyter, sino print string
    try:
        from IPython.display import display
        # Aplicar formato de moneda visualmente
        styler = df_reporte.style.format({
            'Costo "Dejar Romper"': '${:,.2f}', 
            'Costo con IA': '${:,.2f}', 
            'Ahorro Generado': '${:,.2f}'
        })
        display(styler)
    except ImportError:
        # Fallback para terminal pura
        print(df_reporte.to_string())

    print(f"\n======================================================")
    print(f"RESUMEN EJECUTIVO GLOBAL (Toda la Planta)")
    print(f"======================================================")
    print(f"Costo Política Actual (Run-to-Failure):  ${metrics['costo_actual']:,.2f}")
    print(f"Costo Operativo con Modelos IA:          ${metrics['costo_ia']:,.2f}")
    print(f"------------------------------------------------------")
    print(f"AHORRO TOTAL PROYECTADO:                 ${metrics['ahorro']:,.2f}")
    print(f"RETORNO DE INVERSIÓN (ROI):              {metrics['roi_global']:.1f}%")
    print(f"======================================================")

if __name__ == "__main__":
    # --- PRUEBA LOCAL ---
    print("🔧 Ejecutando prueba local de financial_analysis.py...")
    from sklearn.dummy import DummyClassifier

    # 1. Datos Mock
    X_mock = pd.DataFrame(np.random.rand(20, 3), columns=['f1','f2','f3'])
    y_mock = pd.DataFrame({
        'falla_A': [0]*15 + [1]*5, 
        'falla_B': [0]*18 + [1]*2
    })
    
    # 2. Modelos Mock
    models_mock = {}
    for col in y_mock.columns:
        clf = DummyClassifier(strategy="constant", constant=0) # Predice todo 0 (para forzar FN)
        clf.fit(X_mock, y_mock[col])
        models_mock[col] = clf
        
    # 3. Costos Mock
    COSTOS_TEST = {
        'mantenimiento_preventivo': 500,
        'inspeccion_innecesaria': 150,
        'falla_no_planificada': 10000,
        'operacion_normal': 0
    }

    try:
        # 4. Ejecutar lógica
        df_res, totales = calculate_financial_metrics(models_mock, X_mock, y_mock, COSTOS_TEST)
        
        # 5. Ejecutar visualización
        print_executive_report(df_res, totales)
        
    except Exception as e:
        print(f"❌ Error: {e}")