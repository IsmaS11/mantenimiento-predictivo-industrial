import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_optimal_thresholds(
    models_dict: Dict[str, BaseEstimator],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    costs: Dict[str, float],
    step: float = 0.05
) -> Dict[str, Any]:
    """
    Realiza un barrido de umbrales para encontrar el punto de operación que minimiza
    el costo financiero para cada falla.

    Args:
        models_dict (Dict[str, BaseEstimator]): Modelos entrenados.
        X_test (pd.DataFrame): Features de prueba.
        y_test (pd.DataFrame): Targets reales.
        costs (Dict[str, float]): Diccionario de costos ('preventivo', 'inspeccion', 'falla_critica', 'normal').
        step (float): Paso del barrido de umbrales (default 0.05).

    Returns:
        Dict[str, Any]: Diccionario con los resultados detallados por falla:
            {
                'nombre_falla': {
                    'thresholds': np.array,
                    'costs': list,
                    'optimal_threshold': float,
                    'min_cost': float
                }, ...
            }
    """
    umbrales_optimos = {}
    results = {}
    range_thresholds = np.arange(step, 1.0, step)
    
    # Validar que los modelos tengan predict_proba
    for falla, modelo in models_dict.items():
        if not hasattr(modelo, "predict_proba"):
            logging.warning(f"El modelo para {falla} no tiene 'predict_proba'. Se omite.")
            continue
            
        try:
            logging.info(f"Calculando umbrales óptimos para: {falla}")
            
            # 1. Obtener probabilidades (Clase 1)
            y_prob = modelo.predict_proba(X_test)[:, 1]
            y_true = y_test[falla]
            
            costs_list = []
            
            # 2. Barrido de umbrales
            for umbral in range_thresholds:
                # Predicción dura dinámica
                preds_ajustadas = (y_prob >= umbral).astype(int)
                
                tn, fp, fn, tp = confusion_matrix(y_true, preds_ajustadas).ravel()
                
                # Función de Costo
                costo_total = (tp * costs['preventivo']) + \
                              (fp * costs['inspeccion']) + \
                              (fn * costs['falla_critica']) + \
                              (tn * costs['normal'])
                
                costs_list.append(costo_total)
            
            # 3. Encontrar el óptimo
            min_cost = min(costs_list)
            idx_min = costs_list.index(min_cost)
            best_threshold = range_thresholds[idx_min]
            umbrales_optimos[falla] = best_threshold
            results[falla] = {
                'thresholds': range_thresholds,
                'costs': costs_list,
                'optimal_threshold': best_threshold,
                'min_cost': min_cost
            }
            
        except Exception as e:
            logging.error(f"Error optimizando umbrales para {falla}: {e}")
            continue
            
    return results, umbrales_optimos

def plot_threshold_optimization(
    optimization_results: Dict[str, Any],
    filename: Optional[str] = None,
    output_dir: str = 'reports/figures'
) -> Dict[str, float]:
    """
    Genera el gráfico de curvas de costo vs umbral y devuelve el diccionario simple de umbrales óptimos.
    Replica exactamente la visualización solicitada.

    Args:
        optimization_results (Dict): Salida de calculate_optimal_thresholds.
        filename (Optional[str]): Si se define, guarda el gráfico en output_dir.
        output_dir (str): Directorio de salida.

    Returns:
        Dict[str, float]: Diccionario simple {falla: umbral_optimo} para uso en producción.
    """
    fallas = list(optimization_results.keys())
    n_fallas = len(fallas)
    
    if n_fallas == 0:
        logging.warning("No hay resultados para graficar.")
        return {}

    # Configuración de grilla dinámica (similar al script original 2x2 si son 4)
    cols = 2
    rows = (n_fallas + 1) // 2
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 5 * rows))
    axes = axes.flatten()
    
    final_thresholds = {}
    
    print("\n--- RESULTADOS DE OPTIMIZACIÓN DE UMBRALES ---")

    for i, falla in enumerate(fallas):
        data = optimization_results[falla]
        
        # Extraer datos
        rango_umbrales = data['thresholds']
        costos = data['costs']
        mejor_umbral = data['optimal_threshold']
        min_costo = data['min_cost']
        
        # Guardar para retorno
        final_thresholds[falla] = mejor_umbral
        
        # Graficar
        ax = axes[i]
        ax.plot(rango_umbrales, costos, marker='.', color='darkblue', linewidth=2)
        ax.axvline(mejor_umbral, color='red', linestyle='--', label=f'Óptimo: {mejor_umbral:.2f}')
        
        # Estética exacta solicitada
        ax.set_title(f'Optimización para {falla}', fontweight='bold')
        ax.set_xlabel('Umbral de Decisión')
        ax.set_ylabel('Costo Total ($)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Print solicitado
        print(f"Falla {falla}: Umbral Óptimo = {mejor_umbral:.2f} | Costo Mínimo = ${min_costo:,.2f}")

    # Limpiar ejes vacíos si hay número impar de gráficos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    
    if filename:
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Gráfico guardado en: {path}")
        
    plt.show() # Mostrar en notebook
    
    return final_thresholds

if __name__ == "__main__":
    # --- PRUEBA LOCAL ---
    print("🔧 Ejecutando prueba local de threshold_optimization.py...")
    
    # 1. Mock de datos
    from sklearn.dummy import DummyClassifier
    X_mock = pd.DataFrame(np.random.rand(50, 2))
    y_mock = pd.DataFrame({'falla_A': [0]*40 + [1]*10, 'falla_B': [0]*45 + [1]*5})
    
    # 2. Mock modelos (Dummy devuelve probas simples)
    models = {}
    for c in y_mock.columns:
        clf = DummyClassifier(strategy="prior")
        clf.fit(X_mock, y_mock[c])
        models[c] = clf
        
    # 3. Costos
    COSTOS_TEST = {'preventivo': 500, 'inspeccion': 150, 'falla_critica': 10000, 'normal': 0}
    
    # 4. Ejecutar flujo
    res_calc = calculate_optimal_thresholds(models, X_mock, y_mock, COSTOS_TEST)
    umbrales = plot_threshold_optimization(res_calc, filename="test_thresholds.png")
    
    print("Diccionario final:", umbrales)