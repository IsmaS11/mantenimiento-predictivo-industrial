import logging
import copy
import pandas as pd
import xgboost as xgb
from typing import List, Dict, Any, Optional
from sklearn.model_selection import GridSearchCV

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_xgb_optimize_aucpr(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    target_cols: List[str],
    base_param_grid: Optional[Dict[str, List[Any]]] = None,
    cv_folds: int = 3,
    random_state: int = 42
) -> Dict[str, xgb.XGBClassifier]:
    """
    Entrena y optimiza modelos XGBoost para múltiples objetivos utilizando la métrica
    Average Precision (AUCPR), ideal para datasets altamente desbalanceados.

    Calcula dinámicamente el ratio de desbalance para cada columna y lo inyecta
    como opción en la búsqueda de hiperparámetros (scale_pos_weight).

    Args:
        X_train (pd.DataFrame): Features de entrenamiento.
        y_train (pd.DataFrame): Targets de entrenamiento (multilabel/columnas binarias).
        target_cols (List[str]): Lista de nombres de las columnas objetivo.
        base_param_grid (Optional[Dict]): Grid base. NO incluir 'scale_pos_weight', se inyecta auto.
        cv_folds (int): Número de folds para validación cruzada.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        Dict[str, xgb.XGBClassifier]: Diccionario {nombre_falla: mejor_modelo_entrenado}.
    """
    
    # Validaciones
    missing = [col for col in target_cols if col not in y_train.columns]
    if missing:
        raise ValueError(f"Columnas objetivo no encontradas en y_train: {missing}")

    # Grid por defecto si no se pasa uno
    if base_param_grid is None:
        base_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 8],
            'learning_rate': [0.05, 0.1]
        }

    best_models = {}
    
    logging.info("--- Iniciando Optimización XGBoost basada en AUCPR ---")

    for falla in target_cols:
        try:
            logging.info(f"Analizando falla: {falla}")

            # 1. Calcular desbalance específico
            series_falla = y_train[falla]
            n_neg = (series_falla == 0).sum()
            n_pos = (series_falla == 1).sum()

            if n_pos == 0:
                logging.warning(f"La falla '{falla}' no tiene ejemplos positivos. Se omite.")
                continue

            ratio = n_neg / n_pos
            logging.info(f" - Ratio calculado (Neg/Pos): {ratio:.2f}")

            # 2. Preparar Grid específica para esta falla
            # Usamos deepcopy para no modificar la grid base original en cada iteración
            current_grid = copy.deepcopy(base_param_grid)
            
            # Inyectamos los pesos dinámicos: el exacto y uno más agresivo (1.5x)
            current_grid['scale_pos_weight'] = [ratio, ratio * 1.5]

            # 3. Definir Modelo Base
            # Importante: n_jobs=1 en el modelo para que el GridSearch maneje el paralelismo
            # tree_method='hist' acelera mucho el entrenamiento en datos grandes
            xgb_clf = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='aucpr', 
                n_jobs=1,  
                random_state=random_state,
                tree_method='hist'
            )

            # 4. Configurar GridSearchCV
            # scoring='average_precision' es el equivalente de sklearn para AUCPR
            grid = GridSearchCV(
                estimator=xgb_clf,
                param_grid=current_grid,
                scoring='average_precision',
                cv=cv_folds,
                verbose=1,
                n_jobs=-1 
            )

            # 5. Entrenar
            grid.fit(X_train, series_falla)

            # 6. Guardar resultados
            best_models[falla] = grid.best_estimator_
            
            logging.info(f"✅ Mejor AUCPR para {falla}: {grid.best_score_:.4f}")
            logging.info(f"   Mejores params: {grid.best_params_}")

        except Exception as e:
            logging.error(f"❌ Error optimizando falla '{falla}': {e}")
            continue

    logging.info("--- Optimización Finalizada ---")
    return best_models

if __name__ == "__main__":
    # --- BLOQUE DE PRUEBA LOCAL ---
    import numpy as np
    print("🔧 Ejecutando prueba local de optimize_aucpr.py...")

    # Generar datos sintéticos
    # 200 filas, 5 columnas
    X_dummy = pd.DataFrame(np.random.rand(200, 5), columns=[f'feat_{i}' for i in range(5)])
    
    # Targets: 'falla_A' (muy desbalanceada), 'falla_B' (algo desbalanceada)
    y_dummy = pd.DataFrame({
        'falla_A': [0]*190 + [1]*10,  # 5% positivos
        'falla_B': [0]*180 + [1]*20   # 10% positivos
    })
    
    targets = ['falla_A', 'falla_B']

    # Grid mínima para test
    test_grid = {
        'n_estimators': [10],
        'max_depth': [3]
    }

    try:
        modelos = train_xgb_optimize_aucpr(
            X_train=X_dummy, 
            y_train=y_dummy, 
            target_cols=targets,
            base_param_grid=test_grid,
            cv_folds=2
        )
        print(f"Prueba exitosa. Se entrenaron {len(modelos)} modelos.")
        
        # Verificar que podemos predecir con uno de ellos
        if 'falla_A' in modelos:
            prob_pred = modelos['falla_A'].predict_proba(X_dummy.iloc[:5])
            print("Ejemplo predicción (probs):\n", prob_pred)

    except Exception as e:
        print(f"La prueba falló: {e}")