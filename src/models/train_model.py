import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_multioutput_rf_gridsearch(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    param_grid: Optional[Dict[str, Any]] = None,
    cv_folds: int = 4,
    random_state: int = 42
)-> GridSearchCV:
    """
    Entrena un modelo MultiOutputClassifier utilizando RandomForest como estimador base.
    Realiza una búsqueda de hiperparámetros (GridSearchCV) para optimizar la métrica F1-Macro.

    Args:
        X_train (pd.DataFrame): Features de entrenamiento.
        y_train (pd.DataFrame): Etiquetas (targets) de entrenamiento (multietiqueta).
        param_grid (Optional[Dict[str, Any]]): Diccionario de hiperparámetros para GridSearchCV.
            Si es None, se utiliza una configuración por defecto.
            Nota: Las claves deben llevar el prefijo 'estimator__' (ej: 'estimator__n_estimators').
        cv_folds (int): Número de pliegues (folds) para la validación cruzada. Default: 4.
        random_state (int): Semilla aleatoria para reproducibilidad. Default: 42.

    Returns:
        GridSearchCV: Objeto GridSearchCV entrenado, que contiene el mejor modelo (best_estimator_)
                      y los resultados de la búsqueda.

    Raises:
        ValueError: Si los datos de entrada están vacíos o tienen dimensiones incorrectas.
        Exception: Error genérico durante el proceso de fitting.
    """
    # Validaciones básicas
    if X_train.empty or y_train.empty:
        logging.error("Los DataFrames de entrenamiento no pueden estar vacíos.")
        raise ValueError("X_train y y_train deben contener datos.")

    # 1. Definir grilla por defecto si no se provee una externa
    if param_grid is None:
        logging.info("No se proporcionó param_grid, usando configuración por defecto.")
        param_grid = {
            'estimator__n_estimators': [300, 500],
            'estimator__max_depth': [10, 15, 20],
            'estimator__min_samples_split': [2, 5],
            'estimator__min_samples_leaf': [1, 5],
            'estimator__max_features': ['sqrt', 'log2']
        }

    try:
        logging.info("Configurando el modelo MultiOutput RandomForest...")

        # 2. Definir el modelo base y el wrapper
        # class_weight='balanced' es crítico para problemas industriales (pocas fallas vs operación normal)
        forest = RandomForestClassifier(
            random_state=random_state,
            class_weight='balanced'
        )
        multi_model = MultiOutputClassifier(forest)

        # 3. Definir métricas de evaluación
        scoring = {
            'precision': make_scorer(precision_score, average='macro', zero_division=0),
            'recall': make_scorer(recall_score, average='macro', zero_division=0),
            'f1': make_scorer(f1_score, average='macro', zero_division=0)
        }

        # 4. Configurar GridSearchCV
        grid_search = GridSearchCV(
            estimator=multi_model,
            param_grid=param_grid,
            scoring=scoring,
            refit='f1',    # Optimizar basado en F1
            cv=cv_folds,
            n_jobs=-1,     # Usar todos los núcleos disponibles
            verbose=1      # Verbose bajo para producción, subir para debug
        )

        # 5. Ejecutar la búsqueda
        logging.info(f"Iniciando GridSearchCV con {cv_folds} folds y optimizando F1-Macro...")
        grid_search.fit(X_train, y_train)

        logging.info("Entrenamiento finalizado con éxito.")
        logging.info(f"Mejores parámetros encontrados: {grid_search.best_params_}")
        logging.info(f"Mejor Score F1 (CV): {grid_search.best_score_:.4f}")

        return grid_search

    except Exception as e:
        logging.error(f"Error crítico durante el entrenamiento del modelo: {e}")
        raise

if __name__ == "__main__":
    # --- BLOQUE DE PRUEBA LOCAL ---
    print("🔧 Ejecutando prueba local de train_model.py...")

    # 1. Generar datos sintéticos para simular mantenimiento predictivo
    # 100 muestras, 5 features
    X_dummy = pd.DataFrame(np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)])
    
    # Target Multi-salida: Falla Tipo A, Falla Tipo B (0 o 1)
    y_dummy = pd.DataFrame(np.random.randint(0, 2, size=(100, 2)), columns=['Falla_A', 'Falla_B'])

    # 2. Definir una grid muy pequeña para que la prueba sea rápida
    test_grid = {
        'estimator__n_estimators': [10, 20], # Pocos árboles para test rápido
        'estimator__max_depth': [5]
    }

    try:
        # 3. Ejecutar función
        model_result = train_multioutput_rf_gridsearch(
            X_train=X_dummy,
            y_train=y_dummy,
            param_grid=test_grid,
            cv_folds=2 # Pocos folds para test rápido
        )
        
        print("✅ Prueba finalizada. El objeto retornado es:", type(model_result))
        print("Mejor estimador disponible en: model_result.best_estimator_")

    except Exception as e:
        print(f"❌ La prueba falló: {e}")


import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_predict_xgb_dynamic_weights(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    target_cols: List[str],
    X_test: Optional[pd.DataFrame] = None,
    base_param_grid: Optional[Dict[str, List[Any]]] = None,
    cv_folds: int = 3,
    random_state: int = 42
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Entrena múltiples modelos XGBoost (uno por cada columna objetivo) ajustando dinámicamente
    el peso de las clases (scale_pos_weight) según el desbalance específico de cada falla.
    
    Si se proporciona X_test, genera un DataFrame con las predicciones.

    Args:
        X_train (pd.DataFrame): Features de entrenamiento.
        y_train (pd.DataFrame): Targets de entrenamiento.
        target_cols (List[str]): Lista de columnas objetivo a entrenar.
        X_test (Optional[pd.DataFrame]): Features de prueba. Si es None, no devuelve predicciones.
        base_param_grid (Optional[Dict]): Grid base. NO incluir 'scale_pos_weight' aquí, se calcula dinámicamente.
                                          NO usar prefijos 'estimator__' si no es un pipeline.
        cv_folds (int): Número de folds para CV.
        random_state (int): Semilla aleatoria.

    Returns:
        Tuple[Dict, DataFrame]: 
            - Diccionario con los mejores modelos entrenados {nombre_columna: best_estimator}.
            - DataFrame con las predicciones en X_test (o None si no se proveyó X_test).
    """
    if not all(col in y_train.columns for col in target_cols):
        raise ValueError(f"Algunas columnas objetivo no están en y_train: {target_cols}")

    # Configuración de métricas
    scoring = {
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0)
    }

    # Grid por defecto (sin prefijos estimator__ porque usamos el clasificador directo)
    if base_param_grid is None:
        base_param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6],
            'subsample': [0.8, 1.0]
        }
    
    modelos_entrenados = {}
    resultados_pred = pd.DataFrame(index=X_test.index) if X_test is not None else None

    logging.info("--- Iniciando Entrenamiento Iterativo XGBoost ---")

    for falla in target_cols:
        try:
            logging.info(f"Procesando objetivo: {falla}")
            
            # 1. Cálculo de balance dinámico
            n_negativos = (y_train[falla] == 0).sum()
            n_positivos = (y_train[falla] == 1).sum()

            if n_positivos == 0:
                logging.warning(f"La columna {falla} no tiene casos positivos. Saltando entrenamiento.")
                continue

            ratio_balance = n_negativos / n_positivos
            logging.info(f" - Ratio (Neg/Pos) para {falla}: {ratio_balance:.2f}")

            # 2. Copiar grid base y agregar pesos dinámicos
            # IMPORTANTE: scale_pos_weight ayuda al modelo a prestar atención a la clase minoritaria
            current_param_grid = base_param_grid.copy()
            current_param_grid['scale_pos_weight'] = [
                ratio_balance * 0.5, 
                ratio_balance, 
                ratio_balance * 1.5
            ]

            # 3. Configurar GridSearchCV
            xgb = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss', # Evita warnings recientes de XGBoost
                random_state=random_state,
                n_jobs=-1 # Paralelismo interno de XGBoost (se solapa con CV n_jobs)
            )

            # Usamos n_jobs=1 en GridSearch si XGBoost usa n_jobs=-1 para no sobrecargar CPU,
            # o viceversa. Aquí priorizamos GridSearch.
            xgb_cv = GridSearchCV(
                estimator=xgb,
                param_grid=current_param_grid,
                scoring=scoring,
                refit='f1',
                cv=cv_folds,
                n_jobs=2, # Ajustar según núcleos disponibles
                verbose=1
            )

            # 4. Entrenar
            xgb_cv.fit(X_train, y_train[falla])
            
            # 5. Guardar mejor modelo
            best_model = xgb_cv.best_estimator_
            modelos_entrenados[falla] = best_model
            logging.info(f" - Mejor F1 para {falla}: {xgb_cv.best_score_:.4f}")

            # 6. Predecir (si aplica)
            if X_test is not None:
                resultados_pred[falla] = best_model.predict(X_test)

        except Exception as e:
            logging.error(f"Error entrenando columna {falla}: {e}")
            # No lanzamos raise para permitir que las otras columnas se entrenen
            continue

    logging.info("--- Entrenamiento Finalizado ---")
    
    return modelos_entrenados, resultados_pred

if __name__ == "__main__":
    # --- PRUEBA LOCAL ---
    print("🔧 Test local de train_xgb_dynamic.py")
    
    # Datos dummy
    df_len = 100
    X_dummy = pd.DataFrame(np.random.rand(df_len, 4), columns=['A', 'B', 'C', 'D'])
    # Creamos target desbalanceado
    y_dummy = pd.DataFrame({
        'falla_motor': [0]*(df_len-10) + [1]*10,      # 10% fallas
        'falla_sensor': [0]*(df_len-5) + [1]*5        # 5% fallas
    })
    
    target_list = ['falla_motor', 'falla_sensor']
    
    # Grid pequeña para test rápido
    test_grid = {
        'n_estimators': [10],
        'max_depth': [3]
    }

    try:
        modelos, preds = train_predict_xgb_dynamic_weights(
            X_train=X_dummy,
            y_train=y_dummy,
            target_cols=target_list,
            X_test=X_dummy, # Usamos el mismo para probar predicción
            base_param_grid=test_grid,
            cv_folds=2
        )
        
        print("✅ Modelos entrenados:", list(modelos.keys()))
        print("✅ Shape de predicciones:", preds.shape)
        print("✅ Head de predicciones:\n", preds.head(3))
        
    except Exception as e:
        print(f"❌ Error en test: {e}")