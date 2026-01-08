import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_rule_vs_ai(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    falla_target: str,
    columna_regla: str,
    valor_regla: float,
    umbral_ai: float,
    costs: Dict[str, float],
    filename: Optional[str] = None,
    output_dir: str = 'reports/figures'
) -> None:
    """
    Compara el desempeño financiero de una Regla Basada en Condición (ej: Desgaste > 200)
    versus el Modelo de IA optimizado.

    Genera un reporte textual y un gráfico de barras comparativo.

    Args:
        model (BaseEstimator): El modelo de IA entrenado para esta falla.
        X_test (pd.DataFrame): Features de prueba.
        y_test (pd.Series): Target real (binario) para la falla específica.
        falla_target (str): Nombre de la falla (solo para visualización).
        columna_regla (str): Nombre de la columna usada para la regla simple (ej: 'tool_wear_min').
        valor_regla (float): Valor de corte para la regla simple (ej: 200).
        umbral_ai (float): Umbral de probabilidad optimizado para la IA.
        costs (Dict[str, float]): Diccionario de costos ('preventivo', 'inspeccion', 'falla', 'normal').
        filename (Optional[str]): Nombre del archivo para guardar el gráfico.
        output_dir (str): Directorio donde guardar el gráfico.
    """
    # Validación básica de columnas
    if columna_regla not in X_test.columns:
        raise ValueError(f"La columna de regla '{columna_regla}' no existe en X_test.")

    try:
        # ---------------------------------------------------------
        # ESCENARIO A: Mantenimiento Preventivo Tradicional (Regla)
        # ---------------------------------------------------------
        # La predicción es 1 si el valor supera la regla, sino 0
        y_pred_regla = (X_test[columna_regla] > valor_regla).astype(int)

        # Matriz de confusión
        if hasattr(model, "predict_proba"):
            tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_test[falla_target], y_pred_regla).ravel()

        # Cálculo de Costos Regla
        costo_regla = (tp_r * costs['preventivo']) + \
                      (fp_r * costs['inspeccion']) + \
                      (fn_r * costs['falla']) + \
                      (tn_r * costs['normal'])

        # ---------------------------------------------------------
        # ESCENARIO B: Mantenimiento Predictivo (IA Optimizado)
        # ---------------------------------------------------------
        if hasattr(model, "predict_proba"):
            probs_ia = model.predict_proba(X_test)[:, 1]
            y_pred_ia = (probs_ia >= umbral_ai).astype(int)
        else:
            # Fallback por si el modelo no tiene proba (ej: SVM sin probability=True)
            y_pred_ia = model.predict(X_test)

        tn_ai, fp_ai, fn_ai, tp_ai = confusion_matrix(y_test[falla_target], y_pred_ia).ravel()

        costo_ia = (tp_ai * costs['preventivo']) + \
                   (fp_ai * costs['inspeccion']) + \
                   (fn_ai * costs['falla']) + \
                   (tn_ai * costs['normal'])

        # ---------------------------------------------------------
        # RESULTADOS Y VISUALIZACIÓN
        # ---------------------------------------------------------
        ahorro_vs_regla = costo_regla - costo_ia

        # Output de Texto (Formato Exacto solicitado)
        print(f"--- COMPARATIVA DE ESTRATEGIAS PARA {falla_target} ---")
        print(f"1. Estrategia Regla (>{valor_regla} min): ${costo_regla:,.2f}")
        print(f"   - Fallas evitadas: {tp_r}")
        print(f"   - Roturas no previstas: {fn_r}")
        print(f"   - Cambios prematuros: {fp_r}")
        print(f"\n2. Estrategia IA (Umbral {umbral_ai:.2f}): ${costo_ia:,.2f}")
        print(f"   - Fallas evitadas: {tp_ai}")
        print(f"   - Roturas no previstas: {fn_ai}")
        print(f"   - Cambios prematuros: {fp_ai}")
        print(f"\n------------------------------------------------")
        print(f"DIFERENCIA A FAVOR DE LA IA: ${ahorro_vs_regla:,.2f}")

        # Gráfico de Barras Comparativo
        plt.figure(figsize=(8, 6))
        metodos = [f'Preventivo (regla: {columna_regla} > {valor_regla})', 'Predictivo (IA)']
        valores = [costo_regla, costo_ia]
        colores = ['gray', 'green']

        bars = plt.bar(metodos, valores, color=colores)
        plt.title(f'Comparación de Costos: Regla Fija vs IA ({falla_target})', fontsize=14)
        plt.ylabel('Costo Total ($)')
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Etiquetas de valor sobre las barras
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 500, f'${yval:,.0f}', ha='center', fontweight='bold')

        # Guardar gráfico si se solicita
        if filename:
            import os
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            path = os.path.join(output_dir, filename)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logging.info(f"Gráfico comparativo guardado en: {path}")

        plt.show()

    except Exception as e:
        logging.error(f"Error al comparar estrategias: {e}")
        raise

if __name__ == "__main__":
    # --- PRUEBA LOCAL ---
    print("🔧 Ejecutando prueba local de comparison.py...")
    from sklearn.dummy import DummyClassifier

    # 1. Datos Mock
    # Creamos 'tool_wear_min': valores bajos (0-150) sanos, altos (200-300) fallan
    df_len = 100
    X_mock = pd.DataFrame({
        'tool_wear_min': np.concatenate([np.random.randint(0, 180, 80), np.random.randint(210, 300, 20)]),
        'otra_feature': np.random.rand(100)
    })
    # Target: 1 si tool_wear > 220 (regla un poco distinta a la manual para que haya error)
    y_mock = (X_mock['tool_wear_min'] > 220).astype(int)

    # 2. Modelo Mock (Simula IA perfecta)
    # Entrenamos con la misma logica para que de buenos resultados
    model_mock = DummyClassifier(strategy="constant", constant=1) # Dummy placeholder
    # Para simular probas reales, hackeamos un modelo simple
    from sklearn.linear_model import LogisticRegression
    model_mock = LogisticRegression()
    model_mock.fit(X_mock, y_mock)

    # 3. Costos
    COSTOS_TEST = {'preventivo': 500, 'inspeccion': 150, 'falla': 10000, 'normal': 0}

    # 4. Ejecutar
    try:
        compare_rule_vs_ai(
            model=model_mock,
            X_test=X_mock,
            y_test=y_mock,
            falla_target="Falla_Desgaste_Test",
            columna_regla='tool_wear_min',
            valor_regla=200,
            umbral_ai=0.5,
            costs=COSTOS_TEST,
            filename="test_comparison.png"
        )
    except Exception as e:
        print(f"❌ Error en prueba: {e}")