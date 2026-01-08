import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Configuración de estilo global (para que todos se vean iguales)
sns.set_theme(style="whitegrid")

def guardar_grafico(fig, nombre_archivo, output_dir='reports/figures'):
    """
    Guarda una figura de Matplotlib en la ruta especificada.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    path = os.path.join(output_dir, nombre_archivo)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: {path}")

def plot_feature_importance(modelo, nombres_columnas, top_n=10, guardar_como=None):
    """
    Genera y opcionalmente guarda el gráfico de importancia de variables.
    
    Args:
        modelo: Modelo entrenado (XGBoost/RandomForest).
        nombres_columnas: Lista con nombres de las features.
        top_n: Cuántas variables mostrar.
        guardar_como: Nombre del archivo (ej: 'feature_imp.png'). Si es None, no guarda.
        
    Returns:
        fig: El objeto figura de matplotlib (para mostrar en notebook si quieres).
    """
    importancias = modelo.feature_importances_
    df_imp = pd.DataFrame({'Feature': nombres_columnas, 'Importance': importancias})
    df_imp = df_imp.sort_values('Importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis', ax=ax)
    ax.set_title(f'Top {top_n} Variables Más Importantes', fontsize=14, fontweight='bold')
    
    if guardar_como:
        guardar_grafico(fig, guardar_como)
        
    return fig

def plot_matriz_costos(y_true, y_pred, titulo="Matriz de Costos", guardar_como=None):
    """
    Genera un Heatmap visual de la confusión/costos.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Realidad')
    
    if guardar_como:
        guardar_grafico(fig, guardar_como)
        
    return fig