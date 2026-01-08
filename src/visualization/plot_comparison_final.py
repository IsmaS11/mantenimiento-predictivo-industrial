import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparacion_final(costo_actual, costo_propuesto, guardar_como=None):
    """
    Genera un gráfico de barras comparativo de alto impacto para el reporte final.
    """
    # Datos
    escenarios = ['Reactivo (Run-to-Failure)', 'Estrategia Híbrida (IA)']
    costos = [costo_actual, costo_propuesto]
    
    # Colores: Rojo (Peligro/Alto Costo) vs Verde (Éxito/Ahorro)
    colores = ['#e74c3c', '#2ecc71'] 
    
    # Configuración del gráfico
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Barras
    bars = ax.bar(escenarios, costos, color=colores, width=0.5, edgecolor='black', linewidth=1)
    
    # Títulos
    ax.set_title('Impacto Financiero Anual del Proyecto', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Costo Operativo ($ USD)', fontsize=12)
    
    # Ajustar límite Y para dar espacio al texto
    ax.set_ylim(0, max(costos) * 1.15)
    
    # 1. Etiquetas de Valor sobre las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max(costos)*0.02),
                f'${height:,.0f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='#333333')
        
    # 2. Flecha de Ahorro (Curva desde la barra roja a la verde)
    ahorro = costo_actual - costo_propuesto
    roi = (ahorro / costo_propuesto) * 100
    
    # Anotación central con caja amarilla
    ax.annotate(
        f'Ahorro Neto: ${ahorro:,.0f}\nROI: {roi:.0f}%',
        xy=(1, costo_propuesto),             # Punta de la flecha (Barra Verde)
        xytext=(0.4, costo_actual * 0.6),    # Texto (En el medio)
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='black', lw=1.5),
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", fc="#fff7bc", ec="#f39c12", alpha=0.9)
    )

    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {guardar_como}")
        
    return fig
