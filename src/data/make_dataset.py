import pandas as pd
import pyodbc 

def obtener_datos_sql():
    """
    Establece conexión y descarga los datos de mantenimiento.
    Returns: DataFrame con los datos crudos.
    """
    # 1. Configurar (Esto ahora vive seguro dentro de la función)
    server = 'localhost' 
    database = 'MantenimientoIndustrial' 
    
    conn_str = (
        f'DRIVER={{ODBC Driver 17 for SQL Server}};'
        f'SERVER={server};'
        f'DATABASE={database};'
        'Trusted_Connection=yes;'
        'TrustServerCertificate=yes;'
    )

    try:
        # 2. Conectar
        conn = pyodbc.connect(conn_str)
        
        # 3. Extraer
        query = "SELECT * FROM ai4i2020"
        df = pd.read_sql(query, conn)
        
        # 4. Cerrar conexión (¡Buena práctica!)
        conn.close()
        
        print("✅ Conexión exitosa. Datos cargados.")
        return df  # <--- ESTA ES LA CLAVE
        
    except Exception as e:
        print(f"❌ Error en la conexión: {e}")
        return None