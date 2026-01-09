import pandas as pd
import pyodbc 

import pandas as pd
from sqlalchemy import create_engine, text
import urllib
import warnings
from sqlalchemy import exc


def obtener_datos_sql():
    # 1. Configurar tu string de conexión como siempre
    server = 'localhost' 
    database = 'MantenimientoIndustrial' 
    
    # Usamos urllib para convertir la string en formato URL seguro
    params = urllib.parse.quote_plus(
        f'DRIVER={{ODBC Driver 17 for SQL Server}};'
        f'SERVER={server};'
        f'DATABASE={database};'
        'Trusted_Connection=yes;'
        'TrustServerCertificate=yes;'
    )

    # 2. CREAR EL MOTOR (ENGINE) - Esto es lo que pide Pandas
    # El formato es: mssql+pyodbc:///?odbc_connect=TUS_PARAMETROS
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    try:
        # 3. Usar el motor para leer
        # Usamos 'with' para que la conexión se cierre sola automáticamente
        with engine.connect() as conn:
            query = "SELECT * FROM ai4i2020"
            # Pandas ahora es feliz porque recibe una conexión de SQLAlchemy
            df = pd.read_sql(query, conn)
            
        print("✅ Datos cargados correctamente con SQLAlchemy.")
        return df

    except Exception as e:
        print(f"❌ Error: {e}")
        return None
# Ignorar advertencias específicas de SQLAlchemy sobre versiones
warnings.filterwarnings('ignore', category=exc.SAWarning)