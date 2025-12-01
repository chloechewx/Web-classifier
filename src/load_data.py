import sqlite3
import pandas as pd
import config
import logging

logging.basicConfig(level=logging.INFO)


def load_data(database: str = None, table_name: str = None) -> pd.DataFrame:

    db = config.database
    table = config.table_name
    
    query = f"SELECT * FROM {table};"

    try:
        with sqlite3.connect(db) as conn:
            df = pd.read_sql_query(query, conn)

        logging.info(f"Loaded {table} from DB, shape: {df.shape}")
        return df

    except Exception:
        logging.error("Failed to load data", exc_info=True)
        raise

# df = load_data()  