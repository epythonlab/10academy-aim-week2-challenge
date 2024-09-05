from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
import pandas as pd

# Load environment variables from .env file
load_dotenv()

def conn():
    # Get database credentials from the .env file
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')

    # Create SQLAlchemy engine
    engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

    # Test connection by trying to connect to the engine
    connection = engine.connect()
    print("Connection to database successful!")
    connection.close()  # Close the connection after testing

    return engine  # Return the engine

def load_data():
    engine = conn()  # Use the engine from the conn function
    query = "SELECT * FROM xdr_data;"
    xdr_df = pd.read_sql(query, engine)  # Pass the engine, not the connection
    return xdr_df
