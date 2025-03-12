from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from agents import DatabaseAgent

def main():
    # Load environment variables
    load_dotenv()
    
    # Build connection string for Windows Authentication
    connection_string = (
        "mssql+pyodbc://LAPTOP-R5KN1453\\SQLEXPRESS/AdventureWorksDW2022?"
        "driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    )
    
    print(f"Attempting to connect with connection string: {connection_string}")
    
    try:
        # Create engine
        engine = create_engine(connection_string, echo=os.getenv('DB_ECHO', 'True').lower() == 'true')
        
        # Test connection by executing a simple query
        with engine.connect() as conn:
            result = conn.execute(text("SELECT TOP 5 * FROM DimCustomer"))
            for row in result:
                print(row)
                
        print("Successfully connected to database!")
        
    except Exception as e:
        print(f"Error initializing database connection: {str(e)}")
        print("Failed to connect to database")

if __name__ == "__main__":
    main() 