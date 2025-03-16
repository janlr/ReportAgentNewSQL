import pyodbc
import os
from dotenv import load_dotenv

def test_connection():
    # Load environment variables
    load_dotenv()
    
    # Print SQL Server information
    print("Available SQL Server Drivers:")
    for driver in pyodbc.drivers():
        print(f"  - {driver}")
    
    print("\nTrying to connect...")
    
    try:
        # Build connection string directly with pyodbc
        conn_str = (
            "Driver={ODBC Driver 17 for SQL Server};"
            f"Server={os.getenv('DB_HOST')};"
            f"Database={os.getenv('DB_NAME')};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes"
        )
        
        print(f"\nConnection String (without sensitive data):")
        print(conn_str)
        
        # Try to connect
        conn = pyodbc.connect(conn_str, timeout=30)
        cursor = conn.cursor()
        
        # Test the connection
        cursor.execute("SELECT @@version")
        row = cursor.fetchone()
        print("\nSuccess! Connected to SQL Server")
        print(f"SQL Server Version: {row[0]}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print("\nConnection Failed!")
        print(f"Error: {str(e)}")
        print("\nPlease verify:")
        print("1. SQL Server is running")
        print("2. Instance name is correct")
        print("3. Windows Authentication is enabled")
        print("4. SQL Server Browser service is running (for named instances)")

if __name__ == "__main__":
    test_connection() 