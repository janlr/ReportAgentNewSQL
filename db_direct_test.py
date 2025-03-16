import os
import sys
import logging
import traceback
from dotenv import load_dotenv
import pyodbc
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_test")

def test_direct_pyodbc():
    """Test connection using pyodbc directly."""
    logger.info("=== Testing with direct pyodbc connection ===")
    
    try:
        # List available drivers
        logger.info("Available ODBC drivers:")
        for driver in pyodbc.drivers():
            logger.info(f"  - {driver}")
        
        # Build connection string from env variables
        host = os.getenv('DB_HOST')
        database = os.getenv('DB_NAME')
        driver = os.getenv('DB_DRIVER')
        
        logger.info("Connection parameters:")
        logger.info(f"  Host: {host}")
        logger.info(f"  Database: {database}")
        logger.info(f"  Driver: {driver}")
        
        conn_str = (
            f"Driver={{{driver}}};"
            f"Server={host};"
            f"Database={database};"
            f"Trusted_Connection=yes;"
            f"TrustServerCertificate=yes"
        )
        
        logger.info(f"Connection string: {conn_str}")
        
        # Try to connect
        logger.info("Attempting to connect...")
        conn = pyodbc.connect(conn_str, timeout=30)
        cursor = conn.cursor()
        
        # Test the connection
        cursor.execute("SELECT @@version as version")
        row = cursor.fetchone()
        logger.info("Connection successful!")
        logger.info(f"SQL Server Version: {row[0]}")
        
        # List databases
        logger.info("Listing databases:")
        cursor.execute("SELECT name FROM sys.databases ORDER BY name")
        for row in cursor.fetchall():
            logger.info(f"  - {row[0]}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error("Direct pyodbc connection failed!")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_sqlalchemy():
    """Test connection using SQLAlchemy."""
    logger.info("\n=== Testing with SQLAlchemy connection ===")
    
    try:
        # Get connection parameters
        db_type = os.getenv('DB_TYPE', 'mssql')
        driver = os.getenv('DB_DRIVER')
        host = os.getenv('DB_HOST')
        database = os.getenv('DB_NAME')
        trusted_connection = os.getenv('DB_TRUSTED_CONNECTION', 'yes')
        
        logger.info("Connection parameters:")
        logger.info(f"  DB Type: {db_type}")
        logger.info(f"  Driver: {driver}")
        logger.info(f"  Host: {host}")
        logger.info(f"  Database: {database}")
        logger.info(f"  Trusted Connection: {trusted_connection}")
        
        # Build SQLAlchemy connection string
        # Method 1: Using URL parameters
        connection_string_1 = (
            f"mssql+pyodbc:///?odbc_connect=" + quote_plus(
                f"Driver={{{driver}}};"
                f"Server={host};"
                f"Database={database};"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate=yes"
            )
        )
        
        # Try connection with Method 1
        logger.info("Trying connection Method 1 (URL parameters)...")
        logger.info(f"Connection string: {connection_string_1}")
        
        try:
            engine_1 = create_engine(connection_string_1, echo=False, connect_args={"timeout": 30})
            with engine_1.connect() as conn:
                result = conn.execute(text("SELECT @@version as version"))
                version = result.fetchone()[0]
                logger.info("Method 1 Connection successful!")
                logger.info(f"SQL Server Version: {version}")
                
            engine_1.dispose()
            return True
            
        except Exception as e1:
            logger.error("Method 1 connection failed!")
            logger.error(f"Error type: {type(e1).__name__}")
            logger.error(f"Error message: {str(e1)}")
            
            # Try Method 2 as fallback
            try:
                # Method 2: Using DSN-less connection
                connection_string_2 = (
                    f"mssql+pyodbc://{host}/{database}?"
                    f"driver={driver.replace(' ', '+')}&"
                    f"trusted_connection=yes&"
                    f"TrustServerCertificate=yes"
                )
                
                logger.info("\nTrying connection Method 2 (DSN-less)...")
                logger.info(f"Connection string: {connection_string_2}")
                
                engine_2 = create_engine(connection_string_2, echo=False, connect_args={"timeout": 30})
                with engine_2.connect() as conn:
                    result = conn.execute(text("SELECT @@version as version"))
                    version = result.fetchone()[0]
                    logger.info("Method 2 Connection successful!")
                    logger.info(f"SQL Server Version: {version}")
                    
                engine_2.dispose()
                return True
                
            except Exception as e2:
                logger.error("Method 2 connection failed!")
                logger.error(f"Error type: {type(e2).__name__}")
                logger.error(f"Error message: {str(e2)}")
                
                # Try Method 3 as final fallback
                try:
                    # Method 3: Using raw ODBC connection string
                    raw_conn_str = (
                        f"Driver={{{driver}}};"
                        f"Server={host};"
                        f"Database={database};"
                        f"Trusted_Connection=yes;"
                        f"TrustServerCertificate=yes"
                    )
                    connection_string_3 = f"mssql+pyodbc:///?odbc_connect={raw_conn_str}"
                    
                    logger.info("\nTrying connection Method 3 (raw ODBC)...")
                    logger.info(f"Connection string: {connection_string_3}")
                    
                    engine_3 = create_engine(connection_string_3, echo=False, connect_args={"timeout": 30})
                    with engine_3.connect() as conn:
                        result = conn.execute(text("SELECT @@version as version"))
                        version = result.fetchone()[0]
                        logger.info("Method 3 Connection successful!")
                        logger.info(f"SQL Server Version: {version}")
                        
                    engine_3.dispose()
                    return True
                    
                except Exception as e3:
                    logger.error("Method 3 connection failed!")
                    logger.error(f"Error type: {type(e3).__name__}")
                    logger.error(f"Error message: {str(e3)}")
                    raise
            
    except Exception as e:
        logger.error("SQLAlchemy connection failed!")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def check_sql_server_availability():
    """Check if SQL Server is running and reachable."""
    logger.info("\n=== Checking SQL Server availability ===")
    
    import socket
    from subprocess import run, PIPE
    
    try:
        # Get host and port
        host = os.getenv('DB_HOST')
        if '\\' in host:  # Named instance
            logger.info(f"Named instance detected: {host}")
            # Check SQL Browser service for named instances
            base_host = host.split('\\')[0]
            sql_browser_port = 1434  # UDP port for SQL Browser
            
            logger.info(f"Checking SQL Browser on {base_host}:{sql_browser_port}")
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(1)
                s.sendto(b'\x02', (base_host, sql_browser_port))
                response = s.recvfrom(1024)
                logger.info(f"SQL Browser responded: {response}")
            except Exception as e:
                logger.warning(f"SQL Browser test failed: {str(e)}")
            
            # For named instances, try using sqlcmd to test connection
            logger.info("Testing connection with sqlcmd...")
            sqlcmd_result = run(
                ["sqlcmd", "-S", host, "-d", os.getenv('DB_NAME'), "-E", "-Q", "SELECT @@VERSION"],
                stdout=PIPE, stderr=PIPE, text=True
            )
            if sqlcmd_result.returncode == 0:
                logger.info("sqlcmd connection successful!")
                logger.info(f"Output: {sqlcmd_result.stdout}")
            else:
                logger.error("sqlcmd connection failed!")
                logger.error(f"Error: {sqlcmd_result.stderr}")
            
        else:  # Standard port
            port = int(os.getenv('DB_PORT', '1433'))
            logger.info(f"Testing TCP connectivity to {host}:{port}")
            
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            result = s.connect_ex((host, port))
            if result == 0:
                logger.info(f"Port {port} is OPEN on {host}")
            else:
                logger.error(f"Port {port} is CLOSED on {host}")
                logger.error("SQL Server may not be running or is blocked by firewall")
            s.close()
            
    except Exception as e:
        logger.error(f"Error checking SQL Server availability: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """Run all database connection tests."""
    logger.info("Starting database connection tests...")
    load_dotenv()  # Load environment variables from .env file
    
    # Check SQL Server availability
    check_sql_server_availability()
    
    # Test direct pyodbc connection
    pyodbc_result = test_direct_pyodbc()
    
    # Test SQLAlchemy connection
    sqlalchemy_result = test_sqlalchemy()
    
    # Final results
    logger.info("\n=== FINAL RESULTS ===")
    logger.info(f"Direct pyodbc connection: {'SUCCESS' if pyodbc_result else 'FAILED'}")
    logger.info(f"SQLAlchemy connection: {'SUCCESS' if sqlalchemy_result else 'FAILED'}")
    
    if pyodbc_result and not sqlalchemy_result:
        logger.info("DIAGNOSIS: The direct connection works but SQLAlchemy connection failed.")
        logger.info("This suggests an issue with the SQLAlchemy connection string format.")
    elif not pyodbc_result:
        logger.info("DIAGNOSIS: The direct connection failed.")
        logger.info("This suggests an issue with SQL Server configuration or connectivity.")
    
    if pyodbc_result or sqlalchemy_result:
        logger.info("RECOMMENDATION: Use the working connection method in your application.")
    else:
        logger.info("RECOMMENDATION: Check SQL Server configuration, firewall settings, and instance name.")

if __name__ == "__main__":
    main() 