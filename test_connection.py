import asyncio
from agents.database_agent import DatabaseAgent

async def test_database_connection():
    # Configure the database agent
    config = {
        "type": "mssql",
        "host": "localhost",
        "database": "AdventureWorks2017",
        "driver": "ODBC Driver 17 for SQL Server",
        "echo": True,
        "cache_dir": "./cache"
    }
    
    # Initialize the database agent
    agent = DatabaseAgent(config)
    
    try:
        # Initialize connection
        success = await agent.initialize()
        if success:
            print("Successfully connected to database!")
            
            # Test a simple query
            result = await agent.process({
                "action": "execute_query",
                "query": "SELECT TOP 5 * FROM Sales.Customer"
            })
            
            print("\nSample data from Sales.Customer:")
            print(result)
            
            # Get schema information
            schema_info = await agent.process({
                "action": "get_schema_info"
            })
            
            print("\nDatabase schema information:")
            print(f"Number of tables by schema:")
            schema_table_count = {}
            for full_table_name in schema_info["tables"]:
                schema = schema_info["tables"][full_table_name]["schema"]
                schema_table_count[schema] = schema_table_count.get(schema, 0) + 1
            
            for schema, count in schema_table_count.items():
                print(f"  {schema}: {count} tables")
            
            print(f"\nNumber of views by schema:")
            schema_view_count = {}
            for full_view_name in schema_info["views"]:
                schema = schema_info["views"][full_view_name]["schema"]
                schema_view_count[schema] = schema_view_count.get(schema, 0) + 1
            
            for schema, count in schema_view_count.items():
                print(f"  {schema}: {count} views")
            
            print(f"\nNumber of relationships: {len(schema_info['relationships'])}")
            
        else:
            print("Failed to connect to database")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        
    finally:
        # Cleanup
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(test_database_connection()) 