"""Agent-based reporting system powered by AutoGen."""

__version__ = "0.1.0"

# Import base agent first
from .base_agent import BaseAgent
from .database_agent import DatabaseAgent
from .data_manager_agent import DataManagerAgent
from .user_interface_agent import UserInterfaceAgent
from .report_generator_agent import ReportGeneratorAgent
from .insight_generator_agent import InsightGeneratorAgent
from .master_orchestrator_agent import MasterOrchestratorAgent
from .assistant_agent import AssistantAgent

import os
from dotenv import load_dotenv
import streamlit as st

def initialize_environment():
    """Initialize environment variables and Streamlit secrets."""
    # Load environment variables from .env
    load_dotenv()

    # Instead of modifying st.secrets directly, we'll use st.session_state
    # to store our configuration
    if 'config' not in st.session_state:
        st.session_state.config = {
            'database': {
                'type': 'mssql',
                'host': os.getenv('DB_HOST'),
                'port': int(os.getenv('DB_PORT', 1433)),
                'database': os.getenv('DB_NAME'),
                'driver': os.getenv('DB_DRIVER'),
                'echo': os.getenv('DB_ECHO', 'True').lower() == 'true',
                'trusted_connection': os.getenv('DB_TRUSTED_CONNECTION', 'yes'),
                'cache_dir': './cache',
                'schema_mapping': {
                    'sales': {
                        'orders_table': os.getenv('ORDERS_TABLE', 'Sales.SalesOrderHeader'),
                        'order_details_table': os.getenv('ORDER_DETAILS_TABLE', 'Sales.SalesOrderDetail'),
                        'products_table': os.getenv('PRODUCTS_TABLE', 'Production.Product'),
                        'categories_table': os.getenv('CATEGORIES_TABLE', 'Production.ProductCategory'),
                        'customers_table': os.getenv('CUSTOMERS_TABLE', 'Sales.Customer'),
                        'territories_table': os.getenv('TERRITORIES_TABLE', 'Sales.SalesTerritory'),
                        'column_mapping': {
                            'order_date': 'OrderDate',
                            'order_id': 'SalesOrderID',
                            'product_name': 'Name',
                            'product_id': 'ProductID',
                            'quantity': 'OrderQty',
                            'unit_price': 'UnitPrice',
                            'total_amount': 'LineTotal',
                            'customer_id': 'CustomerID',
                            'territory_id': 'TerritoryID',
                            'territory_name': 'Name',
                            'region_code': 'CountryRegionCode'
                        }
                    }
                },
                'type_mapping': {
                    'unsupported_types': [
                        # Basic SQL types
                        'geometry', 'geography', 'hierarchyid',
                        'xml', 'image', 'sql_variant', 'binary',
                        'varbinary', 'timestamp', 'rowversion',
                        'cursor', 'table', 'uniqueidentifier',
                        'money', 'smallmoney', 'datetimeoffset',
                        'filestream', 'sysname',
                        
                        # Extended types
                        'hierarchyid', 'dbversion', 'sqlvariant',
                        'assembly', 'service_broker', 'fulltext',
                        'extended_properties', 'sparse_columns',
                        'filestream_data', 'computed_columns',
                        'identity_columns', 'masked_columns'
                    ],
                    'conversion_type': 'VARCHAR(MAX)',
                    'type_handlers': {
                        # Basic types
                        'geometry': 'VARCHAR(MAX)',
                        'geography': 'VARCHAR(MAX)',
                        'hierarchyid': 'VARCHAR(MAX)',
                        'xml': 'NVARCHAR(MAX)',
                        'image': 'VARCHAR(MAX)',
                        'sql_variant': 'VARCHAR(MAX)',
                        'binary': 'VARCHAR(MAX)',
                        'varbinary': 'VARCHAR(MAX)',
                        'timestamp': 'VARCHAR(50)',
                        'rowversion': 'VARCHAR(50)',
                        
                        # Additional types
                        'uniqueidentifier': 'CHAR(36)',
                        'money': 'DECIMAL(19,4)',
                        'smallmoney': 'DECIMAL(10,4)',
                        'datetimeoffset': 'VARCHAR(50)',
                        'cursor': 'VARCHAR(MAX)',
                        'table': 'VARCHAR(MAX)',
                        'filestream': 'VARCHAR(MAX)',
                        'sysname': 'NVARCHAR(128)',
                        
                        # CLR types
                        'hierarchyid': 'VARCHAR(MAX)',
                        'sql_variant': 'VARCHAR(MAX)',
                        
                        # Large object types
                        'text': 'VARCHAR(MAX)',
                        'ntext': 'NVARCHAR(MAX)',
                        'image': 'VARBINARY(MAX)',
                        
                        # Date/Time types
                        'date': 'VARCHAR(10)',
                        'time': 'VARCHAR(8)',
                        'datetime': 'VARCHAR(23)',
                        'datetime2': 'VARCHAR(27)',
                        'smalldatetime': 'VARCHAR(19)',
                        'datetimeoffset': 'VARCHAR(34)',
                        
                        # Numeric types with precision
                        'decimal': 'VARCHAR(50)',
                        'numeric': 'VARCHAR(50)',
                        'float': 'VARCHAR(50)',
                        'real': 'VARCHAR(50)',
                        
                        # Extended types
                        'sparse_vector': 'VARCHAR(MAX)',
                        'assembly': 'VARCHAR(MAX)',
                        'service_broker': 'VARCHAR(MAX)',
                        'fulltext': 'NVARCHAR(MAX)',
                        'extended_property': 'NVARCHAR(MAX)',
                        'computed_column': 'VARCHAR(MAX)',
                        'identity_column': 'VARCHAR(50)',
                        'masked_column': 'VARCHAR(MAX)',
                        
                        # JSON and structured types
                        'json': 'NVARCHAR(MAX)',
                        'structured': 'NVARCHAR(MAX)',
                        'spatial_data': 'VARCHAR(MAX)',
                        'graph_data': 'VARCHAR(MAX)'
                    },
                    'type_conversion_rules': {
                        # Basic rules
                        'preserve_precision': True,
                        'handle_nulls': True,
                        'handle_special_chars': True,
                        'max_string_length': 8000,
                        'date_format': 'ISO8601',
                        
                        # Advanced rules
                        'trim_strings': True,
                        'remove_control_chars': True,
                        'normalize_unicode': True,
                        'handle_infinity': True,
                        'handle_nan': True,
                        
                        # Numeric conversion rules
                        'decimal_places': 4,
                        'scientific_notation_threshold': 15,
                        'round_numbers': True,
                        'preserve_leading_zeros': True,
                        
                        # Date/Time conversion rules
                        'timezone_handling': 'convert_to_utc',
                        'date_format_string': 'yyyy-MM-dd HH:mm:ss.fff',
                        'handle_invalid_dates': True,
                        
                        # Binary data rules
                        'binary_format': 'base64',
                        'max_binary_length': 1000,
                        'binary_truncation_indicator': '...',
                        
                        # Special value handling
                        'null_string': 'NULL',
                        'empty_string': '',
                        'infinity_string': 'Infinity',
                        'nan_string': 'NaN',
                        'default_error': 'CONVERSION_ERROR',
                        
                        # Character encoding
                        'input_encoding': 'UTF-8',
                        'output_encoding': 'UTF-8',
                        'handle_encoding_errors': 'replace'
                    }
                }
            },
            'api_keys': {
                'openai': os.getenv('OPENAI_API_KEY'),
                'anthropic': os.getenv('ANTHROPIC_API_KEY')
            },
            'settings': {
                'environment': os.getenv('ENVIRONMENT', 'development'),
                'report_output_dir': os.getenv('REPORT_OUTPUT_DIR', 'reports'),
                'template_dir': os.getenv('TEMPLATE_DIR', 'templates'),
                'visualization_dir': os.getenv('VISUALIZATION_DIR', 'reports/visualizations'),
                'enable_email_reports': os.getenv('ENABLE_EMAIL_REPORTS', 'false').lower() == 'true',
                'enable_pdf_reports': os.getenv('ENABLE_PDF_REPORTS', 'true').lower() == 'true',
                'enable_interactive_dashboards': os.getenv('ENABLE_INTERACTIVE_DASHBOARDS', 'true').lower() == 'true'
            }
        }

def get_config(key: str = None):
    """Get configuration from session state."""
    if key is None:
        return st.session_state.config
    return st.session_state.config.get(key)

# Define exports
__all__ = [
    'BaseAgent',
    'DatabaseAgent',
    'DataManagerAgent',
    'UserInterfaceAgent',
    'ReportGeneratorAgent',
    'InsightGeneratorAgent',
    'AssistantAgent',
    'MasterOrchestratorAgent',
    'initialize_environment',
    'get_config'
]
