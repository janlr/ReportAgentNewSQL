import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DB_CONFIG = {
    "AdventureWorks": {
        "driver": "ODBC Driver 17 for SQL Server",
        "server": "localhost",
        "database": "AdventureWorks",
        "trusted_connection": "yes",
        "encrypt_connection": True,
        "connection_timeout": 30,
        "pool_size": 5
    }
}

# Database Security Configuration
DB_SECURITY = {
    "encryption": {
        "enabled": True,
        "algorithm": "AES-256-GCM",
        "key_rotation_days": 30
    },
    "connection_pooling": {
        "enabled": True,
        "min_size": 5,
        "max_size": 20,
        "overflow": 10,
        "timeout": 30
    },
    "credential_storage": {
        "type": "keyring",  # Options: keyring, environment, config_file
        "keyring_service": "report_agent_db"
    },
    "audit": {
        "enabled": True,
        "log_queries": True,
        "log_connections": True,
        "retention_days": 90
    }
}

# Database Type Specific Settings
DB_TYPE_SETTINGS = {
    "MSSQL": {
        "default_schema": "dbo",
        "batch_size": 1000,
        "timeout": 30,
        "isolation_level": "READ COMMITTED"
    },
    "MySQL": {
        "charset": "utf8mb4",
        "collation": "utf8mb4_unicode_ci",
        "batch_size": 1000,
        "timeout": 30
    },
    "PostgreSQL": {
        "search_path": "public",
        "application_name": "report_agent",
        "client_encoding": "UTF8",
        "batch_size": 1000
    },
    "Oracle": {
        "encoding": "UTF-8",
        "nls_lang": "AMERICAN_AMERICA.AL32UTF8",
        "batch_size": 1000,
        "prefetch_rows": 1000
    },
    "SQLite": {
        "isolation_level": "IMMEDIATE",
        "cache_size": -2000,  # 2GB cache
        "journal_mode": "WAL",
        "synchronous": "NORMAL"
    }
}

# Report Templates Configuration
REPORT_TEMPLATES = {
    "Sales Overview": {
        "metrics": [
            "TotalSales",
            "OrderCount",
            "AverageOrderValue",
            "ProductsSold"
        ],
        "dimensions": [
            "ProductCategory",
            "Territory",
            "Date",
            "Customer"
        ],
        "default_visualizations": [
            "SalesTrend",
            "CategoryDistribution",
            "GeographicHeatmap",
            "TopProducts"
        ]
    },
    "Inventory Status": {
        "metrics": [
            "StockLevel",
            "ReorderPoint",
            "StockValue",
            "TurnoverRate"
        ],
        "dimensions": [
            "Product",
            "Location",
            "Category",
            "Time"
        ],
        "default_visualizations": [
            "StockLevelGauge",
            "InventoryTrend",
            "CategoryBreakdown",
            "LocationMap"
        ]
    },
    "Customer Insights": {
        "metrics": [
            "CustomerCount",
            "LTV",
            "ChurnRate",
            "SatisfactionScore"
        ],
        "dimensions": [
            "Segment",
            "Location",
            "PurchaseHistory",
            "Demographics"
        ],
        "default_visualizations": [
            "CustomerSegmentation",
            "LTVDistribution",
            "ChurnAnalysis",
            "SatisfactionTrend"
        ]
    }
}

# Visualization Configuration
VIZ_CONFIG = {
    "color_scheme": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "accent": "#2ca02c",
        "neutral": "#7f7f7f"
    },
    "chart_defaults": {
        "font_family": "Arial, sans-serif",
        "title_font_size": 16,
        "axis_font_size": 12,
        "legend_font_size": 10,
        "background_color": "#ffffff",
        "grid_color": "#e0e0e0"
    },
    "map_defaults": {
        "center": {"lat": 0, "lon": 0},
        "zoom": 2,
        "style": "carto-positron"
    },
    "animation_defaults": {
        "transition_duration": 500,
        "frame_duration": 1000
    }
}

# Export Configuration
EXPORT_CONFIG = {
    "pdf": {
        "page_size": "A4",
        "margin": "1in",
        "font_family": "Arial",
        "header_font_size": 14,
        "body_font_size": 10
    },
    "excel": {
        "sheet_names": {
            "data": "Raw Data",
            "summary": "Summary",
            "charts": "Visualizations"
        },
        "header_format": {
            "bold": True,
            "bg_color": "#f0f0f0"
        }
    }
}

# Cache Configuration
CACHE_CONFIG = {
    "enabled": True,
    "ttl": 3600,  # 1 hour
    "max_size": 1000  # Maximum number of items
}

# Logging Configuration
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "reports.log"
}

# API rate limiting
API_RATE_LIMITS = {
    "max_requests_per_minute": 60,
    "max_tokens_per_request": 4000
}

# Feature flags
FEATURES = {
    "enable_real_time_insights": True,
    "enable_advanced_visualizations": True,
    "enable_export": True,
    "enable_cost_tracking": True
}

# LLM Configuration
LLM_CONFIG = {
    "providers": {
        "anthropic": {
            "models": {
                "claude-3-sonnet-20240229": {
                    "max_tokens": 100000,
                    "cost_per_1k_tokens": 0.015,
                    "max_parallel_requests": 5
                },
                "claude-2": {
                    "max_tokens": 100000,
                    "cost_per_1k_tokens": 0.01,
                    "max_parallel_requests": 5
                }
            },
            "default_model": "claude-3-sonnet-20240229",
            "api_base": "https://api.anthropic.com/v1"
        }
    },
    "default_provider": "anthropic",
    "cache_enabled": True,
    "cache_ttl": 3600,  # 1 hour
    "retry_attempts": 3,
    "timeout": 60,
    "temperature": 0.7,
    "max_tokens_per_request": 4000
}

# Client-specific configurations
CLIENT_CONFIGS = {
    "default": {
        "report_categories": ["general", "custom"],
        "metrics_mapping": {
            "sales": {
                "total_sales": "sum(sales_amount)",
                "order_count": "count(distinct order_id)",
                "average_order": "avg(sales_amount)"
            }
        },
        "table_mapping": {
            "sales": "sales_table",
            "customers": "customer_table",
            "products": "product_table"
        },
        "field_mapping": {
            "customer_id": "customer_key",
            "order_date": "transaction_date",
            "product_id": "product_key"
        }
    },
    "adventureworks": {
        "report_categories": [
            "sales",
            "inventory",
            "customer",
            "financial",
            "custom"
        ],
        "metrics_mapping": {
            "sales": {
                "total_sales": "sum(SalesOrderDetail.LineTotal)",
                "order_count": "count(distinct SalesOrderHeader.SalesOrderID)",
                "average_order": "avg(SalesOrderHeader.TotalDue)"
            }
        },
        "table_mapping": {
            "sales": "Sales.SalesOrderHeader",
            "customers": "Sales.Customer",
            "products": "Production.Product"
        },
        "field_mapping": {
            "customer_id": "CustomerID",
            "order_date": "OrderDate",
            "product_id": "ProductID"
        }
    }
}

# Function to get client config
def get_client_config(client_name: str = "default") -> dict:
    """Get configuration for specific client."""
    return CLIENT_CONFIGS.get(client_name, CLIENT_CONFIGS["default"]) 