# Configuration Guide

This guide covers all configuration options and customization settings.

## Environment Configuration

### Basic Settings
Copy `.env.example` to `.env` and configure:

```ini
# Database Configuration
DB_HOST=localhost
DB_PORT=1433
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password

# API Keys
OPENAI_API_KEY=your_openai_api_key

# Report Configuration
REPORT_OUTPUT_DIR=reports
TEMPLATE_DIR=templates
VISUALIZATION_DIR=reports/visualizations

# Web Dashboard
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/reporting_system.log

# Security
SECRET_KEY=your_secret_key
ENCRYPTION_KEY=your_encryption_key

# Cache Configuration
CACHE_TYPE=redis
CACHE_REDIS_HOST=localhost
CACHE_REDIS_PORT=6379
CACHE_REDIS_DB=0

# Performance
MAX_WORKERS=4
BATCH_SIZE=1000
CACHE_TIMEOUT=3600

# Feature Flags
ENABLE_EMAIL_REPORTS=true
ENABLE_PDF_REPORTS=true
ENABLE_INTERACTIVE_DASHBOARDS=true
ENABLE_CACHE=true
ENABLE_COMPRESSION=true
```

## Agent Configuration

### Database Agent
```python
config = {
    'connection_pool_size': 10,
    'connection_timeout': 30,
    'query_timeout': 60,
    'max_retries': 3,
    'retry_delay': 1
}
```

### LLM Agent
```python
config = {
    'model': 'gpt-4',
    'temperature': 0.7,
    'max_tokens': 1000,
    'cache_enabled': True,
    'cache_ttl': 3600
}
```

### Visualization Agent
```python
config = {
    'default_theme': 'light',
    'chart_style': 'modern',
    'interactive': True,
    'output_format': 'html'
}
```

## Security Configuration

### Authentication
```python
config = {
    'auth_enabled': True,
    'auth_provider': 'oauth2',
    'oauth2_settings': {
        'client_id': 'your_client_id',
        'client_secret': 'your_client_secret',
        'authorize_url': 'https://auth.example.com/authorize',
        'token_url': 'https://auth.example.com/token'
    }
}
```

### Data Encryption
```python
config = {
    'encryption_enabled': True,
    'encryption_algorithm': 'AES-256-GCM',
    'key_rotation_days': 90
}
```

## Performance Tuning

### Caching
```python
config = {
    'cache_provider': 'redis',
    'cache_settings': {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'ttl': 3600
    }
}
```

### Query Optimization
```python
config = {
    'query_cache_size': '1GB',
    'max_parallel_queries': 4,
    'batch_size': 1000,
    'timeout': 30
}
```

## Logging Configuration

### Basic Logging
```python
config = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'logs/app.log',
    'max_log_size': '100MB',
    'backup_count': 5
}
```

### Advanced Logging
```python
config = {
    'handlers': ['file', 'console', 'sentry'],
    'sentry_dsn': 'your_sentry_dsn',
    'log_rotation': 'daily',
    'include_metrics': True
}
```

## Report Templates

### Template Configuration
```python
config = {
    'template_dir': 'templates',
    'default_template': 'standard',
    'custom_templates': {
        'executive': 'templates/executive.html',
        'detailed': 'templates/detailed.html'
    }
}
```

### Output Formats
```python
config = {
    'formats': ['pdf', 'html', 'excel'],
    'pdf_settings': {
        'page_size': 'A4',
        'orientation': 'portrait'
    },
    'excel_settings': {
        'sheet_name': 'Report',
        'include_charts': True
    }
}
```

For more detailed configuration options, see the [API Reference](./api/index.md). 