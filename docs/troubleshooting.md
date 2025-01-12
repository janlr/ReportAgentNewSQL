# Troubleshooting Guide

This guide helps you diagnose and resolve common issues.

## Database Issues

### Connection Problems

**Symptoms:**
- Database connection errors
- Timeout errors
- Authentication failures

**Solutions:**
1. Check Database Credentials
   ```bash
   # Verify .env settings
   DB_HOST=localhost
   DB_PORT=1433
   DB_NAME=your_database
   DB_USER=your_username
   DB_PASSWORD=your_password
   ```

2. Test Database Connection
   ```python
   from agents import DatabaseAgent
   
   agent = DatabaseAgent()
   status = agent.test_connection()
   print(status)  # Should show connection details
   ```

3. Common Fixes:
   - Ensure database server is running
   - Check network connectivity
   - Verify firewall settings
   - Confirm user permissions

## Virtual Environment Issues

### Package Installation Problems

**Symptoms:**
- Import errors
- Missing dependencies
- Version conflicts

**Solutions:**
1. Recreate Virtual Environment
   ```bash
   # Remove existing venv
   rm -rf venv/
   
   # Create new venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. Check Python Version
   ```bash
   python --version  # Should be 3.9 or higher
   ```

3. Update pip and setuptools
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

## LLM Integration Issues

### API Connection Problems

**Symptoms:**
- OpenAI API errors
- Rate limiting
- Authentication failures

**Solutions:**
1. Verify API Key
   ```bash
   # Check .env file
   OPENAI_API_KEY=your_api_key
   ```

2. Test API Connection
   ```python
   from agents import InsightGeneratorAgent
   
   agent = InsightGeneratorAgent()
   status = agent.test_llm_connection()
   print(status)
   ```

3. Common Fixes:
   - Regenerate API key
   - Check API usage limits
   - Verify network connectivity

## Performance Issues

### Slow Report Generation

**Symptoms:**
- Long processing times
- Timeout errors
- High memory usage

**Solutions:**
1. Check Query Performance
   ```python
   from agents import DatabaseAgent
   
   agent = DatabaseAgent()
   stats = agent.get_query_stats()
   print(stats)  # Shows query performance metrics
   ```

2. Monitor Resource Usage
   ```python
   from agents import CostOptimizerAgent
   
   agent = CostOptimizerAgent()
   metrics = agent.get_resource_usage()
   print(metrics)
   ```

3. Optimization Tips:
   - Enable query caching
   - Optimize database indexes
   - Reduce data payload size
   - Use data pagination

## Logging and Debugging

### Enable Debug Logging
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Check Log Files
```bash
# View recent logs
tail -f logs/app.log

# Search for errors
grep "ERROR" logs/app.log
```

## Getting Help

If you can't resolve an issue:

1. Check the [GitHub Issues](https://github.com/yourusername/reporting-system/issues)
2. Review the [API Documentation](./api/index.md)
3. Contact the development team
4. Create a new issue with:
   - Error message
   - Steps to reproduce
   - Environment details
   - Relevant logs 