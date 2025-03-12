import os
import subprocess
import sys
import json
import logging
from pathlib import Path

def setup_logging():
    """Configure logging for development environment setup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/dev_setup.log')
        ]
    )

def create_virtualenv():
    """Create and activate virtual environment"""
    if not os.path.exists('venv'):
        logging.info("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
    
    # Get the path to the virtual environment's Python executable
    if sys.platform == 'win32':
        venv_python = os.path.join('venv', 'Scripts', 'python.exe')
    else:
        venv_python = os.path.join('venv', 'bin', 'python')
    
    return venv_python

def install_dependencies(venv_python):
    """Install project dependencies"""
    logging.info("Installing dependencies...")
    subprocess.run([venv_python, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([venv_python, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)

def setup_git_hooks():
    """Set up Git pre-commit hooks"""
    hooks_dir = Path('.git/hooks')
    if not hooks_dir.exists():
        logging.warning("Git repository not initialized. Skipping hooks setup.")
        return

    pre_commit_hook = hooks_dir / 'pre-commit'
    hook_content = """#!/bin/sh
# Run code formatting
black .
# Run linting
flake8 .
# Run type checking
mypy agents/
"""
    pre_commit_hook.write_text(hook_content)
    pre_commit_hook.chmod(0o755)
    logging.info("Git pre-commit hooks configured")

def setup_dev_config():
    """Create development configuration"""
    dev_config = {
        "database": {
            "development": {
                "driver": "sqlite",
                "database": "dev.db"
            }
        },
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key": os.getenv("ANTHROPIC_API_KEY")
        },
        "cache": {
            "type": "local",
            "directory": "cache",
            "ttl": 3600
        },
        "logging": {
            "level": "DEBUG",
            "file": "logs/development.log"
        },
        "development": {
            "debug": True,
            "reload": True,
            "port": 8501,
            "host": "localhost"
        }
    }
    
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / 'development.json', 'w') as f:
        json.dump(dev_config, f, indent=2)
    logging.info("Development configuration created")

def setup_vscode():
    """Configure VSCode settings"""
    vscode_dir = Path('.vscode')
    vscode_dir.mkdir(exist_ok=True)
    
    settings = {
        "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
        "python.linting.enabled": True,
        "python.linting.flake8Enabled": True,
        "python.linting.mypyEnabled": True,
        "python.formatting.provider": "black",
        "editor.formatOnSave": True,
        "editor.rulers": [88],
        "files.exclude": {
            "**/__pycache__": True,
            "**/.pytest_cache": True,
            "**/*.pyc": True
        }
    }
    
    with open(vscode_dir / 'settings.json', 'w') as f:
        json.dump(settings, f, indent=2)
    
    launch = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Streamlit",
                "type": "python",
                "request": "launch",
                "module": "streamlit",
                "args": ["run", "app.py"],
                "justMyCode": True
            },
            {
                "name": "Python: Current File",
                "type": "python",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "justMyCode": True
            }
        ]
    }
    
    with open(vscode_dir / 'launch.json', 'w') as f:
        json.dump(launch, f, indent=2)
    
    logging.info("VSCode configuration created")

def setup_llm_config():
    """Configure LLM settings."""
    config = {
        "provider": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "temperature": 0.7,
        "max_tokens": 1000,
        "api_key": os.getenv("ANTHROPIC_API_KEY")
    }
    
    # Validate API key
    if not config["api_key"]:
        print("Warning: ANTHROPIC_API_KEY not found in environment variables")
        config["api_key"] = "dummy_key_for_testing"
    
    return config

def main():
    """Main setup function"""
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    setup_logging()
    logging.info("Starting development environment setup")
    
    try:
        venv_python = create_virtualenv()
        install_dependencies(venv_python)
        setup_git_hooks()
        setup_dev_config()
        setup_vscode()
        
        logging.info("Development environment setup completed successfully")
        logging.info("\nNext steps:")
        logging.info("1. Activate virtual environment:")
        if sys.platform == 'win32':
            logging.info("   .\\venv\\Scripts\\activate")
        else:
            logging.info("   source venv/bin/activate")
        logging.info("2. Create a .env file from .env.example")
        logging.info("3. Start the application: streamlit run app.py")
        
    except Exception as e:
        logging.error(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 