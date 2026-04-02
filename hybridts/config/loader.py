# !pip install pyyaml

import yaml
from pathlib import Path

def load_config(file_name="settings.yaml"):
    """
    Lê o arquivo YAML de configuração e retorna um dicionário Python.
    """
    # Encontra o caminho absoluto da pasta config
    config_path = Path(__file__).parent / file_name
    
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    return config