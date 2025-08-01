import os
from pathlib import Path
from typing import Dict, Any
import json

class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            # 数据相关配置
            "data": {
                "papers_dir": "data/papers",
                "processed_dir": "data/processed", 
                "index_dir": "data/index",
                "index_file": "data/index/finance_papers.index",
                "metadata_file": "data/index/finance_papers_metadata.json"
            },
            
            # 模型相关配置
            "models": {
                "llm_path": "models/mistral-7b-instruct",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "auto",  # auto, cuda, cpu
                "load_in_4bit": True,
                "torch_dtype": "float16"
            },
            
            # 数据处理配置
            "processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_results_per_category": 100,
                "download_delay": 1.0,
                "batch_size": 32,
                "arxiv_page_size": 100,
                "arxiv_delay_seconds": 3.0,
                "arxiv_num_retries": 3
            },
            
            # RAG系统配置
            "rag": {
                "top_k": 5,
                "max_length": 512,
                "temperature": 0.7,
                "max_new_tokens": 1024,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            },
            
            # 服务器配置
            "server": {
                "host": "127.0.0.1",
                "port": 8000,
                "reload": True,
                "cors_origins": ["*"]
            },
            
            # 日志配置
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "file": "logs/rag_system.log"
            },
            
            # arXiv分类配置
            "arxiv_categories": [
                "q-fin.CP",  # Computational Finance
                "q-fin.EC",  # Economics  
                "q-fin.GN",  # General Finance
                "q-fin.MF",  # Mathematical Finance
                "q-fin.PM",  # Portfolio Management
            ]
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 深度合并配置
                self._merge_configs(default_config, user_config)
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}")
        
        return default_config
    
    def _merge_configs(self, default: Dict, user: Dict):
        """深度合并配置字典"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def get(self, key: str, default=None):
        """获取配置值，支持点号分隔的键"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self):
        """保存配置到文件"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def create_directories(self):
        """创建必要的目录"""
        dirs = [
            self.get("data.papers_dir"),
            self.get("data.processed_dir"),
            self.get("data.index_dir"),
            Path(self.get("logging.file")).parent
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

# 全局配置实例
config = Config() 