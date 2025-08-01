import logging
from pathlib import Path
from config import config
import sys

class Logger:
    
    def __init__(self, name: str = "RAGSystem"):
        self.name = name
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        
        # 避免重复添加handler
        if logger.handlers:
            return logger
        
        logger.setLevel(getattr(logging, config.get("logging.level", "INFO")))
        
        # 创建格式化器
        formatter = logging.Formatter(
            config.get("logging.format", "%(asctime)s - %(levelname)s - %(message)s")
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = config.get("logging.file", "logs/rag_system.log")
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """记录一般信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告信息"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误信息"""
        self.logger.error(message)
    
    def exception(self, message: str):
        """记录异常信息（包含堆栈跟踪）"""
        self.logger.exception(message)

# 全局日志实例
logger = Logger() 