#!/usr/bin/env python3
"""
配置管理工具
用于查看、修改和验证系统配置
"""

import json
import argparse
from pathlib import Path
from config import config
from logger import logger

def show_config():
    """显示当前配置"""
    print("=== 当前系统配置 ===")
    print(json.dumps(config.config, indent=2, ensure_ascii=False))

def show_section(section):
    """显示特定配置节"""
    if section in config.config:
        print(f"=== {section} 配置 ===")
        print(json.dumps(config.config[section], indent=2, ensure_ascii=False))
    else:
        print(f"配置节 '{section}' 不存在")

def set_config(key, value):
    """设置配置值"""
    try:
        # 尝试解析JSON值
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)
        
        config.set(key, value)
        config.save()
        print(f"配置已更新: {key} = {value}")
        logger.info(f"Configuration updated: {key} = {value}")
    except Exception as e:
        print(f"设置配置失败: {e}")
        logger.error(f"Failed to set configuration: {e}")

def validate_config():
    """验证配置"""
    print("=== 配置验证 ===")
    
    # 检查必要的目录
    required_dirs = [
        config.get("data.papers_dir"),
        config.get("data.processed_dir"),
        config.get("data.index_dir"),
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ 目录存在: {dir_path}")
        else:
            print(f"✗ 目录不存在: {dir_path}")
    
    # 检查模型路径
    model_path = Path(config.get("models.llm_path"))
    if model_path.exists():
        print(f"✓ 模型路径存在: {model_path}")
    else:
        print(f"✗ 模型路径不存在: {model_path}")
    
    # 检查配置文件
    if Path("config.json").exists():
        print("✓ 配置文件存在")
    else:
        print("✗ 配置文件不存在")

def create_default_config():
    """创建默认配置文件"""
    config.save()
    print("默认配置文件已创建: config.json")
    logger.info("Default configuration file created")

def main():
    parser = argparse.ArgumentParser(description="配置管理工具")
    parser.add_argument("action", choices=["show", "section", "set", "validate", "create"], 
                       help="操作类型")
    parser.add_argument("--key", help="配置键")
    parser.add_argument("--value", help="配置值")
    parser.add_argument("--section", help="配置节名称")
    
    args = parser.parse_args()
    
    if args.action == "show":
        show_config()
    elif args.action == "section":
        if args.section:
            show_section(args.section)
        else:
            print("请指定 --section 参数")
    elif args.action == "set":
        if args.key and args.value is not None:
            set_config(args.key, args.value)
        else:
            print("请指定 --key 和 --value 参数")
    elif args.action == "validate":
        validate_config()
    elif args.action == "create":
        create_default_config()

if __name__ == "__main__":
    main() 