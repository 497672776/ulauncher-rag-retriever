#!/usr/bin/env python3
"""
RAG Service - Systemd服务包装器
监控指定目录，自动加载文档并生成向量数据库和BM25模型
"""

import os
import sys
import time
import signal
import logging
import argparse

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_indexer import DocumentIndexingSystem


class DocumentMonitoringService:
    """文档监控服务包装器"""
    
    def __init__(self, watch_dir: str, model_name: str = "bge-m3:latest", log_level: str = "INFO"):
        """
        初始化文档监控服务
        
        Args:
            watch_dir: 监控的目录路径
            model_name: 嵌入模型名称
            log_level: 日志级别
        """
        # 智能路径处理：支持相对路径、绝对路径和~扩展
        self.watch_dir = self._resolve_path(watch_dir)
        self.model_name = model_name
        self.document_system = None
        self.running = True
        
        # 配置日志
        self._setup_logging(log_level)
        
        # 注册信号处理器
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.logger.info(f"文档监控服务初始化完成")
        self.logger.info(f"监控目录: {self.watch_dir}")
        self.logger.info(f"嵌入模型: {self.model_name}")
    
    def _resolve_path(self, path: str) -> str:
        """
        智能路径解析，支持相对路径、绝对路径和用户目录扩展
        
        Args:
            path: 输入路径
            
        Returns:
            解析后的绝对路径
        """
        # 扩展用户目录（~）
        path = os.path.expanduser(path)
        
        # 如果是相对路径，基于当前工作目录解析
        if not os.path.isabs(path):
            # 获取脚本所在目录作为基准
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, path)
        
        # 返回规范化的绝对路径
        return os.path.abspath(path)
    
    def _setup_logging(self, log_level: str):
        """设置日志配置"""
        # 创建日志目录
        log_dir = "/tmp/rag-service-logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志格式
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/rag-service.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("DocumentMonitoringService")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"接收到信号 {signum}，准备退出...")
        self.running = False
        
        if self.document_system:
            self.document_system.stop_file_monitoring()
    
    def _ensure_directory_exists(self):
        """确保监控目录存在"""
        if not os.path.exists(self.watch_dir):
            self.logger.info(f"创建监控目录: {self.watch_dir}")
            os.makedirs(self.watch_dir, exist_ok=True)
        
        if not os.path.isdir(self.watch_dir):
            raise ValueError(f"监控路径不是目录: {self.watch_dir}")
    
    def _initialize_document_system(self):
        """初始化文档索引和检索系统"""
        try:
            self.logger.info("初始化文档索引和检索系统...")
            self.document_system = DocumentIndexingSystem(
                knowledge_base_path=self.watch_dir,
                model_name=self.model_name
            )
            
            # 初始加载文档
            self.logger.info("执行初始文档加载...")
            doc_count = self.document_system.load_and_index_documents()
            
            if doc_count > 0:
                self.logger.info(f"成功加载 {doc_count} 个文档")
                
                # 显示统计信息
                stats = self.document_system.get_stats()
                self.logger.info("系统统计:")
                for key, value in stats.items():
                    self.logger.info(f"  {key}: {value}")
            else:
                self.logger.warning("没有找到任何文档，将监控新文档的添加")
            
            return True
            
        except Exception as e:
            self.logger.error(f"初始化文档索引和检索系统失败: {e}")
            return False
    
    def _validate_system_dependencies(self):
        """验证系统依赖项"""
        # 1. 检查Python虚拟环境
        if not self._check_python_environment():
            raise RuntimeError("Python虚拟环境检查失败")
        
        # 2. 检查Ollama服务和模型
        if not self._check_ollama_service():
            raise RuntimeError("Ollama服务或模型检查失败")
        
        self.logger.info("✅ 所有依赖项检查通过")
    
    def _check_python_environment(self):
        """检查Python虚拟环境和必要包"""
        try:
            # 检查是否在虚拟环境中
            import sys
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                self.logger.info(f"✅ 运行在虚拟环境: {sys.prefix}")
            else:
                self.logger.warning(f"⚠️ 未检测到虚拟环境，当前Python路径: {sys.executable}")
            
            # 检查必要的包
            required_packages = [
                'llama_index',
                'chromadb', 
                'requests',
                'jieba',
                'rank_bm25'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                    self.logger.debug(f"✅ 包可用: {package}")
                except ImportError:
                    missing_packages.append(package)
                    self.logger.error(f"❌ 缺少必要包: {package}")
            
            if missing_packages:
                self.logger.error(f"❌ 缺少以下必要包: {', '.join(missing_packages)}")
                self.logger.error("请安装缺少的包: pip install " + " ".join(missing_packages))
                return False
            
            self.logger.info("✅ Python环境检查通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Python环境检查失败: {e}")
            return False
    
    def _check_ollama_service(self):
        """检查Ollama服务和模型可用性"""
        try:
            import requests
            
            # 检查Ollama服务连接
            self.logger.info("🔍 检查Ollama服务连接...")
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"❌ Ollama服务不可用，状态码: {response.status_code}")
                return False
            
            self.logger.info("✅ Ollama服务连接正常")
            
            # 检查模型可用性
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            self.logger.info(f"📋 检测到 {len(models)} 个可用模型")
            for model in models:
                self.logger.debug(f"   - {model['name']} (大小: {model.get('size', 'N/A')})")
            
            if self.model_name in model_names:
                self.logger.info(f"✅ 目标嵌入模型可用: {self.model_name}")
                return True
            else:
                self.logger.error(f"❌ 嵌入模型不可用: {self.model_name}")
                self.logger.error(f"可用模型列表: {', '.join(model_names) if model_names else '无'}")
                
                if model_names:
                    self.logger.info("💡 建议:")
                    self.logger.info(f"   1. 检查模型名称是否正确: {self.model_name}")
                    self.logger.info("   2. 或使用以下命令拉取模型:")
                    self.logger.info(f"      ollama pull {self.model_name}")
                else:
                    self.logger.info("💡 建议:")
                    self.logger.info(f"   使用以下命令拉取模型: ollama pull {self.model_name}")
                
                return False
                
        except requests.exceptions.ConnectionError:
            self.logger.error("❌ 无法连接到Ollama服务 (http://localhost:11434)")
            self.logger.error("💡 请确保Ollama服务正在运行:")
            self.logger.error("   sudo systemctl start ollama")
            self.logger.error("   或手动启动: ollama serve")
            return False
            
        except requests.exceptions.Timeout:
            self.logger.error("❌ Ollama服务连接超时")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Ollama服务检查失败: {e}")
            return False
    
    def start_monitoring_service(self):
        """启动文档监控服务"""
        self.logger.info("启动文档监控服务...")
        
        try:
            # 验证系统依赖项
            self._validate_system_dependencies()
            
            # 确保目录存在
            self._ensure_directory_exists()
            
            # 初始化文档系统
            if not self._initialize_document_system():
                self.logger.error("文档索引和检索系统初始化失败，退出服务")
                return 1
            
            # 启动文件变化监控
            self.document_system.start_file_monitoring()
            
            self.logger.info("文档监控服务启动成功，开始监控文件变化...")
            
            # 主循环
            while self.running:
                try:
                    time.sleep(10)  # 每10秒检查一次服务状态
                    
                    # 检查监控是否还在运行
                    if not self.document_system.is_file_monitoring_active():
                        self.logger.warning("文件监控已停止，尝试重新启动...")
                        self.document_system.start_file_monitoring()
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"服务运行时出错: {e}")
                    time.sleep(5)
            
            self.logger.info("文档监控服务正常退出")
            return 0
            
        except Exception as e:
            self.logger.error(f"服务启动失败: {e}")
            return 1
        
        finally:
            if self.document_system:
                self.document_system.stop_file_monitoring()


def load_config(config_file: str = None) -> dict:
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径，如果为None则尝试默认位置
        
    Returns:
        配置字典
    """
    import json
    
    # 默认配置
    default_config = {
        "watch_dir": "./data",  # 改为相对路径
        "model_name": "bge-m3:latest",
        "log_level": "INFO"
    }
    
    # 尝试的配置文件位置
    config_paths = []
    if config_file:
        config_paths.append(config_file)
    
    # 添加默认配置文件位置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_paths.extend([
        os.path.join(script_dir, "document_monitor_config.json"),
        os.path.expanduser("~/.config/rag-service/config.json"),
        "/etc/rag-service/config.json"
    ])
    
    # 尝试加载配置文件
    for config_path in config_paths:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    # 合并配置
                    default_config.update(file_config)
                    print(f"✅ 加载配置文件: {config_path}")
                    break
        except Exception as e:
            print(f"⚠️ 配置文件加载失败 {config_path}: {e}")
    
    return default_config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAG监控服务 - 支持配置文件和命令行参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置文件（./document_monitor_config.json）
  python3 document_monitor_service.py
  
  # 指定配置文件
  python3 document_monitor_service.py --config /path/to/config.json
  
  # 命令行参数覆盖配置文件
  python3 document_monitor_service.py --watch-dir ./my-docs --model-name bge-m3:latest
  
  # 使用相对路径（基于脚本目录）
  python3 document_monitor_service.py --watch-dir ../documents
  
  # 使用用户目录
  python3 document_monitor_service.py --watch-dir ~/Documents/knowledge-base
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        help="配置文件路径 (JSON格式)"
    )
    parser.add_argument(
        "--watch-dir", "-w",
        help="监控的目录路径 (支持相对路径、绝对路径、~/路径)"
    )
    parser.add_argument(
        "--model-name", "-m",
        help="嵌入模型名称"
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置文件
        config = load_config(args.config)
        
        # 命令行参数覆盖配置文件
        if args.watch_dir:
            config["watch_dir"] = args.watch_dir
        if args.model_name:
            config["model_name"] = args.model_name
        if args.log_level:
            config["log_level"] = args.log_level
        
        # 显示最终配置
        print(f"📋 最终配置:")
        print(f"   监控目录: {config['watch_dir']}")
        print(f"   嵌入模型: {config['model_name']}")
        print(f"   日志级别: {config['log_level']}")
        
        # 创建并启动服务
        monitoring_service = DocumentMonitoringService(
            watch_dir=config["watch_dir"],
            model_name=config["model_name"],
            log_level=config["log_level"]
        )
        
        exit_code = monitoring_service.start_monitoring_service()
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()