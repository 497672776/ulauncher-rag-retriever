# RAG Systemd Service

基于RAG的文档监控和向量化服务，自动监控目录变化并生成向量数据库。

## 核心文件

- `rag_demo.py` - RAG核心功能模块
- `rag_service.py` - Systemd服务包装器  
- `rag.service` - Systemd服务配置文件
- `rag_config.json` - 服务配置文件
- `start_service.sh` - 服务启动脚本

## 快速部署

### 1. 环境要求

```bash
# Ollama服务运行中，包含bge-m3:latest模型
curl http://localhost:11434/api/tags

# 确保Python虚拟环境已创建并包含必要依赖
# 虚拟环境路径可在rag_config.json中配置
```

### 2. 配置服务

编辑 `rag_config.json` 配置文件：

```json
{
  "watch_dir": "./data",
  "model_name": "bge-m3:latest", 
  "log_level": "INFO",
  "python_venv": "/path/to/your/venv/bin/python"
}
```

**配置说明**：
- `watch_dir`: 监控目录路径（支持相对路径、绝对路径、~/用户目录）
- `model_name`: Ollama嵌入模型名称
- `log_level`: 日志级别（DEBUG/INFO/WARNING/ERROR）
- `python_venv`: Python虚拟环境可执行文件路径

### 3. 部署启动

```bash
# 确保启动脚本可执行
chmod +x start_service.sh

# 部署服务
sudo cp rag.service /etc/systemd/system/
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start rag
sudo systemctl enable rag

# 查看状态
sudo systemctl status rag
sudo journalctl -u rag -f
```

## 功能特性

- ✅ **实时监控**: 5秒间隔检测目录变化
- ✅ **智能去重**: 自动跳过已处理文档
- ✅ **混合检索**: 向量检索 + BM25关键词检索
- ✅ **数据持久化**: ChromaDB向量库 + BM25模型文件
- ✅ **支持格式**: .txt, .md, .pdf, .docx, .doc

## 生成文件

服务运行后会在工作目录生成：

```
demo_chroma_db/
├── chroma.sqlite3      # ChromaDB向量数据库
├── bm25_model.pkl     # BM25检索模型
└── nodes_data.pkl     # 文档节点数据
```

## 使用示例

将文档文件放入监控目录即可自动处理：

```bash
# 添加新文档
echo "测试文档内容" > /path/to/monitor/test.txt

# 查看处理日志  
sudo journalctl -u rag --since "10 seconds ago"
```

## 故障排除

```bash
# 检查服务状态
sudo systemctl status rag

# 查看详细日志
sudo journalctl -u rag -n 50

# 检查依赖服务
curl http://localhost:11434/api/tags

# 手动测试配置
./start_service.sh

# 或直接测试RAG功能
python3 rag_demo.py
```