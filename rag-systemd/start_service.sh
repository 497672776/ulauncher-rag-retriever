#!/bin/bash

# RAG服务启动脚本 - 从配置文件读取Python虚拟环境路径

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/rag_config.json"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 从JSON配置文件中提取python_venv路径
PYTHON_VENV=$(python3 -c "
import json
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = json.load(f)
    print(config.get('python_venv', ''))
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
")

# 检查是否成功读取配置
if [ -z "$PYTHON_VENV" ]; then
    echo "错误: 无法从配置文件中读取python_venv路径"
    exit 1
fi

# 如果是相对路径，转换为基于脚本目录的绝对路径
if [[ "$PYTHON_VENV" != /* ]]; then
    PYTHON_VENV="$SCRIPT_DIR/$PYTHON_VENV"
fi

# 检查Python可执行文件是否存在
if [ ! -f "$PYTHON_VENV" ]; then
    echo "错误: Python虚拟环境不存在: $PYTHON_VENV"
    exit 1
fi

# 启动RAG服务
exec "$PYTHON_VENV" "$SCRIPT_DIR/rag_service.py" --config "$CONFIG_FILE"