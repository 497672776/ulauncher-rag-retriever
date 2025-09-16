#!/bin/bash

# 文档检索插件安装脚本
# 用法: ./setup.sh <用户名>

if [ $# -eq 0 ]; then
    echo "用法: $0 <用户名>"
    echo "示例: $0 liudecheng"
    exit 1
fi

USERNAME=$1
EXT_DIR="/home/$USERNAME/.local/share/ulauncher/extensions/com.github.497672776.ulauncher-rag-retriever"

echo "🚀 开始安装文档检索插件..."
echo "👤 用户名: $USERNAME"
echo "📁 安装目录: $EXT_DIR"

# 检查目录是否存在
if [ ! -d "$EXT_DIR" ]; then
    echo "❌ 错误: 目录不存在 $EXT_DIR"
    echo "请先确保插件代码已安装到正确位置"
    exit 1
fi

cd "$EXT_DIR"

echo "📦 1. 安装虚拟环境和依赖..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "⚙️ 2. 配置后台服务..."
# 修改document-indexer.service文件中的用户名
sed -i "s|/home/[^/]*/|/home/$USERNAME/|g" rag-systemd/document-indexer.service
sed -i "s|User=.*|User=$USERNAME|g" rag-systemd/document-indexer.service
sed -i "s|Group=.*|Group=$USERNAME|g" rag-systemd/document-indexer.service

echo "🔧 3. 启动后台服务..."
cd rag-systemd
sudo cp document-indexer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart document-indexer
sudo systemctl enable document-indexer

echo "✅ 安装完成！"
echo ""
echo "📋 下一步:"
echo "1. 放入测试文件到: $EXT_DIR/rag-systemd/data"
echo "2. 查看日志: sudo journalctl -u document-indexer -f"
echo "3. 等待向量库生成完成后即可使用Ulauncher"