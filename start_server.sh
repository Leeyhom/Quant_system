#!/bin/bash
# ============================================================
# 量化策略实时服务启动脚本
# ============================================================

set -e

echo "=" 60
echo "🚀 量化策略实时服务启动中..."
echo "=" 60

# 激活conda环境（如使用）
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate quant 2>/dev/null || true
fi

# 检查是否安装服务依赖
if ! python -c "import fastapi, uvicorn, apscheduler" 2>/dev/null; then
    echo "📦 安装服务依赖..."
    pip install -r requirements-server.txt
fi

# 创建必要目录
mkdir -p logs data results templates static

# 设置环境变量
export PYTHONPATH=$(pwd)

# 启动服务
echo "🌐 访问地址: http://localhost:${PORT:-8080}"
echo "📊 API文档: http://localhost:${PORT:-8080}/docs"
echo "📱 飞书通知: $(if [ "$FEISHU_NOTIFY" = "1" ]; then echo已开启; else echo已关闭; fi)"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

exec python quant_server.py "$@"
