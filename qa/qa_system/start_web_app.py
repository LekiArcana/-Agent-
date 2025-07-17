#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Langchain 多 Agent 法律问答系统 Web 应用启动脚本
"""

import sys
import os
import argparse
import asyncio
import warnings
warnings.filterwarnings("ignore")

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'jinja2',
        'langchain',
        'sentence_transformers',
        'faiss_cpu',
        'ollama'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('_cpu', ''))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🏛️  Langchain 多 Agent 法律问答系统 Web 应用              ║
║                                                              ║
║    基于先进的 AI 技术为您提供专业法律咨询服务                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_system_info():
    """打印系统信息"""
    print("📊 系统信息:")
    print(f"   • Python 版本: {sys.version.split()[0]}")
    print(f"   • 工作目录: {os.getcwd()}")
    print(f"   • FastAPI 模式: 开发模式")
    print(f"   • 支持功能: 多 Agent 协作、会话管理、内容分析")
    print("")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Langchain 多 Agent 法律问答系统 Web 应用"
    )
    parser.add_argument(
        '--host', 
        default='127.0.0.1',
        help='服务器主机地址 (默认: 127.0.0.1)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000,
        help='服务器端口 (默认: 8000)'
    )
    parser.add_argument(
        '--reload', 
        action='store_true',
        help='启用自动重载 (开发模式)'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=1,
        help='工作进程数 (默认: 1)'
    )
    parser.add_argument(
        '--no-banner', 
        action='store_true',
        help='不显示启动横幅'
    )
    
    args = parser.parse_args()
    
    if not args.no_banner:
        print_banner()
        print_system_info()
    
    # 检查依赖
    print("🔍 检查系统依赖...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ 所有依赖已满足")
    print("")
    
    # 检查数据文件
    data_dir = os.path.join("..", "Data")
    required_files = [
        "law_index.faiss",
        "law_metadata.pkl"
    ]
    
    print("📁 检查数据文件...")
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  缺少以下数据文件:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n请确保已运行向量化脚本生成数据文件")
    else:
        print("✅ 所有数据文件完整")
    print("")
    
    # 启动服务器
    print("🚀 启动 Web 服务器...")
    print(f"   • 服务地址: http://{args.host}:{args.port}")
    print(f"   • API 文档: http://{args.host}:{args.port}/docs")
    print(f"   • 重载模式: {'开启' if args.reload else '关闭'}")
    print("")
    print("💡 提示:")
    print("   • 浏览器访问上述地址使用 Web 界面")
    print("   • 按 Ctrl+C 停止服务器")
    print("   • 首次启动可能需要下载模型，请耐心等待")
    print("")
    print("━" * 60)
    
    try:
        import uvicorn
        uvicorn.run(
            "web_app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 