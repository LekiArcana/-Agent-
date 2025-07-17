#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速问答测试，验证修复后的系统
"""

import asyncio
import warnings
warnings.filterwarnings("ignore")

from langchain_qa_system import LangchainLegalQASystem

async def quick_test():
    """快速测试修复后的系统"""
    print("🔧 测试修复后的系统")
    print("=" * 50)
    
    # 初始化系统
    system = LangchainLegalQASystem()
    ready = await system.initialize()
    
    if not ready:
        print("❌ 系统初始化失败")
        return
    
    print("✅ 系统初始化成功")
    
    # 测试基础问答
    test_questions = [
        "什么是合同？",
        "合同的定义是什么？",
        "故意伤害罪的构成要件是什么？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 测试 {i}: {question}")
        print("-" * 40)
        
        try:
            result = await system.ask_question(question, show_details=True)
            
            if result.get("success"):
                print("✅ 问答成功")
                answer = result["answer"]
                print(f"📄 回答长度: {len(answer)} 字符")
                print(f"📝 回答预览: {answer[:300]}...")
                
                # 检查检索信息
                retrieval_info = result.get("retrieval_info", {})
                docs_found = retrieval_info.get("documents_found", 0)
                print(f"🔍 检索到文档数: {docs_found}")
                
                # 对于合同问题，检查是否检索到第464条
                if "合同" in question:
                    documents = retrieval_info.get("documents", [])
                    found_464 = False
                    for doc in documents:
                        content = doc.get("content", "")
                        if "第四百六十四条" in content:
                            found_464 = True
                            print("🎯 成功检索到第464条（合同定义）")
                            break
                    
                    if not found_464:
                        print("⚠️  未检索到第464条")
            else:
                print(f"❌ 问答失败: {result.get('message', '未知错误')}")
        
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 50)
    print("🎉 快速测试完成")

if __name__ == "__main__":
    asyncio.run(quick_test()) 