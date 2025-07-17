#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试问答系统的上下文功能
"""

import asyncio
import time

async def test_context_functionality():
    """测试上下文功能"""
    try:
        from langchain_qa_system import LangchainLegalQASystem
        
        print("🧠 测试问答系统的上下文功能")
        print("=" * 60)
        
        # 初始化系统
        print("1. 初始化系统...")
        qa_system = LangchainLegalQASystem()
        success = await qa_system.initialize()
        
        if not success:
            print("❌ 系统初始化失败")
            return
        
        print("✅ 系统初始化成功")
        
        # 测试场景1：基础上下文记忆
        print(f"\n{'='*50}")
        print("📝 测试场景1：基础上下文记忆")
        print(f"{'='*50}")
        
        conversation_1 = [
            "什么是合同？",
            "它有哪些基本条款？",  # 追问：它指代合同
            "如果违反了会怎么样？",  # 继续追问
            "有什么法律后果？"  # 深度追问
        ]
        
        print("开始多轮对话测试...")
        for i, question in enumerate(conversation_1, 1):
            print(f"\n--- 第 {i} 轮对话 ---")
            print(f"🙋 用户: {question}")
            
            start_time = time.time()
            result = await qa_system.ask_question(
                question=question,
                answer_style="simple",  # 使用简单风格便于阅读
                show_details=False
            )
            end_time = time.time()
            
            if result.get("success"):
                answer = result["answer"]
                print(f"🤖 助手: {answer[:200]}...")  # 只显示前200字符
                print(f"⏱️ 耗时: {end_time - start_time:.2f}秒")
                
                # 检查上下文相关性
                if i > 1:  # 从第二轮开始检查上下文
                    context_keywords = ["合同", "条款", "违约", "责任"]
                    has_context = any(keyword in answer for keyword in context_keywords)
                    if has_context:
                        print("✅ 检测到上下文相关回答")
                    else:
                        print("⚠️ 回答可能缺乏上下文关联")
            else:
                print(f"❌ 回答失败: {result.get('message')}")
            
            # 短暂等待，模拟真实对话节奏
            await asyncio.sleep(1)
        
        # 测试场景2：会话切换与上下文隔离
        print(f"\n{'='*50}")
        print("🔄 测试场景2：会话切换与上下文隔离")
        print(f"{'='*50}")
        
        # 创建新会话
        session_1 = qa_system.create_new_session("合同法咨询")
        print(f"📝 创建新会话: {session_1}")
        
        # 在新会话中问一个不同的问题
        print(f"\n🙋 用户（新会话）: 什么是劳动法？")
        result = await qa_system.ask_question(
            question="什么是劳动法？",
            answer_style="professional",
            show_details=False
        )
        
        if result.get("success"):
            answer = result["answer"]
            print(f"🤖 助手: {answer[:200]}...")
            
            # 检查是否泄露了之前合同相关的上下文
            contract_leakage = any(word in answer for word in ["合同", "违约", "条款"])
            if not contract_leakage:
                print("✅ 会话隔离正常，无上下文泄露")
            else:
                print("⚠️ 可能存在会话间上下文泄露")
        
        # 测试场景3：对话总结功能
        print(f"\n{'='*50}")
        print("📋 测试场景3：对话总结功能")
        print(f"{'='*50}")
        
        print("生成当前会话总结...")
        summary_result = await qa_system.get_conversation_summary()
        
        if summary_result.get("success"):
            summary = summary_result.get("summary", "")
            print(f"📄 对话总结:\n{summary}")
            
            # 检查总结质量
            summary_keywords = ["劳动法", "劳动者", "权利"]
            coverage = sum(1 for keyword in summary_keywords if keyword in summary)
            print(f"📊 关键词覆盖率: {coverage}/{len(summary_keywords)} ({coverage/len(summary_keywords)*100:.1f}%)")
            
            if coverage >= len(summary_keywords) * 0.5:  # 50%以上覆盖率
                print("✅ 总结质量良好")
            else:
                print("⚠️ 总结可能不够全面")
        else:
            print(f"❌ 总结生成失败: {summary_result.get('message')}")
        
        # 测试场景4：上下文长度处理
        print(f"\n{'='*50}")
        print("📏 测试场景4：上下文长度处理")
        print(f"{'='*50}")
        
        # 连续问多个问题，测试长上下文处理
        long_conversation = [
            "什么是民法典？",
            "它包含哪些内容？",
            "合同编在其中的地位如何？",
            "与之前的合同法有什么区别？"
        ]
        
        print("开始长对话测试...")
        for i, question in enumerate(long_conversation, 1):
            print(f"\n--- 长对话第 {i} 轮 ---")
            print(f"🙋 用户: {question}")
            
            result = await qa_system.ask_question(
                question=question,
                answer_style="simple",
                show_details=False
            )
            
            if result.get("success"):
                answer = result["answer"]
                print(f"🤖 助手: {answer[:150]}...")  # 显示更少内容以节省空间
                
                # 检查是否能处理逐渐增长的上下文
                execution_time = result.get("execution_time", 0)
                if execution_time < 120:  # 2分钟内完成
                    print(f"✅ 响应时间正常: {execution_time:.1f}秒")
                else:
                    print(f"⚠️ 响应时间较长: {execution_time:.1f}秒")
            else:
                print(f"❌ 回答失败: {result.get('message')}")
            
            await asyncio.sleep(0.5)  # 短暂等待
        
        # 测试场景5：会话列表和管理
        print(f"\n{'='*50}")
        print("📋 测试场景5：会话管理功能")
        print(f"{'='*50}")
        
        # 获取会话列表
        sessions = qa_system.list_sessions()
        print(f"📝 当前会话数量: {len(sessions)}")
        for session in sessions:
            print(f"   会话ID: {session.get('session_id')}")
            print(f"   标题: {session.get('title', '未命名')}")
            print(f"   创建时间: {session.get('created_at', '未知')}")
            print(f"   消息数: {session.get('message_count', 0)}")
        
        # 测试结果总结
        print(f"\n{'='*60}")
        print("🎯 上下文功能测试结果总结")
        print(f"{'='*60}")
        
        # 获取系统状态
        system_info = qa_system.get_system_info()
        if system_info.get("ready"):
            status = system_info.get("system_status", {})
            memory_info = status.get("memory_info", {})
            execution_stats = status.get("execution_stats", {})
            
            print(f"📊 系统统计:")
            print(f"   🔧 总执行任务: {execution_stats.get('total_tasks', 0)}")
            print(f"   ✅ 成功率: {execution_stats.get('successful_tasks', 0)}/{execution_stats.get('total_tasks', 0)}")
            print(f"   ⏱️ 平均耗时: {execution_stats.get('avg_execution_time', 0):.2f}秒")
            print(f"   💾 活跃会话: {memory_info.get('sessions_count', 0)}")
            print(f"   📝 对话消息数: {memory_info.get('total_messages', 0)}")
        
        print(f"\n✅ 上下文功能测试完成！")
        print(f"主要验证了:")
        print(f"   🧠 多轮对话记忆能力")
        print(f"   🔄 会话切换与隔离")
        print(f"   📋 对话总结生成")
        print(f"   📏 长上下文处理")
        print(f"   📋 会话管理功能")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_context_functionality()) 