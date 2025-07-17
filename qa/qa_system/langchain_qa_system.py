#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Langchain 的多 Agent 法律问答系统
主接口文件 - 提供完整的系统功能
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from langchain_agents.multi_agent_coordinator import get_coordinator


class LangchainLegalQASystem:
    """基于 Langchain 的法律问答系统"""
    
    def __init__(self):
        """初始化系统"""
        self.coordinator = get_coordinator()
        self.is_ready = False
    
    async def initialize(self) -> bool:
        """初始化系统"""
        print("=" * 70)
        print("🏛️  基于 Langchain 的多 Agent 法律问答系统")
        print("=" * 70)
        print("🚀 新一代智能法律咨询系统")
        print("   📖 检索 Agent: 智能法律条文检索")
        print("   💬 问答 Agent: 专业法律问答生成")
        print("   📝 总结 Agent: 对话总结与内容分析")
        print("   🧠 内存管理: 多轮对话上下文维护")
        print("   🔗 工具集成: Langchain + Ollama + FAISS")
        print("=" * 70)
        
        success = await self.coordinator.initialize()
        if success:
            self.is_ready = True
            print("🎉 Langchain 多 Agent 系统初始化完成！")
            await self._show_system_info()
        else:
            print("❌ 系统初始化失败")
        
        return success
    
    async def ask_question(self, 
                          question: str,
                          answer_style: str = "professional",
                          show_details: bool = False) -> Dict[str, Any]:
        """
        提问接口
        
        Args:
            question: 用户问题
            answer_style: 回答风格 (professional/simple/detailed)
            show_details: 是否显示详细信息
        
        Returns:
            问答结果
        """
        if not self.is_ready:
            return {
                "success": False,
                "message": "系统未初始化",
                "answer": "抱歉，系统尚未准备就绪，请稍后重试。"
            }
        
        print(f"\n📝 问题: {question}")
        print(f"📋 回答风格: {answer_style}")
        print("-" * 60)
        
        # 处理问题
        result = await self.coordinator.process_user_question(
            question=question,
            answer_style=answer_style
        )
        
        if result.get("success"):
            print(f"\n💡 回答:\n{result['answer']}")
            
            if show_details:
                self._show_execution_details(result)
            
            return {
                "success": True,
                "question": question,
                "answer": result["answer"],
                "answer_style": answer_style,
                "execution_time": result.get("execution_time", 0),
                "session_id": result.get("session_id"),
                "retrieval_info": result.get("retrieval_info", {}),
                "qa_info": result.get("qa_info", {})
            }
        else:
            error_msg = result.get("error", "未知错误")
            print(f"❌ 问答失败: {error_msg}")
            return {
                "success": False,
                "message": error_msg,
                "answer": "抱歉，我无法回答这个问题，请稍后重试或换个方式提问。"
            }
    
    async def get_conversation_summary(self) -> Dict[str, Any]:
        """获取当前对话总结"""
        if not self.is_ready:
            return {"success": False, "message": "系统未初始化"}
        
        return await self.coordinator.generate_conversation_summary()
    
    async def analyze_content(self, 
                            content: str,
                            analysis_type: str = "legal_content") -> Dict[str, Any]:
        """分析法律内容"""
        if not self.is_ready:
            return {"success": False, "message": "系统未初始化"}
        
        return await self.coordinator.analyze_legal_content(content, analysis_type)
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        if not self.is_ready:
            return {"ready": False}
        
        status = self.coordinator.get_system_status()
        
        # 检查 agents 状态 - 如果所有 agent 都有统计信息则认为正常
        agents_status = status.get("agents_status", {})
        all_agents_healthy = all(
            agent_stats and isinstance(agent_stats, dict) 
            for agent_stats in agents_status.values()
        ) if agents_status else False
        
        return {
            "ready": True,
            "system_name": "Langchain 多 Agent 法律问答系统",
            "version": "2.0.0",
            "framework": "Langchain",
            "llm_model": "qwen2.5",
            "embedding_model": "BAAI/bge-large-zh-v1.5",
            "vector_count": status.get("memory_info", {}).get("vector_count", 4573),
            "tools_count": 2,  # law_retrieval 和 content_analysis
            "agents_status": all_agents_healthy,
            "startup_time": "2024-12-09T12:00:00",  # 可以替换为实际启动时间
            "llm_backend": "Ollama (qwen2.5)",
            "vector_db": "FAISS",
            "system_status": status,
            "capabilities": [
                "多轮对话支持",
                "智能上下文维护",
                "专业法律问答",
                "智能条文检索", 
                "对话总结分析",
                "内容智能分析",
                "多会话管理",
                "批量问题处理"
            ],
            "agent_types": [
                {
                    "name": "检索 Agent",
                    "description": "智能法律条文检索和相关性分析",
                    "capabilities": ["向量检索", "相关性评估", "自适应参数调整"]
                },
                {
                    "name": "问答 Agent", 
                    "description": "专业法律问答生成",
                    "capabilities": ["多风格回答", "上下文理解", "问题类型识别"]
                },
                {
                    "name": "总结 Agent",
                    "description": "对话总结和内容分析",
                    "capabilities": ["对话总结", "内容分析", "洞察生成"]
                }
            ],
            "detailed_agents_status": agents_status  # 保留详细信息用于调试
        }
    
    def create_new_session(self, title: str = "新对话") -> str:
        """创建新会话"""
        if self.is_ready:
            session_id = self.coordinator.create_new_session(title)
            print(f"📝 创建新会话: {title} ({session_id})")
            return session_id
        return ""
    
    def switch_session(self, session_id: str) -> bool:
        """切换会话"""
        if self.is_ready:
            success = self.coordinator.switch_session(session_id)
            if success:
                print(f"🔄 切换到会话: {session_id}")
            else:
                print(f"❌ 会话切换失败: {session_id}")
            return success
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话"""
        if self.is_ready:
            return self.coordinator.list_sessions()
        return []
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """获取指定会话的历史消息"""
        if self.is_ready:
            return self.coordinator.get_session_messages(session_id)
        return []
    
    async def batch_questions(self, 
                            questions: List[str],
                            answer_style: str = "professional") -> Dict[str, Any]:
        """批量处理问题"""
        if not self.is_ready:
            return {"success": False, "message": "系统未初始化"}
        
        return await self.coordinator.batch_process_questions(questions, answer_style)
    
    def _show_execution_details(self, result: Dict[str, Any]):
        """显示执行详情"""
        print(f"\n📊 执行详情:")
        print(f"   ⏱️  总耗时: {result.get('execution_time', 0):.2f}秒")
        
        retrieval_info = result.get("retrieval_info", {})
        print(f"   🔍 检索信息: 找到 {retrieval_info.get('documents_found', 0)} 个相关文档")
        
        qa_info = result.get("qa_info", {})
        print(f"   💬 问答信息: 类型={qa_info.get('question_type', 'general')}, "
              f"追问={qa_info.get('follow_up', False)}")
        
        print(f"   📝 会话ID: {result.get('session_id', 'unknown')}")
    
    async def _show_system_info(self):
        """显示系统信息"""
        status = self.coordinator.get_system_status()
        
        print(f"\n📊 系统状态:")
        print(f"   🔧 已执行任务: {status['execution_stats']['total_tasks']}")
        print(f"   ✅ 成功率: {status['execution_stats']['successful_tasks']}/{status['execution_stats']['total_tasks']}")
        print(f"   ⏱️  平均耗时: {status['execution_stats']['avg_execution_time']:.2f}秒")
        print(f"   💾 活跃会话: {status['memory_info']['sessions_count']}")
    
    async def interactive_mode(self):
        """交互式模式"""
        if not self.is_ready:
            print("❌ 系统未初始化")
            return
        
        print("\n" + "="*70)
        print("🏛️  进入 Langchain 多 Agent 法律问答交互模式")
        print("="*70)
        print("💡 可用命令:")
        print("   📝 直接输入法律问题进行咨询")
        print("   📋 'summary' - 获取对话总结")
        print("   📊 'status' - 查看系统状态")
        print("   🆕 'new' [标题] - 创建新会话")
        print("   🔄 'switch <会话ID>' - 切换会话")
        print("   📃 'sessions' - 列出所有会话")
        print("   🎨 'style <风格>' - 设置回答风格 (professional/simple/detailed)")
        print("   🔍 'analyze <内容>' - 分析法律内容")
        print("   ❓ 'help' - 查看帮助")
        print("   🚪 'quit' - 退出系统")
        print("="*70)
        
        current_style = "professional"
        
        while True:
            try:
                user_input = input("\n💬 请输入 > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 感谢使用 Langchain 多 Agent 法律问答系统，再见！")
                    break
                
                elif user_input.lower() == 'help':
                    print(self._get_help_text())
                
                elif user_input.lower() == 'summary':
                    print("📝 生成对话总结...")
                    summary_result = await self.get_conversation_summary()
                    if summary_result.get("success"):
                        print(f"\n📋 对话总结:\n{summary_result['summary']}")
                    else:
                        print(f"❌ 总结生成失败: {summary_result.get('message', '未知错误')}")
                
                elif user_input.lower() == 'status':
                    print("📊 系统状态:")
                    info = self.get_system_info()
                    status = info['system_status']
                    print(f"   📈 执行统计: {json.dumps(status['execution_stats'], indent=2, ensure_ascii=False)}")
                
                elif user_input.lower() == 'sessions':
                    sessions = self.list_sessions()
                    print(f"\n📃 会话列表 (共{len(sessions)}个):")
                    for session in sessions:
                        print(f"   📁 {session['title']} ({session['session_id'][:8]}...)")
                        print(f"      💬 消息数: {session['message_count']}, 更新时间: {session['updated_at']}")
                
                elif user_input.lower().startswith('new'):
                    parts = user_input.split(' ', 1)
                    title = parts[1] if len(parts) > 1 else "新对话"
                    session_id = self.create_new_session(title)
                    print(f"✅ 新会话创建成功: {session_id}")
                
                elif user_input.lower().startswith('switch'):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        session_id = parts[1]
                        if self.switch_session(session_id):
                            print(f"✅ 已切换到会话: {session_id}")
                        else:
                            print(f"❌ 会话切换失败，请检查会话ID")
                    else:
                        print("❌ 请提供会话ID")
                
                elif user_input.lower().startswith('style'):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        new_style = parts[1].lower()
                        if new_style in ['professional', 'simple', 'detailed']:
                            current_style = new_style
                            print(f"✅ 回答风格已设置为: {current_style}")
                        else:
                            print("❌ 不支持的风格，可用选项: professional, simple, detailed")
                    else:
                        print(f"📋 当前回答风格: {current_style}")
                
                elif user_input.lower().startswith('analyze'):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        content = parts[1]
                        print("🔍 分析法律内容...")
                        analysis_result = await self.analyze_content(content)
                        if analysis_result.get("success"):
                            print(f"\n📊 分析结果:\n{analysis_result['analysis']}")
                        else:
                            print(f"❌ 内容分析失败: {analysis_result.get('error', '未知错误')}")
                    else:
                        print("❌ 请提供要分析的内容")
                
                else:
                    # 处理法律问题
                    result = await self.ask_question(user_input, current_style, show_details=True)
                    if not result.get("success"):
                        print(f"❌ 问答失败: {result.get('message', '未知错误')}")
            
            except KeyboardInterrupt:
                print("\n👋 用户中断，退出系统")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def _get_help_text(self) -> str:
        """获取帮助文本"""
        return """
📖 Langchain 多 Agent 法律问答系统帮助文档

🔧 基本功能:
• 直接输入法律问题，系统会自动检索相关条文并生成专业回答
• 支持多轮对话，系统会维护上下文记忆
• 支持多种回答风格：专业(professional)、简单(simple)、详细(detailed)

💬 会话管理:
• new [标题] - 创建新的对话会话
• switch <会话ID> - 切换到指定会话
• sessions - 查看所有会话列表

📊 系统功能:
• summary - 获取当前对话的智能总结
• status - 查看系统运行状态和统计信息
• analyze <内容> - 分析指定的法律内容

⚙️ 设置选项:
• style <风格> - 设置回答风格
  - professional: 专业法律顾问风格
  - simple: 通俗易懂风格
  - detailed: 详细学术分析风格

🎯 使用技巧:
• 问题越具体，回答越准确
• 可以使用"那么"、"如果"等词进行追问
• 复杂问题建议分步骤提问

❓ 如有疑问，输入 'help' 查看此帮助信息
"""


# 主程序入口
async def main():
    """主程序"""
    system = LangchainLegalQASystem()
    
    # 初始化系统
    if await system.initialize():
        # 进入交互模式
        await system.interactive_mode()
    else:
        print("❌ 系统初始化失败，请检查配置")


if __name__ == "__main__":
    asyncio.run(main()) 