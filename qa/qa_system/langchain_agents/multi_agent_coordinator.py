#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Langchain 的多 Agent 协作系统
实现检索、问答、总结 Agent 的协调工作和多轮对话支持
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .llm_config import get_default_llm, test_llm_connection
from .tools import initialize_tools
from .memory_manager import get_memory_manager
from .retrieval_agent import get_retrieval_agent
from .qa_agent import get_qa_agent
from .summary_agent import get_summary_agent


class TaskType(Enum):
    """任务类型枚举"""
    QA = "qa"  # 问答任务
    RETRIEVAL = "retrieval"  # 检索任务
    SUMMARY = "summary"  # 总结任务
    ANALYSIS = "analysis"  # 分析任务
    MULTI_TURN = "multi_turn"  # 多轮对话任务


@dataclass
class AgentTask:
    """Agent 任务定义"""
    task_id: str
    task_type: TaskType
    input_data: Dict[str, Any]
    priority: int = 1  # 优先级，数字越小优先级越高
    depends_on: List[str] = None  # 依赖的任务ID列表
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


@dataclass
class AgentResult:
    """Agent 执行结果"""
    task_id: str
    agent_type: str
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MultiAgentCoordinator:
    """多 Agent 协作协调器"""
    
    def __init__(self):
        """初始化协调器"""
        self.memory_manager = get_memory_manager()
        self.retrieval_agent = get_retrieval_agent()
        self.qa_agent = get_qa_agent()
        self.summary_agent = get_summary_agent()
        
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: Dict[str, AgentResult] = {}
        self.is_initialized = False
        
        # 执行统计
        self.execution_stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_execution_time": 0.0,
            "agent_usage": {
                "retrieval": 0,
                "qa": 0,
                "summary": 0
            }
        }
    
    async def initialize(self) -> bool:
        """初始化协调器和所有 Agent"""
        if self.is_initialized:
            return True
        
        print("🚀 初始化多 Agent 协作系统...")
        
        try:
            # 测试 LLM 连接
            print("🔗 测试 LLM 连接...")
            if not test_llm_connection():
                print("❌ LLM 连接失败，请确保 Ollama 服务正在运行")
                return False
            
            # 初始化工具
            print("🛠️ 初始化工具...")
            if not initialize_tools():
                print("❌ 工具初始化失败")
                return False
            
            # 初始化各个 Agent
            agents_init_tasks = [
                ("检索 Agent", self.retrieval_agent.initialize()),
                ("问答 Agent", self.qa_agent.initialize()),
                ("总结 Agent", self.summary_agent.initialize())
            ]
            
            for agent_name, init_result in agents_init_tasks:
                if not init_result:
                    print(f"❌ {agent_name} 初始化失败")
                    return False
                print(f"✅ {agent_name} 初始化成功")
            
            # 创建默认会话
            if not self.memory_manager.current_session_id:
                session_id = self.memory_manager.create_session("Langchain 多 Agent 会话")
                print(f"📝 创建默认会话: {session_id}")
            
            self.is_initialized = True
            print("🎉 多 Agent 协作系统初始化完成！")
            
            return True
            
        except Exception as e:
            print(f"❌ 协调器初始化失败: {e}")
            return False
    
    async def process_user_question(self, 
                                  question: str,
                                  answer_style: str = "professional",
                                  retrieval_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理用户问题的完整流程
        
        Args:
            question: 用户问题
            answer_style: 回答风格
            retrieval_params: 检索参数
        
        Returns:
            完整的处理结果
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        task_id = f"qa_task_{int(time.time())}"
        
        print(f"\n📋 开始处理问题: {question}")
        print("=" * 60)
        
        try:
            # 阶段1: 检索相关法律条文
            print("🔍 阶段1: 检索相关法律条文")
            retrieval_result = await self.retrieval_agent.adaptive_retrieve(
                query=question,
                context=retrieval_params
            )
            
            if not retrieval_result.get("success"):
                return {
                    "success": False,
                    "error": "检索阶段失败",
                    "details": retrieval_result
                }
            
            # 提取检索到的法律条文
            legal_context = ""
            retrieved_docs = retrieval_result.get("retrieved_documents", [])
            
            if retrieved_docs:
                legal_context = "\n".join([doc["content"] for doc in retrieved_docs])
            else:
                # 从分析结果中提取
                analysis = retrieval_result.get("analysis", "")
                if "【检索结果" in analysis:
                    legal_context = analysis
            
            if not legal_context:
                legal_context = "未找到直接相关的法律条文。"
            
            print(f"✅ 检索完成，获得 {len(retrieved_docs)} 个相关文档")
            
            # 阶段2: 生成专业回答
            print("💬 阶段2: 生成专业回答")
            
            # 设置问答风格
            self.qa_agent.set_answer_style(answer_style)
            
            qa_result = await self.qa_agent.contextual_answer(
                question=question,
                legal_context=legal_context,
                follow_up=self._is_follow_up_question(question)
            )
            
            if not qa_result.get("success"):
                return {
                    "success": False,
                    "error": "问答阶段失败",
                    "details": qa_result
                }
            
            print("✅ 回答生成完成")
            
            # 保存到内存
            self.memory_manager.add_message(question, qa_result["answer"])
            
            # 更新统计
            self._update_execution_stats("qa", time.time() - start_time, True)
            
            # 构造最终结果
            final_result = {
                "success": True,
                "task_id": task_id,
                "question": question,
                "answer": qa_result["answer"],
                "answer_style": answer_style,
                "retrieval_info": {
                    "documents_found": len(retrieved_docs),
                    "retrieval_analysis": retrieval_result.get("analysis", ""),
                    "documents": retrieved_docs
                },
                "qa_info": {
                    "question_type": qa_result.get("question_type", "general"),
                    "follow_up": qa_result.get("follow_up", False),
                    "context_used": qa_result.get("context_used", False)
                },
                "execution_time": time.time() - start_time,
                "session_id": self.memory_manager.current_session_id
            }
            
            print(f"🎯 问题处理完成 (耗时: {final_result['execution_time']:.2f}秒)")
            
            return final_result
            
        except Exception as e:
            print(f"❌ 问题处理失败: {e}")
            self._update_execution_stats("qa", time.time() - start_time, False)
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id,
                "question": question
            }
    
    async def generate_conversation_summary(self, 
                                          session_id: Optional[str] = None) -> Dict[str, Any]:
        """生成对话总结"""
        if not self.is_initialized:
            await self.initialize()
        
        print("📝 生成对话总结...")
        
        start_time = time.time()
        
        try:
            # 使用总结 Agent 生成总结
            summary_result = await self.summary_agent.summarize_conversation(session_id)
            
            # 更新统计
            self._update_execution_stats("summary", time.time() - start_time, 
                                       summary_result.get("success", False))
            
            return summary_result
            
        except Exception as e:
            print(f"❌ 对话总结生成失败: {e}")
            self._update_execution_stats("summary", time.time() - start_time, False)
            return {
                "success": False,
                "error": str(e),
                "agent_type": "summary"
            }
    
    async def analyze_legal_content(self, 
                                  content: str,
                                  analysis_type: str = "legal_content") -> Dict[str, Any]:
        """分析法律内容"""
        if not self.is_initialized:
            await self.initialize()
        
        print(f"🔍 分析法律内容 (类型: {analysis_type})...")
        
        start_time = time.time()
        
        try:
            # 使用总结 Agent 进行内容分析
            analysis_result = await self.summary_agent.analyze_legal_content(content, analysis_type)
            
            # 更新统计
            self._update_execution_stats("summary", time.time() - start_time,
                                       analysis_result.get("success", False))
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ 法律内容分析失败: {e}")
            self._update_execution_stats("summary", time.time() - start_time, False)
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type
            }
    
    def _is_follow_up_question(self, question: str) -> bool:
        """判断是否为追问"""
        # 简单的追问检测逻辑
        follow_up_indicators = [
            "那么", "那", "如果", "假如", "还有", "另外", "此外",
            "进一步", "具体", "详细", "比如", "例如"
        ]
        
        return any(indicator in question for indicator in follow_up_indicators)
    
    def _update_execution_stats(self, agent_type: str, execution_time: float, success: bool):
        """更新执行统计"""
        self.execution_stats["total_tasks"] += 1
        
        if success:
            self.execution_stats["successful_tasks"] += 1
        else:
            self.execution_stats["failed_tasks"] += 1
        
        # 更新平均执行时间
        total_time = (self.execution_stats["avg_execution_time"] * 
                     (self.execution_stats["total_tasks"] - 1) + execution_time)
        self.execution_stats["avg_execution_time"] = total_time / self.execution_stats["total_tasks"]
        
        # 更新 Agent 使用统计
        if agent_type in self.execution_stats["agent_usage"]:
            self.execution_stats["agent_usage"][agent_type] += 1
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "initialized": self.is_initialized,
            "current_session": self.memory_manager.current_session_id,
            "execution_stats": self.execution_stats,
            "agents_status": {
                "retrieval": self.retrieval_agent.get_retrieval_statistics(),
                "qa": self.qa_agent.get_qa_statistics(),
                "summary": self.summary_agent.get_summary_statistics()
            },
            "memory_info": {
                "sessions_count": len(self.memory_manager.sessions),
                "current_session_stats": self.memory_manager.get_session_stats()
            }
        }
    
    def create_new_session(self, title: str = "新对话") -> str:
        """创建新的对话会话"""
        return self.memory_manager.create_session(title)
    
    def switch_session(self, session_id: str) -> bool:
        """切换到指定会话"""
        return self.memory_manager.switch_session(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话"""
        return self.memory_manager.list_sessions()
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """获取指定会话的历史消息"""
        return self.memory_manager.get_session_messages(session_id)
    
    async def batch_process_questions(self, 
                                    questions: List[str],
                                    answer_style: str = "professional") -> Dict[str, Any]:
        """批量处理问题"""
        if not self.is_initialized:
            await self.initialize()
        
        print(f"📋 开始批量处理 {len(questions)} 个问题...")
        
        results = []
        start_time = time.time()
        
        for i, question in enumerate(questions, 1):
            print(f"\n处理问题 {i}/{len(questions)}: {question[:50]}...")
            
            result = await self.process_user_question(question, answer_style)
            results.append(result)
            
            # 短暂休息，避免过度负载
            await asyncio.sleep(0.5)
        
        # 统计批量处理结果
        successful = sum(1 for r in results if r.get("success", False))
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "total_questions": len(questions),
            "successful_answers": successful,
            "success_rate": successful / len(questions),
            "total_time": total_time,
            "average_time_per_question": total_time / len(questions),
            "results": results
        }


# 创建全局协调器实例
multi_agent_coordinator = MultiAgentCoordinator()


def get_coordinator() -> MultiAgentCoordinator:
    """获取全局协调器实例"""
    return multi_agent_coordinator 