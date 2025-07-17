#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Langchain 的总结 Agent
专门负责对话总结、法律内容分析和洞察提取
"""
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from .llm_config import get_default_llm
from .tools import content_analysis_tool
from .memory_manager import get_memory_manager


class LawSummaryAgent:
    """法律总结 Agent"""
    
    def __init__(self, 
                 llm: Optional[BaseLanguageModel] = None):
        """
        初始化总结 Agent
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm or get_default_llm()
        self.memory_manager = get_memory_manager()
        self.summary_chain = None
        self.analysis_chain = None
        self._initialized = False
        
        # 不同类型的总结模板
        self.summary_templates = {
            "conversation": self._get_conversation_summary_template(),
            "legal_content": self._get_legal_content_template(),
            "session_insights": self._get_session_insights_template(),
            "law_analysis": self._get_law_analysis_template()
        }
    
    def _get_conversation_summary_template(self) -> str:
        """对话总结模板"""
        return """请对以下法律咨询对话进行总结：

对话内容：
{content}

请从以下方面进行总结：

1. **咨询主题**：用户主要咨询的法律问题
2. **涉及法律领域**：涉及哪些法律领域和相关法规
3. **核心问题**：对话中的核心法律问题和争议点
4. **提供建议**：助手提供的主要法律建议和指导
5. **后续事项**：需要进一步关注或处理的事项

要求：
- 总结简洁明了，重点突出
- 保留重要的法律术语和关键信息
- 结构清晰，便于理解

总结："""

    def _get_legal_content_template(self) -> str:
        """法律内容分析模板"""
        return """请对以下法律内容进行分析：

法律内容：
{content}

请从以下方面进行分析：

1. **内容性质**：分析内容的法律性质和类型
2. **关键条款**：提取关键的法律条款和规定
3. **适用范围**：说明适用的对象、情形和条件
4. **法律后果**：分析违反或适用该内容的法律后果
5. **实务要点**：指出在实际应用中需要注意的要点
6. **相关法律**：列出相关的其他法律法规

要求：
- 分析客观准确，逻辑清晰
- 重点突出实用性
- 避免过度解读

分析："""

    def _get_session_insights_template(self) -> str:
        """会话洞察模板"""
        return """请对以下法律咨询会话进行深度分析，提供洞察：

会话数据：
{content}

请提供以下洞察：

1. **咨询模式**：分析用户的咨询模式和特点
2. **问题演进**：分析问题的发展和深入过程
3. **知识需求**：分析用户的法律知识需求层次
4. **满意度评估**：评估咨询效果和用户满意度
5. **改进建议**：对未来咨询服务的改进建议

要求：
- 分析深入客观
- 重点关注用户体验
- 提供可行的建议

洞察："""

    def _get_law_analysis_template(self) -> str:
        """法律条文分析模板"""
        return """请对以下法律条文进行专业分析：

法律条文：
{content}

请进行以下分析：

1. **条文解读**：逐条解读条文的含义和要求
2. **立法目的**：分析条文的立法目的和保护对象
3. **适用条件**：说明条文的适用条件和例外情况
4. **操作指引**：提供条文在实践中的操作指引
5. **案例应用**：分析条文在典型案例中的应用
6. **注意事项**：指出理解和应用中的注意事项

要求：
- 分析专业准确
- 结合实践经验
- 通俗易懂

分析："""

    def initialize(self) -> bool:
        """初始化 Agent"""
        if self._initialized:
            return True
        
        try:
            # 创建总结链
            summary_prompt = PromptTemplate(
                input_variables=["content"],
                template=self.summary_templates["conversation"]
            )
            
            self.summary_chain = LLMChain(
                llm=self.llm,
                prompt=summary_prompt,
                output_parser=StrOutputParser()
            )
            
            # 创建分析链
            analysis_prompt = PromptTemplate(
                input_variables=["content"],
                template=self.summary_templates["legal_content"]
            )
            
            self.analysis_chain = LLMChain(
                llm=self.llm,
                prompt=analysis_prompt,
                output_parser=StrOutputParser()
            )
            
            self._initialized = True
            print("✅ 总结 Agent 初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ 总结 Agent 初始化失败: {e}")
            return False
    
    async def summarize_conversation(self, 
                                   session_id: Optional[str] = None,
                                   max_length: int = 500) -> Dict[str, Any]:
        """
        总结对话内容
        
        Args:
            session_id: 会话ID，None表示当前会话
            max_length: 最大总结长度
        
        Returns:
            总结结果
        """
        if not self._initialized:
            if not self.initialize():
                return {"error": "Agent 初始化失败"}
        
        try:
            # 获取对话历史
            chat_history = self.memory_manager.get_chat_history(session_id)
            
            if not chat_history:
                return {
                    "success": False,
                    "message": "暂无对话历史可供总结",
                    "agent_type": "summary"
                }
            
            # 格式化对话内容
            conversation_text = ""
            for i, msg in enumerate(chat_history):
                if hasattr(msg, 'content'):
                    role = "用户" if msg.__class__.__name__ == "HumanMessage" else "助手"
                    conversation_text += f"{role}: {msg.content}\n\n"
            
            print(f"📝 总结对话 (共{len(chat_history)}条消息)...")
            
            # 执行总结
            summary_result = await self.summary_chain.ainvoke({"content": conversation_text})
            summary_text = summary_result["text"] if isinstance(summary_result, dict) else summary_result
            
            # 限制长度
            if len(summary_text) > max_length:
                summary_text = summary_text[:max_length] + "..."
            
            # 获取会话统计
            session_stats = self.memory_manager.get_session_stats(session_id)
            
            response = {
                "success": True,
                "summary": summary_text,
                "session_stats": session_stats,
                "message_count": len(chat_history),
                "agent_type": "summary"
            }
            
            print("✅ 对话总结完成")
            
            return response
            
        except Exception as e:
            print(f"❌ 对话总结失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_type": "summary"
            }
    
    async def analyze_legal_content(self, 
                                  content: str,
                                  analysis_type: str = "legal_content") -> Dict[str, Any]:
        """
        分析法律内容
        
        Args:
            content: 要分析的内容
            analysis_type: 分析类型
        
        Returns:
            分析结果
        """
        if not self._initialized:
            if not self.initialize():
                return {"error": "Agent 初始化失败"}
        
        try:
            # 获取对应的模板
            template = self.summary_templates.get(analysis_type, self.summary_templates["legal_content"])
            
            # 创建专用的分析链
            analysis_prompt = PromptTemplate(
                input_variables=["content"],
                template=template
            )
            
            analysis_chain = LLMChain(
                llm=self.llm,
                prompt=analysis_prompt,
                output_parser=StrOutputParser()
            )
            
            print(f"🔍 分析法律内容 (类型: {analysis_type})...")
            
            # 执行分析
            analysis_result = await analysis_chain.ainvoke({"content": content})
            analysis_text = analysis_result["text"] if isinstance(analysis_result, dict) else analysis_result
            
            response = {
                "success": True,
                "analysis": analysis_text,
                "analysis_type": analysis_type,
                "content_length": len(content),
                "agent_type": "summary"
            }
            
            print("✅ 法律内容分析完成")
            
            return response
            
        except Exception as e:
            print(f"❌ 法律内容分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type,
                "agent_type": "summary"
            }
    
    async def generate_session_insights(self, 
                                      session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        生成会话洞察
        
        Args:
            session_id: 会话ID
        
        Returns:
            洞察结果
        """
        if not self._initialized:
            if not self.initialize():
                return {"error": "Agent 初始化失败"}
        
        try:
            # 获取会话统计和历史
            session_stats = self.memory_manager.get_session_stats(session_id)
            chat_history = self.memory_manager.get_chat_history(session_id)
            
            if not chat_history:
                return {
                    "success": False,
                    "message": "暂无会话数据可供分析",
                    "agent_type": "summary"
                }
            
            # 构造分析数据
            analysis_data = {
                "stats": session_stats,
                "message_count": len(chat_history),
                "recent_topics": []
            }
            
            # 提取最近的话题
            for msg in chat_history[-6:]:  # 最近3轮对话
                if hasattr(msg, 'content') and msg.__class__.__name__ == "HumanMessage":
                    analysis_data["recent_topics"].append(msg.content[:50])
            
            content = f"会话统计：{analysis_data}"
            
            print("📊 生成会话洞察...")
            
            # 使用洞察模板进行分析
            result = await self.analyze_legal_content(content, "session_insights")
            
            if result.get("success"):
                result["session_id"] = session_id or self.memory_manager.current_session_id
                result["analysis_data"] = analysis_data
            
            return result
            
        except Exception as e:
            print(f"❌ 会话洞察生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_type": "summary"
            }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取总结统计信息"""
        return {
            "agent_type": "summary",
            "status": "ready" if self._initialized else "not_initialized",
            "available_analysis_types": list(self.summary_templates.keys())
        }


class EnhancedSummaryAgent(LawSummaryAgent):
    """增强版总结 Agent，支持批量总结和智能分类"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_history = []
    
    async def batch_summarize(self, 
                            session_ids: List[str],
                            summary_type: str = "conversation") -> Dict[str, Any]:
        """
        批量总结多个会话
        
        Args:
            session_ids: 会话ID列表
            summary_type: 总结类型
        
        Returns:
            批量总结结果
        """
        results = []
        
        for session_id in session_ids:
            if summary_type == "conversation":
                result = await self.summarize_conversation(session_id)
            elif summary_type == "insights":
                result = await self.generate_session_insights(session_id)
            else:
                result = {"error": f"不支持的总结类型: {summary_type}"}
            
            result["session_id"] = session_id
            results.append(result)
        
        # 统计成功率
        successful = sum(1 for r in results if r.get("success", False))
        success_rate = successful / len(results) if results else 0
        
        return {
            "success": True,
            "batch_results": results,
            "total_sessions": len(session_ids),
            "successful_summaries": successful,
            "success_rate": success_rate,
            "summary_type": summary_type
        }
    
    async def smart_content_analysis(self, content: str) -> Dict[str, Any]:
        """
        智能内容分析，自动选择合适的分析类型
        
        Args:
            content: 分析内容
        
        Returns:
            分析结果
        """
        # 简单的内容类型识别
        analysis_type = self._detect_content_type(content)
        
        # 执行对应类型的分析
        result = await self.analyze_legal_content(content, analysis_type)
        
        if result.get("success"):
            result["detected_type"] = analysis_type
            result["content_preview"] = content[:100] + "..." if len(content) > 100 else content
        
        return result
    
    def _detect_content_type(self, content: str) -> str:
        """检测内容类型"""
        content_lower = content.lower()
        
        # 检查是否为对话内容
        if "用户:" in content and "助手:" in content:
            return "conversation"
        
        # 检查是否为法律条文
        elif any(keyword in content for keyword in ["第", "条", "款", "项", "法律", "法规"]):
            return "law_analysis"
        
        # 检查是否为会话数据
        elif "stats" in content_lower or "session" in content_lower:
            return "session_insights"
        
        # 默认为法律内容分析
        else:
            return "legal_content"
    
    def get_summary_insights(self) -> Dict[str, Any]:
        """获取总结洞察"""
        if not self.summary_history:
            return {"message": "暂无总结历史"}
        
        # 统计不同类型的总结数量
        type_counts = {}
        for summary in self.summary_history:
            summary_type = summary.get("type", "unknown")
            type_counts[summary_type] = type_counts.get(summary_type, 0) + 1
        
        return {
            "total_summaries": len(self.summary_history),
            "type_distribution": type_counts,
            "recent_summaries": self.summary_history[-5:]
        }


# 创建全局总结 Agent 实例
summary_agent = EnhancedSummaryAgent()


def get_summary_agent() -> EnhancedSummaryAgent:
    """获取全局总结 Agent 实例"""
    return summary_agent 