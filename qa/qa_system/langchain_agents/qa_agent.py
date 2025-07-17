#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Langchain 的问答 Agent
专门负责根据检索到的法律条文生成专业准确的法律回答
"""
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from .llm_config import get_default_llm
from .tools import content_analysis_tool
from .memory_manager import get_memory_manager


class LawQAAgent:
    """法律问答 Agent"""
    
    def __init__(self, 
                 llm: Optional[BaseLanguageModel] = None,
                 answer_style: str = "professional"):
        """
        初始化问答 Agent
        
        Args:
            llm: 语言模型实例
            answer_style: 回答风格 (professional/simple/detailed)
        """
        self.llm = llm or get_default_llm()
        self.answer_style = answer_style
        self.memory_manager = get_memory_manager()
        self.qa_chain = None
        self._initialized = False
        
        # 定义不同风格的提示模板
        self.style_prompts = {
            "professional": self._get_professional_prompt(),
            "simple": self._get_simple_prompt(),
            "detailed": self._get_detailed_prompt()
        }
    
    def _get_professional_prompt(self) -> str:
        """专业风格提示模板"""
        return """你是一位资深的法律顾问，具有深厚的法学功底和丰富的实务经验。请基于提供的法律条文，为用户提供专业、准确、连贯的法律咨询服务。

对话历史：
{chat_history}

用户问题：
{question}

相关法律条文：
{legal_context}

请提供专业的法律回答，要求：
- 严格基于提供的法律条文进行分析，明确指出适用的具体法律条款
- 全面阐述相关的法律规定和可能的法律后果
- 结合实务经验提供实用的法律建议，并指出需要注意的法律风险
- 语言专业规范，逻辑清晰，回答连贯完整
- 如果条文不足以完全回答问题，请明确说明
- 对于复杂案件建议咨询专业律师，避免提供具体的诉讼指导

回答："""

    def _get_simple_prompt(self) -> str:
        """简单风格提示模板"""
        return """你是一个友善的法律助手，请用通俗易懂的语言回答用户的法律问题。

对话历史：
{chat_history}

用户问题：
{question}

相关法律条文：
{legal_context}

请用简单明了的语言回答用户的问题：
- 先用通俗的话解释相关的法律规定，避免使用过多专业术语
- 语言通俗易懂，贴近生活，重点突出,字数尽量少而精
- 如果情况比较复杂，建议咨询专业律师

回答："""

    def _get_detailed_prompt(self) -> str:
        """详细风格提示模板"""
        return """你是一位法学教授和实务专家，请提供详细、全面的法律分析。

对话历史：
{chat_history}

用户问题：
{question}

相关法律条文：
{legal_context}

请提供详细全面的法律分析：
首先深入分析问题的法律性质和争议焦点，然后逐条解读相关法律条文的含义和适用条件，阐述相关的法理基础和立法目的。进一步结合司法实践和案例进行分析，如果存在争议，应列举不同的法律观点。在此基础上全面评估各种可能的法律风险，最后提供具体的操作指导和注意事项。

要求分析全面深入、逻辑严密，引用具体的法律条文和编号，考虑不同情况和例外情形，提供可操作的建议。对于复杂案件应强调需要专业律师的指导。

回答："""

    def initialize(self) -> bool:
        """初始化 Agent"""
        if self._initialized:
            return True
        
        try:
            # 创建问答链
            prompt_template = self.style_prompts.get(self.answer_style, self.style_prompts["professional"])
            prompt = PromptTemplate(
                input_variables=["question", "legal_context", "chat_history"],
                template=prompt_template
            )
            
            self.qa_chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_parser=StrOutputParser()
            )
            
            self._initialized = True
            print(f"✅ 问答 Agent 初始化成功 (风格: {self.answer_style})")
            return True
            
        except Exception as e:
            print(f"❌ 问答 Agent 初始化失败: {e}")
            return False
    
    async def answer(self, 
                    question: str,
                    legal_context: str,
                    additional_context: Optional[str] = None) -> Dict[str, Any]:
        """
        生成法律问答
        
        Args:
            question: 用户问题
            legal_context: 法律条文上下文
            additional_context: 额外上下文信息
        
        Returns:
            问答结果
        """
        if not self._initialized:
            if not self.initialize():
                return {"error": "Agent 初始化失败"}
        
        try:
            # 获取对话历史
            formatted_history = self.memory_manager.format_chat_history_for_context()
            
            # 准备上下文
            context = legal_context
            if additional_context:
                context += f"\n\n补充信息：\n{additional_context}"
            
            # 构造输入
            chain_input = {
                "question": question,
                "legal_context": context,
                "chat_history": formatted_history
            }
            
            print(f"💬 生成回答: {question[:50]}...")
            
            # 执行问答链
            answer = await self.qa_chain.ainvoke(chain_input)
            
            # 处理结果
            response = {
                "success": True,
                "question": question,
                "answer": answer["text"] if isinstance(answer, dict) else answer,
                "answer_style": self.answer_style,
                "context_used": bool(legal_context),
                "agent_type": "qa"
            }
            
            print("✅ 回答生成完成")
            
            return response
            
        except Exception as e:
            print(f"❌ 问答生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "agent_type": "qa"
            }
    
    def set_answer_style(self, style: str) -> bool:
        """设置回答风格"""
        if style in self.style_prompts:
            self.answer_style = style
            self._initialized = False  # 需要重新初始化
            print(f"📝 回答风格已设置为: {style}")
            return True
        else:
            print(f"❌ 不支持的回答风格: {style}")
            return False
    
    def get_available_styles(self) -> List[str]:
        """获取可用的回答风格"""
        return list(self.style_prompts.keys())
    
    def get_qa_statistics(self) -> Dict[str, Any]:
        """获取问答统计信息"""
        return {
            "agent_type": "qa",
            "status": "ready" if self._initialized else "not_initialized",
            "current_style": self.answer_style,
            "available_styles": self.get_available_styles()
        }


class EnhancedQAAgent(LawQAAgent):
    """增强版问答 Agent，支持多轮对话和上下文理解"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qa_history = []
        self.context_keywords = []
    
    async def contextual_answer(self, 
                              question: str,
                              legal_context: str,
                              follow_up: bool = False) -> Dict[str, Any]:
        """
        上下文感知的问答
        
        Args:
            question: 用户问题
            legal_context: 法律条文上下文
            follow_up: 是否为追问
        
        Returns:
            问答结果
        """
        # 分析问题类型
        question_type = self._analyze_question_type(question)
        
        # 如果是追问，添加上下文关键词
        if follow_up and self.qa_history:
            last_qa = self.qa_history[-1]
            enhanced_context = f"{legal_context}\n\n【上下文】\n上一轮问题：{last_qa['question']}\n上一轮回答：{last_qa['answer'][:200]}..."
        else:
            enhanced_context = legal_context
        
        # 执行问答
        result = await self.answer(question, enhanced_context)
        
        # 记录问答历史
        if result.get("success"):
            self.qa_history.append({
                "question": question,
                "answer": result["answer"],
                "question_type": question_type,
                "timestamp": "now",  # 可以改为实际时间戳
                "follow_up": follow_up
            })
            
            # 更新上下文关键词
            self._update_context_keywords(question, result["answer"])
        
        # 添加问题类型信息
        result["question_type"] = question_type
        result["follow_up"] = follow_up
        
        return result
    
    def _analyze_question_type(self, question: str) -> str:
        """分析问题类型"""
        question_lower = question.lower()
        
        if any(word in question for word in ["怎么办", "如何", "怎样", "怎么做"]):
            return "how_to"  # 操作指导类
        elif any(word in question for word in ["是什么", "什么是", "定义", "含义"]):
            return "definition"  # 定义解释类
        elif any(word in question for word in ["是否", "能否", "可以", "允许", "禁止"]):
            return "yes_no"  # 是否判断类
        elif any(word in question for word in ["责任", "后果", "处罚", "赔偿"]):
            return "consequence"  # 后果责任类
        elif any(word in question for word in ["程序", "流程", "步骤", "手续"]):
            return "procedure"  # 程序流程类
        else:
            return "general"  # 一般咨询类
    
    def _update_context_keywords(self, question: str, answer: str):
        """更新上下文关键词"""
        # 简单的关键词提取逻辑
        legal_terms = ["法律", "条文", "规定", "法规", "责任", "义务", "权利", "合同", "侵权"]
        
        question_keywords = [term for term in legal_terms if term in question]
        answer_keywords = [term for term in legal_terms if term in answer]
        
        new_keywords = list(set(question_keywords + answer_keywords))
        
        # 更新关键词列表（保持最近的10个）
        self.context_keywords.extend(new_keywords)
        self.context_keywords = list(set(self.context_keywords))[-10:]
    
    def get_conversation_insights(self) -> Dict[str, Any]:
        """获取对话洞察"""
        if not self.qa_history:
            return {"message": "暂无对话历史"}
        
        # 统计问题类型分布
        type_count = {}
        for qa in self.qa_history:
            q_type = qa.get("question_type", "unknown")
            type_count[q_type] = type_count.get(q_type, 0) + 1
        
        # 统计追问比例
        follow_up_count = sum(1 for qa in self.qa_history if qa.get("follow_up", False))
        follow_up_rate = follow_up_count / len(self.qa_history) if self.qa_history else 0
        
        return {
            "total_questions": len(self.qa_history),
            "question_type_distribution": type_count,
            "follow_up_rate": follow_up_rate,
            "context_keywords": self.context_keywords,
            "recent_questions": [qa["question"] for qa in self.qa_history[-3:]]
        }
    
    def reset_context(self):
        """重置上下文"""
        self.qa_history = []
        self.context_keywords = []
        print("🔄 问答上下文已重置")


# 创建全局问答 Agent 实例
qa_agent = EnhancedQAAgent()


def get_qa_agent() -> EnhancedQAAgent:
    """获取全局问答 Agent 实例"""
    return qa_agent


def create_qa_agent_with_style(style: str) -> EnhancedQAAgent:
    """创建指定风格的问答 Agent"""
    agent = EnhancedQAAgent(answer_style=style)
    return agent 