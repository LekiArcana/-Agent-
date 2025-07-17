#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Langchain 的检索 Agent
专门负责法律条文的智能检索和相关性分析
"""
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import Tool
from .llm_config import get_default_llm
from .tools import law_retrieval_tool, get_tools
from .memory_manager import get_memory_manager


class LawRetrievalAgent:
    """法律检索 Agent"""
    
    def __init__(self, 
                 llm: Optional[BaseLanguageModel] = None,
                 max_iterations: int = 3):
        """
        初始化检索 Agent
        
        Args:
            llm: 语言模型实例
            max_iterations: 最大迭代次数
        """
        self.llm = llm or get_default_llm()
        self.max_iterations = max_iterations
        self.memory_manager = get_memory_manager()
        self.agent_executor = None
        self._initialized = False
    
    def _validate_react_format(self, text: str) -> bool:
        """验证ReAct格式是否正确"""
        lines = text.strip().split('\n')
        has_thought = False
        has_action_after_thought = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('Thought:'):
                has_thought = True
                has_action_after_thought = False
            elif line.startswith('Action:') and has_thought:
                has_action_after_thought = True
            elif line.startswith('Final Answer:'):
                return True  # 直接给出最终答案也是合法的
                
        return has_action_after_thought or not has_thought
    
    def _fix_react_format(self, text: str) -> str:
        """尝试修复ReAct格式"""
        if "Final Answer:" in text:
            # 如果已经有Final Answer，直接返回
            return text
            
        if text.strip().startswith('Thought:') and 'Action:' not in text:
            # 如果只有Thought没有Action，添加默认的检索Action
            lines = text.strip().split('\n')
            thought_line = lines[0] if lines else "Thought: 我需要检索相关法律条文"
            
            return f"""{thought_line}
Action: law_retrieval
Action Input: {{"query": "法律条文检索", "k": 3, "min_score": 0.4}}"""
        
        return text
    
    def initialize(self) -> bool:
        """初始化 Agent"""
        if self._initialized:
            return True
        
        try:
            # 创建符合 ReAct 格式的提示模板
            retrieval_prompt = PromptTemplate.from_template("""
你是一个专业的法律条文检索专家。根据用户的法律问题，使用工具检索相关的法律条文。

对话历史：
{chat_history}

问题：{input}

你可以使用的工具：
{tool_names}

{tools}

重要提示：你必须严格按照以下格式回答，每行都不能省略：

Thought: 我需要思考如何回答这个问题
Action: 要使用的工具名称
Action Input: 工具的输入参数（必须是有效的JSON格式）
Observation: 工具返回的结果
... (这个 Thought/Action/Action Input/Observation 可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

格式要求：
1. 每个"Thought:"后面必须跟一个"Action:"
2. 每个"Action:"后面必须跟一个"Action Input:"
3. Action Input必须是有效的JSON格式，例如: {{"query": "查询内容", "k": 3, "min_score": 0.4}}
4. 不要使用query="..."这种格式，必须使用JSON格式
5. 如果不需要使用工具，直接写"Final Answer:"
6. 绝对不能省略任何标签

示例正确格式：
Action: law_retrieval
Action Input: {{"query": "什么是合同", "k": 3, "min_score": 0.4}}

开始！

{agent_scratchpad}""")
            
            # 获取检索相关的工具
            tools = [law_retrieval_tool]
            
            # 创建 ReAct Agent
            agent = create_react_agent(
                llm=self.llm,
                tools=tools,
                prompt=retrieval_prompt
            )
            
            # 创建 Agent 执行器
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=self.max_iterations,
                verbose=True,
                handle_parsing_errors="Check your output and make sure it conforms to the expected format! Use the correct format: Thought: ... Action: ... Action Input: ... or Final Answer: ...",
                return_intermediate_steps=True,
                max_execution_time=60,  # 增加超时时间
                early_stopping_method="generate"  # 改进停止方法
            )
            
            self._initialized = True
            print("✅ 检索 Agent 初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ 检索 Agent 初始化失败: {e}")
            return False
    
    async def retrieve(self, 
                      query: str,
                      retrieval_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行法律条文检索
        
        Args:
            query: 查询问题
            retrieval_params: 检索参数 (k, min_score等)
        
        Returns:
            检索结果和分析
        """
        if not self._initialized:
            if not self.initialize():
                return {"error": "Agent 初始化失败"}
        
        try:
            # 获取对话历史作为上下文
            context_vars = self.memory_manager.get_context_variables()
            chat_history = context_vars.get("chat_history", [])
            
            # 格式化对话历史
            formatted_history = ""
            if chat_history:
                history_parts = []
                for msg in chat_history[-6:]:  # 取最近3轮对话
                    if hasattr(msg, 'content'):
                        if msg.__class__.__name__ == "HumanMessage":
                            history_parts.append(f"用户: {msg.content}")
                        elif msg.__class__.__name__ == "AIMessage":
                            history_parts.append(f"助手: {msg.content}")
                formatted_history = "\n".join(history_parts)
            
            # 构造输入
            agent_input = {
                "input": query,
                "chat_history": formatted_history
            }
            
            # 如果有检索参数，将其加入到问题中
            if retrieval_params:
                param_text = f"检索参数：{retrieval_params}"
                agent_input["input"] = f"{query}\n\n{param_text}"
            
            print(f"🔍 开始检索: {query}")
            
            # 执行 Agent (带重试机制)
            max_retries = 2
            result = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = self.agent_executor.invoke(agent_input)
                    break
                except Exception as e:
                    error_str = str(e)
                    if "Invalid Format" in error_str or "Missing 'Action:'" in error_str:
                        print(f"⚠️  检测到ReAct格式错误 (尝试 {attempt + 1}/{max_retries + 1}): {error_str[:100]}...")
                        
                        if attempt < max_retries:
                            print(f"🔄 重试中...")
                            # 可以在这里调整输入或参数
                            continue
                        else:
                            print(f"❌ 格式错误重试失败，使用备用策略")
                            # 备用策略：直接调用检索工具
                            from .tools import law_retrieval_tool
                            backup_result = law_retrieval_tool.invoke({
                                "query": query,
                                "k": 3,
                                "min_score": 0.4
                            })
                            result = {
                                "output": backup_result,
                                "intermediate_steps": []
                            }
                            break
                    else:
                        # 其他错误直接抛出
                        raise e
            
            if result is None:
                raise Exception("Agent执行失败")
            
            # 处理结果
            response = {
                "success": True,
                "query": query,
                "analysis": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "agent_type": "retrieval"
            }
            
            # 提取具体的检索结果（优先从 intermediate_steps 中提取）
            retrieved_docs = []
            raw_retrieval_results = []
            
            # 从 intermediate_steps 中提取检索工具的原始结果
            for step in result.get("intermediate_steps", []):
                if len(step) >= 2 and hasattr(step[0], 'tool') and step[0].tool == "law_retrieval":
                    raw_result = step[1]
                    if raw_result and raw_result.strip():
                        raw_retrieval_results.append(raw_result)
            
            # 处理检索结果
            if raw_retrieval_results:
                # 合并所有检索结果
                combined_content = "\n".join(raw_retrieval_results)
                retrieved_docs.append({
                    "content": combined_content,
                    "source": "law_retrieval_tool",
                    "raw_count": len(raw_retrieval_results)
                })
            
            # 如果没有从 intermediate_steps 提取到结果，从 Final Answer 中提取
            if not retrieved_docs and result.get("output"):
                final_answer = result.get("output", "")
                if final_answer.strip() and "【检索结果" in final_answer:
                    retrieved_docs.append({
                        "content": final_answer,
                        "source": "final_answer"
                    })
            
            # 如果仍然没有结果，添加默认提示
            if not retrieved_docs:
                retrieved_docs.append({
                    "content": "检索过程完成，但未找到明确的法律条文结果。",
                    "source": "default_message"
                })
            
            response["retrieved_documents"] = retrieved_docs
            response["document_count"] = len(retrieved_docs)
            
            print(f"✅ 检索完成，找到 {len(retrieved_docs)} 个相关文档")
            
            return response
            
        except Exception as e:
            print(f"❌ 检索执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "agent_type": "retrieval"
            }
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        # 这里可以添加检索统计逻辑
        return {
            "agent_type": "retrieval",
            "status": "ready" if self._initialized else "not_initialized",
            "max_iterations": self.max_iterations,
            "available_tools": ["law_retrieval"]
        }


class EnhancedRetrievalAgent(LawRetrievalAgent):
    """增强版检索 Agent，支持多轮检索策略"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrieval_history = []
    
    async def adaptive_retrieve(self, 
                              query: str,
                              context: Optional[str] = None) -> Dict[str, Any]:
        """
        自适应检索，根据上下文调整检索策略
        
        Args:
            query: 查询问题
            context: 额外上下文信息
        
        Returns:
            检索结果
        """
        # 分析查询复杂度
        query_complexity = self._analyze_query_complexity(query)
        
        # 根据复杂度调整检索参数
        if query_complexity == "simple":
            params = {"k": 3, "min_score": 0.4}
        elif query_complexity == "moderate":
            params = {"k": 5, "min_score": 0.3}
        else:  # complex
            params = {"k": 8, "min_score": 0.25}
        
        # 执行检索
        result = await self.retrieve(query, params)
        
        # 记录检索历史
        self.retrieval_history.append({
            "query": query,
            "complexity": query_complexity,
            "params": params,
            "success": result.get("success", False),
            "doc_count": result.get("document_count", 0)
        })
        
        return result
    
    def _analyze_query_complexity(self, query: str) -> str:
        """分析查询复杂度"""
        # 简单的复杂度分析逻辑
        keywords_count = len(query.split())
        question_marks = query.count("？") + query.count("?")
        legal_terms = ["法律", "条文", "规定", "法规", "责任", "义务", "权利"]
        legal_term_count = sum(1 for term in legal_terms if term in query)
        
        if keywords_count <= 10 and question_marks <= 1 and legal_term_count <= 2:
            return "simple"
        elif keywords_count <= 20 and legal_term_count <= 4:
            return "moderate"
        else:
            return "complex"
    
    def get_retrieval_insights(self) -> Dict[str, Any]:
        """获取检索洞察信息"""
        if not self.retrieval_history:
            return {"message": "暂无检索历史"}
        
        total_queries = len(self.retrieval_history)
        successful_queries = sum(1 for h in self.retrieval_history if h["success"])
        avg_docs = sum(h["doc_count"] for h in self.retrieval_history) / total_queries
        
        complexity_stats = {}
        for h in self.retrieval_history:
            complexity = h["complexity"]
            if complexity not in complexity_stats:
                complexity_stats[complexity] = {"count": 0, "success_rate": 0}
            complexity_stats[complexity]["count"] += 1
        
        return {
            "total_queries": total_queries,
            "success_rate": successful_queries / total_queries,
            "average_documents_found": avg_docs,
            "complexity_distribution": complexity_stats,
            "recent_queries": self.retrieval_history[-5:]  # 最近5次查询
        }


# 创建全局检索 Agent 实例
retrieval_agent = EnhancedRetrievalAgent()


def get_retrieval_agent() -> EnhancedRetrievalAgent:
    """获取全局检索 Agent 实例"""
    return retrieval_agent 