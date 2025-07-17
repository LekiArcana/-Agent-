#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Langchain 的内存和会话管理
支持多轮对话和上下文维护
"""
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models.base import BaseLanguageModel
from .llm_config import get_default_llm


class ChatSession:
    """聊天会话类"""
    
    def __init__(self, session_id: str, title: str = "新对话"):
        self.session_id = session_id
        self.title = title
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.message_count = 0
        self.context_summary = ""
        
    def update(self):
        """更新会话信息"""
        self.updated_at = datetime.now()
        self.message_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
            "context_summary": self.context_summary
        }


class LangchainMemoryManager:
    """基于 Langchain 的内存管理器"""
    
    def __init__(self, 
                 llm: Optional[BaseLanguageModel] = None,
                 max_token_limit: int = 2000,
                 window_size: int = 10):
        """
        初始化内存管理器
        
        Args:
            llm: 语言模型实例
            max_token_limit: 最大token限制
            window_size: 滑动窗口大小
        """
        self.llm = llm or get_default_llm()
        self.max_token_limit = max_token_limit
        self.window_size = window_size
        
        # 会话存储
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
        
        # 内存存储：为每个会话创建独立的内存
        self.session_memories: Dict[str, ConversationSummaryBufferMemory] = {}
        
    def create_session(self, title: str = "新对话") -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id, title)
        self.sessions[session_id] = session
        
        # 为新会话创建内存
        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=self.max_token_limit,
            return_messages=True,
            memory_key="chat_history"
        )
        self.session_memories[session_id] = memory
        
        self.current_session_id = session_id
        print(f"创建新会话: {session_id}")
        return session_id
    
    def switch_session(self, session_id: str) -> bool:
        """切换到指定会话"""
        if session_id in self.sessions:
            self.current_session_id = session_id
            print(f"切换到会话: {session_id}")
            return True
        else:
            print(f"会话不存在: {session_id}")
            return False
    
    def get_current_session(self) -> Optional[ChatSession]:
        """获取当前会话"""
        if self.current_session_id:
            return self.sessions.get(self.current_session_id)
        return None
    
    def get_current_memory(self) -> Optional[ConversationSummaryBufferMemory]:
        """获取当前会话的内存"""
        if self.current_session_id:
            return self.session_memories.get(self.current_session_id)
        return None
    
    def add_message(self, user_input: str, ai_response: str):
        """添加对话消息到当前会话"""
        if not self.current_session_id:
            self.create_session()
        
        memory = self.get_current_memory()
        session = self.get_current_session()
        
        if memory and session:
            # 添加到内存
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(ai_response)
            
            # 更新会话信息
            session.update()
            
            # 更新会话标题（如果是第一条消息）
            if session.message_count == 1:
                session.title = user_input[:30] + "..." if len(user_input) > 30 else user_input
    
    def get_chat_history(self, session_id: Optional[str] = None) -> List[BaseMessage]:
        """获取对话历史"""
        target_session_id = session_id or self.current_session_id
        if target_session_id and target_session_id in self.session_memories:
            memory = self.session_memories[target_session_id]
            return memory.chat_memory.messages
        return []
    
    def get_context_variables(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取上下文变量，用于传递给 Agent"""
        target_session_id = session_id or self.current_session_id
        memory = self.session_memories.get(target_session_id) if target_session_id else None
        
        if memory:
            return memory.load_memory_variables({})
        return {"chat_history": []}
    
    def get_session_summary(self, session_id: Optional[str] = None) -> str:
        """获取会话总结"""
        target_session_id = session_id or self.current_session_id
        memory = self.session_memories.get(target_session_id) if target_session_id else None
        
        if memory and hasattr(memory, 'moving_summary_buffer') and memory.moving_summary_buffer:
            return memory.moving_summary_buffer
        return "暂无对话总结"
    
    def clear_session(self, session_id: Optional[str] = None):
        """清空指定会话的对话历史"""
        target_session_id = session_id or self.current_session_id
        if target_session_id and target_session_id in self.session_memories:
            memory = self.session_memories[target_session_id]
            memory.clear()
            
            # 重置会话信息
            if target_session_id in self.sessions:
                session = self.sessions[target_session_id]
                session.message_count = 0
                session.context_summary = ""
    
    def delete_session(self, session_id: str):
        """删除指定会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.session_memories:
            del self.session_memories[session_id]
        
        # 如果删除的是当前会话，则创建新会话
        if session_id == self.current_session_id:
            self.create_session()
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话"""
        return [session.to_dict() for session in self.sessions.values()]
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """获取指定会话的历史消息"""
        if session_id not in self.session_memories:
            return []
        
        memory = self.session_memories[session_id]
        messages = memory.chat_memory.messages
        
        # 转换为前端可用的格式
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({
                    "type": "user",
                    "content": msg.content,
                    "timestamp": getattr(msg, 'timestamp', datetime.now().isoformat())
                })
            elif isinstance(msg, AIMessage):
                formatted_messages.append({
                    "type": "assistant", 
                    "content": msg.content,
                    "timestamp": getattr(msg, 'timestamp', datetime.now().isoformat())
                })
        
        return formatted_messages
    
    def get_session_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取会话统计信息"""
        target_session_id = session_id or self.current_session_id
        if not target_session_id or target_session_id not in self.sessions:
            return {"error": "会话不存在"}
        
        session = self.sessions[target_session_id]
        memory = self.session_memories[target_session_id]
        
        # 统计消息数量
        messages = memory.chat_memory.messages
        user_messages = sum(1 for msg in messages if isinstance(msg, HumanMessage))
        ai_messages = sum(1 for msg in messages if isinstance(msg, AIMessage))
        
        return {
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "total_exchanges": session.message_count,
            "user_messages": user_messages,
            "ai_messages": ai_messages,
            "has_summary": bool(self.get_session_summary(target_session_id)),
            "memory_size": len(messages)
        }
    
    def format_chat_history_for_context(self, 
                                      session_id: Optional[str] = None,
                                      max_messages: int = 10) -> str:
        """格式化对话历史为上下文字符串"""
        messages = self.get_chat_history(session_id)
        if not messages:
            return "暂无对话历史"
        
        # 取最近的消息
        recent_messages = messages[-max_messages:]
        
        formatted_history = []
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"助手: {msg.content}")
        
        return "\n".join(formatted_history)


# 全局内存管理器实例
memory_manager = LangchainMemoryManager()


def get_memory_manager() -> LangchainMemoryManager:
    """获取全局内存管理器实例"""
    return memory_manager 