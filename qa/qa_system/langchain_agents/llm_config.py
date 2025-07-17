#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 配置模块
支持本地 Ollama 服务的 qwen2.5 模型
"""
from typing import Optional, Dict, Any
from langchain_community.llms import Ollama
from langchain_core.language_models.base import BaseLanguageModel


class OllamaConfig:
    """Ollama 配置管理器"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model_name: str = "qwen2.5"):
        self.base_url = base_url
        self.model_name = model_name
        self._llm: Optional[BaseLanguageModel] = None
    
    def get_llm(self, **kwargs) -> BaseLanguageModel:
        """获取配置好的 LLM 实例"""
        if self._llm is None:
            # 为 qwen2.5 优化的默认参数
            default_params = {
                "temperature": 0.1,  # 降低温度提高格式稳定性
                "top_p": 0.8,
                "top_k": 20,
                "repeat_penalty": 1.1,
                "num_predict": 1000,
            }
            
            # 合并用户参数
            params = {**default_params, **kwargs}
            
            self._llm = Ollama(
                base_url=self.base_url,
                model=self.model_name,
                **params
            )
        
        return self._llm
    
    def test_connection(self) -> bool:
        """测试 Ollama 连接"""
        try:
            llm = self.get_llm()
            # 简单测试调用
            response = llm.invoke("你好")
            return bool(response and len(response.strip()) > 0)
        except Exception as e:
            print(f"Ollama 连接测试失败: {e}")
            return False
    
    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "connection_status": self.test_connection()
        }


# 全局配置实例
ollama_config = OllamaConfig()


def get_default_llm(**kwargs) -> BaseLanguageModel:
    """获取默认的 LLM 实例"""
    return ollama_config.get_llm(**kwargs)


def test_llm_connection() -> bool:
    """测试 LLM 连接状态"""
    return ollama_config.test_connection() 