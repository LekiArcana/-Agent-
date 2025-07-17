#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Langchain 工具定义
包含法律检索、内容分析等专用工具
"""
import os
import sys
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# 添加父目录到路径，以便导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LawRetrievalInput(BaseModel):
    """法律检索工具输入模型"""
    query: str = Field(description="法律查询问题")
    k: int = Field(default=5, description="返回结果数量")
    min_score: float = Field(default=0.1, description="最小相似度分数")


class LawRetriever:
    """法律检索器（复用现有FAISS索引）"""
    
    def __init__(self):
        # 使用基于当前脚本的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(script_dir))
        data_dir = os.path.join(project_dir, "Data")
        
        self.index_file = os.path.join(data_dir, "law_index.faiss")
        self.metadata_file = os.path.join(data_dir, "law_metadata.pkl")
        self.model_name = "BAAI/bge-large-zh-v1.5"
        self.index = None
        self.segments = None
        self.model = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化检索器"""
        if self._initialized:
            return True
            
        try:
            # 加载模型
            print("正在加载embedding模型...")
            try:
                self.model = SentenceTransformer(self.model_name)
                print(f"BGE中文模型加载成功: {self.model_name}")
            except Exception as e:
                print(f"主模型加载失败: {e}")
                backup_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                self.model = SentenceTransformer(backup_model)
                print(f"备选模型加载成功: {backup_model}")
            
            # 加载FAISS索引
            if not os.path.exists(self.index_file):
                print(f"索引文件不存在: {self.index_file}")
                return False
            
            self.index = faiss.read_index(self.index_file)
            print(f"索引加载成功！包含 {self.index.ntotal} 个向量")
            
            # 加载元数据
            if not os.path.exists(self.metadata_file):
                print(f"元数据文件不存在: {self.metadata_file}")
                return False
                
            with open(self.metadata_file, 'rb') as f:
                self.segments = pickle.load(f)
            print(f"元数据加载成功！包含 {len(self.segments)} 个法律条款")
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"检索器初始化失败: {e}")
            return False
    
    def retrieve(self, query: str, k: int = 5, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """检索相关法律条款"""
        if not self._initialized:
            if not self.initialize():
                return []
        
        try:
            # 获取查询向量
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            query_embedding = query_embedding.astype(np.float32)
            
            # 标准化查询向量
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # 搜索
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # 检查有效索引
                    segment = self.segments[idx].copy()
                    # IndexFlatIP返回的是内积值（相似度），直接使用
                    similarity_score = float(distances[0][i])
                    segment['similarity_score'] = similarity_score
                    
                    # 应用最小分数过滤
                    if similarity_score >= min_score:
                        results.append(segment)
            
            return results
            
        except Exception as e:
            print(f"检索失败: {e}")
            return []


# 全局检索器实例
_law_retriever = None

def get_law_retriever():
    """获取或创建检索器实例"""
    global _law_retriever
    if _law_retriever is None:
        _law_retriever = LawRetriever()
    return _law_retriever


def clean_query(query: str) -> str:
    """清理查询字符串，移除影响检索的元素"""
    import json
    import re
    
    cleaned = query.strip()
    
    # 1. 检查是否是JSON格式的参数字符串（Agent框架bug导致）
    if cleaned.startswith('{') and cleaned.endswith('}'):
        try:
            # 尝试解析JSON并提取真正的查询
            json_data = json.loads(cleaned)
            if 'query' in json_data:
                cleaned = json_data['query']
                print(f"🔧 检测到JSON格式查询，已提取: '{cleaned}'")
        except json.JSONDecodeError:
            # 如果JSON解析失败，保持原查询
            pass
    
    # 2. 检查是否是 query="..." 格式（ReAct Agent错误格式）
    query_pattern = r'query\s*=\s*["\']([^"\']+)["\']'
    match = re.search(query_pattern, cleaned)
    if match:
        cleaned = match.group(1)
        print(f"🔧 检测到query=格式查询，已提取: '{cleaned}'")
    
    # 3. 检查是否包含其他参数格式，提取query部分
    if 'query=' in cleaned and ('k=' in cleaned or 'min_score=' in cleaned):
        # 尝试提取query部分
        parts = cleaned.split(',')
        for part in parts:
            if 'query=' in part:
                query_part = part.strip()
                # 移除 query= 前缀和引号
                query_part = re.sub(r'^.*query\s*=\s*["\']?', '', query_part)
                query_part = re.sub(r'["\'].*$', '', query_part)
                if query_part.strip():
                    cleaned = query_part.strip()
                    print(f"🔧 检测到复合参数格式，已提取query: '{cleaned}'")
                    break
    
    # 4. 移除结尾的问号（保留句子中的问号）
    if cleaned.endswith('？') or cleaned.endswith('?'):
        cleaned = cleaned[:-1]
    
    # 5. 如果查询太长，只取前面的核心部分
    if '\n' in cleaned:
        cleaned = cleaned.split('\n')[0].strip()
    
    return cleaned

@tool("law_retrieval", args_schema=LawRetrievalInput)
def law_retrieval_tool(query: str, k: int = 5, min_score: float = 0.1) -> str:
    """
    法律条文检索工具。
    
    使用向量相似度搜索找到与查询问题最相关的法律条文。
    
    Args:
        query: 查询问题
        k: 返回结果数量（默认5）
        min_score: 最小相似度分数（默认0.3）
    
    Returns:
        格式化的法律条文结果
    """
    # 清理查询以提高检索精度
    cleaned_query = clean_query(query)
    retriever = get_law_retriever()
    results = retriever.retrieve(cleaned_query, k, min_score)
    
    if not results:
        return "未找到相关的法律条文。"
    
    # 格式化结果
    formatted_results = []
    for i, result in enumerate(results, 1):
        article_info = f"【检索结果 {i}】\n"
        article_info += f"法律名称：{result['law_name']}\n"
        if result.get('chapter'):
            article_info += f"章节：{result['chapter']}\n"
        if result.get('section'):
            article_info += f"节：{result['section']}\n"
        article_info += f"条文：{result['article']}\n"
        article_info += f"内容：{result['content']}\n"
        article_info += f"相似度：{result['similarity_score']:.3f}\n"
        formatted_results.append(article_info)
    
    return "\n" + "\n".join(formatted_results)


class ContentAnalysisInput(BaseModel):
    """内容分析工具输入模型"""
    content: str = Field(description="需要分析的法律内容")
    analysis_type: str = Field(default="summary", description="分析类型：summary, key_points, legal_advice")


@tool("content_analysis", args_schema=ContentAnalysisInput)
def content_analysis_tool(content: str, analysis_type: str = "summary") -> str:
    """
    法律内容分析工具。
    
    对给定的法律内容进行分析，提取关键信息。
    
    Args:
        content: 需要分析的内容
        analysis_type: 分析类型（summary/key_points/legal_advice）
    
    Returns:
        分析结果
    """
    if not content.strip():
        return "内容为空，无法进行分析。"
    
    # 根据分析类型返回不同的格式化指引
    if analysis_type == "summary":
        return f"【内容总结】\n需要对以下内容进行总结：\n{content[:500]}..."
    elif analysis_type == "key_points":
        return f"【关键要点】\n需要提取以下内容的关键法律要点：\n{content[:500]}..."
    elif analysis_type == "legal_advice":
        return f"【法律建议】\n需要基于以下内容提供法律建议：\n{content[:500]}..."
    else:
        return f"【内容分析】\n需要分析以下内容：\n{content[:500]}..."


# 工具列表
ALL_TOOLS = [
    law_retrieval_tool,
    content_analysis_tool
]


def get_tools() -> List:
    """获取所有可用工具"""
    return ALL_TOOLS


def initialize_tools() -> List:
    """初始化所有工具"""
    retriever = get_law_retriever()
    if retriever.initialize():
        return ALL_TOOLS
    else:
        return [] 