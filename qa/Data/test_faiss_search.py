#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试FAISS向量索引的搜索功能
"""
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

class LawSearchEngine:
    def __init__(self, index_file="law_index.faiss", metadata_file="law_metadata.pkl"):
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = None
        self.segments = None
        self.model = None
        
    def load_index_and_metadata(self):
        """加载索引和元数据"""
        try:
            # 加载FAISS索引
            print("正在加载FAISS索引...")
            self.index = faiss.read_index(self.index_file)
            print(f"索引加载成功！包含 {self.index.ntotal} 个向量，维度 {self.index.d}")
            
            # 加载元数据
            print("正在加载元数据...")
            with open(self.metadata_file, 'rb') as f:
                self.segments = pickle.load(f)
            print(f"元数据加载成功！包含 {len(self.segments)} 个法律条款")
            
            return True
        except Exception as e:
            print(f"加载失败: {e}")
            return False
    
    def load_model(self):
        """加载embedding模型"""
        try:
            print("正在加载embedding模型...")
            # 尝试加载中文BGE模型
            try:
                self.model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
                print("BGE中文模型加载成功！")
            except:
                # 备选模型
                self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                print("多语言备选模型加载成功！")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """搜索相关法律条款"""
        if not self.model or not self.index or not self.segments:
            print("请先加载模型和索引！")
            return []
        
        try:
            # 获取查询向量
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            query_embedding = query_embedding.astype(np.float32)
            
            # 标准化查询向量
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # 搜索
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # 有效索引
                    segment = self.segments[idx].copy()
                    segment['similarity_score'] = float(scores[0][i])
                    results.append(segment)
            
            return results
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def get_index_info(self):
        """获取索引信息"""
        if not self.index or not self.segments:
            return None
        
        # 统计各个法律的条款数量
        law_stats = {}
        for segment in self.segments:
            law_name = segment['law_name']
            if law_name not in law_stats:
                law_stats[law_name] = 0
            law_stats[law_name] += 1
        
        return {
            'total_vectors': self.index.ntotal,
            'vector_dimension': self.index.d,
            'total_segments': len(self.segments),
            'law_statistics': law_stats
        }
    
    def interactive_search(self):
        """交互式搜索"""
        print("\n=== 法律条款智能搜索系统 ===")
        print("输入查询内容，系统将返回相关的法律条款")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'info' 查看索引信息")
        print("输入 'demo' 运行演示查询\n")
        
        while True:
            query = input("请输入查询内容: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                print("退出搜索系统")
                break
            
            if query.lower() == 'info':
                self.show_index_info()
                continue
            
            if query.lower() == 'demo':
                self.run_demo_queries()
                continue
            
            if not query:
                print("请输入有效的查询内容")
                continue
            
            # 执行搜索
            print(f"\n正在搜索: {query}")
            results = self.search(query, k=5)
            
            if results:
                print(f"找到 {len(results)} 个相关条款:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. 【{result['law_name']}】")
                    print(f"   条款: {result['article']}")
                    if result['chapter']:
                        print(f"   章节: {result['chapter']}")
                    if result['section']:
                        print(f"   节: {result['section']}")
                    print(f"   内容: {result['content'][:200]}...")
                    print(f"   相似度: {result['similarity_score']:.4f}")
            else:
                print("未找到相关条款")
    
    def show_index_info(self):
        """显示索引信息"""
        info = self.get_index_info()
        if info:
            print(f"\n=== 索引信息 ===")
            print(f"总向量数: {info['total_vectors']}")
            print(f"向量维度: {info['vector_dimension']}")
            print(f"总条款数: {info['total_segments']}")
            print(f"\n各法律条款统计:")
            for law, count in sorted(info['law_statistics'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {law}: {count} 条")
    
    def run_demo_queries(self):
        """运行演示查询"""
        demo_queries = [
            "公司设立的条件",
            "股东的权利和义务",
            "合同的效力",
            "刑事责任年龄",
            "劳动者的权利",
            "环境保护的法律责任",
            "知识产权保护",
            "婚姻家庭关系",
            "房屋买卖合同",
            "交通事故处理"
        ]
        
        print(f"\n=== 演示查询 ===")
        for query in demo_queries:
            print(f"\n查询: {query}")
            results = self.search(query, k=3)
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. 【{result['law_name']}】{result['article']}")
                    print(f"     {result['content'][:100]}...")
                    print(f"     相似度: {result['similarity_score']:.4f}")
            else:
                print("  未找到相关条款")

def main():
    # 创建搜索引擎
    search_engine = LawSearchEngine()
    
    # 加载索引和元数据
    if not search_engine.load_index_and_metadata():
        print("索引加载失败，请先运行 law_vectorization.py 构建索引")
        return
    
    # 加载模型
    if not search_engine.load_model():
        print("模型加载失败")
        return
    
    # 显示索引信息
    search_engine.show_index_info()
    
    # 启动交互式搜索
    search_engine.interactive_search()

if __name__ == "__main__":
    main() 