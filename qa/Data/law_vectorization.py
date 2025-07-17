import os
import re
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings("ignore")

class LawTextProcessor:
    def __init__(self, law_data_dir: str = "LawData", model_name: str = "BAAI/bge-large-zh-v1.5"):
        self.law_data_dir = law_data_dir
        self.model_name = model_name
        self.max_text_length = 512  # sentence-transformers通常限制512
        
        # 加载embedding模型
        print(f"正在加载模型: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            print("模型加载成功！")
        except Exception as e:
            print(f"加载模型失败: {e}")
            # 备选模型
            backup_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            print(f"尝试加载备选模型: {backup_model}")
            self.model = SentenceTransformer(backup_model)
            print("备选模型加载成功！")
        
    def read_law_files(self) -> Dict[str, str]:
        """读取所有法律文本文件"""
        law_texts = {}
        for filename in os.listdir(self.law_data_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.law_data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    law_texts[filename] = content
                    print(f"已读取: {filename}")
        return law_texts
    
    def segment_law_text(self, text: str, law_name: str) -> List[Dict]:
        """分段法律文本"""
        lines = text.split('\n')
        
        # 提取法律标题
        title = lines[0].strip() if lines else law_name
        
        segments = []
        current_chapter = ""
        current_section = ""
        current_article = ""
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检测章节标题 - 支持中文数字和阿拉伯数字
            if re.match(r'第(?:[一二三四五六七八九十百千万零○〇]+|\d+)章', line):
                # 保存之前的条款
                if current_article and current_content:
                    segments.append({
                        'law_name': law_name,
                        'title': title,
                        'chapter': current_chapter,
                        'section': current_section,
                        'article': current_article,
                        'content': '\n'.join(current_content).strip()
                    })
                    current_content = []
                    current_article = ""
                current_chapter = line
                current_section = ""
                continue
            
            # 检测节标题 - 支持中文数字和阿拉伯数字
            if re.match(r'第(?:[一二三四五六七八九十百千万零○〇]+|\d+)节', line):
                # 保存之前的条款
                if current_article and current_content:
                    segments.append({
                        'law_name': law_name,
                        'title': title,
                        'chapter': current_chapter,
                        'section': current_section,
                        'article': current_article,
                        'content': '\n'.join(current_content).strip()
                    })
                    current_content = []
                    current_article = ""
                current_section = line
                continue
            
            # 检测条款 - 支持中文数字和阿拉伯数字
            # 匹配模式：第一条、第十条、第二十条、第一百条、第一百零一条、第二百条、第123条等
            article_match = re.match(r'第(?:[一二三四五六七八九十百千万零○〇]+|\d+)条', line)
            if article_match:
                # 保存之前的条款
                if current_article and current_content:
                    segments.append({
                        'law_name': law_name,
                        'title': title,
                        'chapter': current_chapter,
                        'section': current_section,
                        'article': current_article,
                        'content': '\n'.join(current_content).strip()
                    })
                    current_content = []
                
                current_article = article_match.group(0)
                # 提取条款内容（去除条款编号）
                content_part = line[len(current_article):].strip()
                if content_part:
                    current_content.append(content_part)
                continue
            
            # 普通内容行
            if current_article or current_chapter:
                current_content.append(line)
        
        # 保存最后一个条款
        if current_article and current_content:
            segments.append({
                'law_name': law_name,
                'title': title,
                'chapter': current_chapter,
                'section': current_section,
                'article': current_article,
                'content': '\n'.join(current_content).strip()
            })
        
        return segments
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """使用sentence-transformers获取文本向量"""
        try:
            # 限制文本长度
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
            
            # 获取embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"获取文本向量失败: {e}")
            return None
    
    def build_faiss_index(self, segments: List[Dict]) -> Tuple[faiss.Index, List[Dict]]:
        """构建FAISS索引"""
        print("正在生成文本向量...")
        
        # 准备文本列表
        texts = []
        for segment in segments:
            # 构造用于向量化的文本
            context_text = f"{segment['title']} {segment['chapter']} {segment['section']} {segment['article']} {segment['content']}"
            texts.append(context_text)
        
        # 批量生成向量
        try:
            embeddings = self.model.encode(texts, 
                                         batch_size=32,
                                         show_progress_bar=True,
                                         convert_to_numpy=True)
            embeddings = embeddings.astype(np.float32)
            
            # 创建FAISS索引
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
            
            # 标准化向量（用于余弦相似度）
            faiss.normalize_L2(embeddings)
            
            # 添加向量到索引
            index.add(embeddings)
            
            print(f"已构建FAISS索引，包含 {index.ntotal} 个向量，维度 {dimension}")
            print(f"成功处理了 {len(segments)} 个条款")
            
            return index, segments
            
        except Exception as e:
            print(f"构建索引失败: {e}")
            return None, []
    
    def save_index_and_metadata(self, index: faiss.Index, segments: List[Dict], 
                               index_file: str = "law_index.faiss", 
                               metadata_file: str = "law_metadata.pkl"):
        """保存索引和元数据"""
        # 保存FAISS索引
        faiss.write_index(index, index_file)
        
        # 保存元数据
        with open(metadata_file, 'wb') as f:
            pickle.dump(segments, f)
        
        print(f"索引已保存到: {index_file}")
        print(f"元数据已保存到: {metadata_file}")
    
    def search_similar_texts(self, query: str, index: faiss.Index, 
                           segments: List[Dict], k: int = 5) -> List[Dict]:
        """搜索相似文本"""
        # 获取查询向量
        query_embedding = self.get_text_embedding(query)
        if query_embedding is None:
            return []
        
        # 标准化查询向量
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # 有效索引
                segment = segments[idx].copy()
                segment['similarity_score'] = float(scores[0][i])
                results.append(segment)
        
        return results
    
    def process_all_laws(self):
        """处理所有法律文本"""
        # 读取法律文本
        law_texts = self.read_law_files()
        
        # 分段所有文本
        all_segments = []
        for law_name, text in law_texts.items():
            segments = self.segment_law_text(text, law_name)
            all_segments.extend(segments)
            print(f"{law_name}: 分段 {len(segments)} 个条款")
        
        print(f"总计分段: {len(all_segments)} 个条款")
        
        # 构建FAISS索引
        index, valid_segments = self.build_faiss_index(all_segments)
        
        if index is not None:
            # 保存索引和元数据
            self.save_index_and_metadata(index, valid_segments)
            
            # 测试搜索
            print("\n=== 搜索测试 ===")
            test_queries = [
                "公司设立的条件",
                "股东的权利和义务",
                "合同的效力",
                "劳动者的权利",
                "刑事责任年龄"
            ]
            
            for query in test_queries:
                print(f"\n查询: {query}")
                results = self.search_similar_texts(query, index, valid_segments, k=3)
                for i, result in enumerate(results, 1):
                    print(f"  {i}. [{result['law_name']}] {result['article']}: {result['content'][:100]}...")
                    print(f"     相似度: {result['similarity_score']:.4f}")

def main():
    # 创建处理器实例
    processor = LawTextProcessor()
    
    # 处理所有法律文本
    processor.process_all_laws()

if __name__ == "__main__":
    main() 