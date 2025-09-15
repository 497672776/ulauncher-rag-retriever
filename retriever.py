"""
轻量级检索器 - 只负责查询，不负责建立索引
"""

import os
import pickle
import json
import time
from typing import List, Dict, Any, Optional

# ChromaDB相关
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever

# BM25和分词
from rank_bm25 import BM25Okapi
import jieba


class RAGRetriever:
    """轻量级RAG检索器"""
    
    def __init__(self, data_base_path: str):
        self.data_base_path = os.path.expanduser(data_base_path)
        
        # 数据路径
        self.chroma_db_path = os.path.join(self.data_base_path, "chroma_db")
        self.pickle_path = os.path.join(self.data_base_path, "models")
        self.state_path = os.path.join(self.data_base_path, "state")
        
        # 创建必要的目录结构
        self._ensure_directories()
        
        # 核心组件
        self.chroma_client = None
        self.vector_store = None
        self.vector_index = None
        self.bm25_model = None
        self.nodes_data = None
        self.embed_model = None
        
        # 状态
        self._loaded = False
        self._last_version = None
        
    def _ensure_directories(self):
        """确保所需的目录结构存在"""
        directories = [
            self.data_base_path,
            self.chroma_db_path,
            self.pickle_path,
            self.state_path
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                # 如果创建失败，记录但不抛出异常，让后续的is_available检查处理
                pass
        
    def is_available(self) -> bool:
        """检查RAG系统是否可用"""
        # 检查必要的文件是否存在
        required_files = [
            self.chroma_db_path,
            os.path.join(self.pickle_path, "bm25_model.pkl"),
            os.path.join(self.pickle_path, "nodes_data.pkl")
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                return False
                
        return True
        
    def _check_version_change(self) -> bool:
        """检查数据版本是否有变化"""
        version_file = os.path.join(self.state_path, "data_version.json")
        
        try:
            if os.path.exists(version_file):
                with open(version_file, 'r', encoding='utf-8') as f:
                    version_data = json.load(f)
                    current_version = version_data.get('version')
                    
                    if current_version != self._last_version:
                        self._last_version = current_version
                        return True
        except:
            pass
            
        return False
        
    def _lazy_load(self):
        """延迟加载数据，只在需要时加载"""
        if self._loaded and not self._check_version_change():
            return
            
        print("🔄 加载RAG数据...")
        
        try:
            # 1. 设置嵌入模型
            self._setup_embedding_model()
            
            # 2. 加载向量存储
            self._load_vector_store()
            
            # 3. 加载BM25模型
            self._load_bm25_model()
            
            self._loaded = True
            print("✅ RAG数据加载完成")
            
        except Exception as e:
            print(f"❌ RAG数据加载失败: {e}")
            raise
            
    def _setup_embedding_model(self):
        """设置嵌入模型"""
        self.embed_model = OllamaEmbedding(
            model_name="bge-m3:latest",
            base_url="http://localhost:11434",
            embed_batch_size=8,
            request_timeout=30,
        )
        Settings.embed_model = self.embed_model
        
    def _load_vector_store(self):
        """加载向量存储和索引"""
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        collection_name = "rag_demo"
        chroma_collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        self.vector_index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
    def _load_bm25_model(self):
        """加载BM25模型和节点数据"""
        bm25_file = os.path.join(self.pickle_path, "bm25_model.pkl")
        nodes_file = os.path.join(self.pickle_path, "nodes_data.pkl")
        
        with open(bm25_file, 'rb') as f:
            self.bm25_model = pickle.load(f)
            
        with open(nodes_file, 'rb') as f:
            nodes_data = pickle.load(f)
            
        # 重建节点对象
        self.nodes_data = []
        for node_data in nodes_data:
            node = TextNode(
                text=node_data['text'],
                metadata=node_data['metadata'],
                id_=node_data['node_id']
            )
            self.nodes_data.append(node)
            
    def search(self, query: str, top_k: int = 9) -> List[Dict[str, Any]]:
        """执行混合检索 - 与rag_demo.py的检索逻辑保持一致"""
        if not self.is_available():
            return []
            
        self._lazy_load()
        
        if not self._loaded:
            return []
            
        try:
            # 创建混合检索器 - 参考rag_demo.py的create_hybrid_retriever方法
            retriever = self._create_hybrid_retriever(similarity_top_k=top_k * 2)
            
            # 执行混合检索
            nodes = retriever.retrieve(query)
            
            # 格式化结果
            formatted_results = []
            for node in nodes:
                # 获取文件路径
                file_path = node.metadata.get('file_path', '未知路径')
                filename = os.path.basename(file_path) if file_path != '未知路径' else '未知文件'
                
                # 获取分数：优先从metadata获取混合分数，否则使用节点原有分数
                score = 0.0
                if hasattr(node, 'metadata') and node.metadata and 'hybrid_score' in node.metadata:
                    score = node.metadata['hybrid_score']
                    retrieval_source = node.metadata.get('retrieval_source', 'hybrid')
                else:
                    score = getattr(node, 'score', 0.0)
                    retrieval_source = 'vector'
                
                # 确保score是浮点数
                try:
                    score = float(score) if score is not None else 0.0
                    if score < 0:
                        score = 0.0
                except (ValueError, TypeError):
                    score = 0.0
                
                formatted_results.append({
                    'rank': len(formatted_results) + 1,
                    'content': node.text.strip(),
                    'score': score,
                    'file_path': file_path,
                    'filename': filename,
                    'content_length': len(node.text),
                    'retrieval_source': retrieval_source
                })
            
            # 按相似度得分排序并限制返回数量
            formatted_results.sort(key=lambda x: x['score'], reverse=True)
            formatted_results = formatted_results[:top_k]
            
            # 重新分配排名
            for i, result in enumerate(formatted_results):
                result['rank'] = i + 1
                
            return formatted_results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
            
    def _bm25_search(self, query: str, top_k: int) -> List[tuple]:
        """BM25检索"""
        if not self.bm25_model or not self.nodes_data:
            return []
            
        # 对查询进行分词
        tokenized_query = list(jieba.cut(query, cut_all=False))
        
        # BM25检索
        scores = self.bm25_model.get_scores(tokenized_query)
        
        # 获取top_k结果
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.nodes_data):
                node = self.nodes_data[idx]
                score = float(scores[idx])
                results.append((node, score, 'bm25'))
                
        return results
        
    def _fuse_results(self, vector_results, bm25_results, top_k: int) -> List[tuple]:
        """融合向量检索和BM25检索结果"""
        all_results = {}
        
        # 处理向量检索结果
        for i, node in enumerate(vector_results):
            node_key = hash(node.text[:200])
            rrf_score = 1.0 / (i + 1)  # 倒数排名融合
            all_results[node_key] = {
                'node': node,
                'score': rrf_score,
                'source': 'vector'
            }
            
        # 处理BM25检索结果
        for i, (node, bm25_score, source) in enumerate(bm25_results):
            node_key = hash(node.text[:200])
            rrf_score = 1.0 / (i + 1)
            
            if node_key in all_results:
                # 合并分数
                all_results[node_key]['score'] += rrf_score
                all_results[node_key]['source'] = 'hybrid'
            else:
                all_results[node_key] = {
                    'node': node,
                    'score': rrf_score,
                    'source': 'bm25'
                }
                
        # 按融合分数排序
        sorted_results = sorted(all_results.values(), 
                              key=lambda x: x['score'], reverse=True)
        
        # 转换为返回格式
        final_results = []
        for result in sorted_results[:top_k]:
            final_results.append((
                result['node'], 
                result['score'], 
                result['source']
            ))
            
        return final_results
            
    def _create_hybrid_retriever(self, similarity_top_k: int = 10):
        """创建混合检索器 - 与rag_demo.py的create_hybrid_retriever方法保持一致"""
        try:
            
            # 1. 创建向量检索器
            vector_retriever = VectorIndexRetriever(
                index=self.vector_index,
                similarity_top_k=similarity_top_k,
            )
            
            # 2. 创建BM25检索器（需要所有节点）
            if not self.nodes_data:
                print("⚠️ 没有节点可用于BM25检索，仅使用向量检索")
                return vector_retriever
                
            # 准备BM25语料库（中文分词）
            tokenized_corpus = []
            for node in self.nodes_data:
                # 中文分词
                tokens = list(jieba.cut(node.text, cut_all=False))
                tokenized_corpus.append(tokens)
            
            # 创建BM25模型
            bm25 = BM25Okapi(tokenized_corpus)
            
            # 创建自定义BM25检索器
            class CustomBM25Retriever:
                def __init__(self, bm25_model, nodes, similarity_top_k=10):
                    self.bm25 = bm25_model
                    self.nodes = nodes
                    self.similarity_top_k = similarity_top_k
                
                def retrieve(self, query_str):
                    # 对查询进行分词
                    tokenized_query = list(jieba.cut(query_str, cut_all=False))
                    
                    # BM25检索
                    scores = self.bm25.get_scores(tokenized_query)
                    
                    # 获取top_k结果
                    top_indices = scores.argsort()[-self.similarity_top_k:][::-1]
                    
                    # 返回节点和分数信息
                    results = []
                    for idx in top_indices:
                        if idx < len(self.nodes):
                            node = self.nodes[idx]
                            results.append((node, float(scores[idx])))
                    
                    return results
            
            bm25_retriever = CustomBM25Retriever(bm25, self.nodes_data, similarity_top_k)
            
            # 3. 创建简化的混合检索器
            class SimpleHybridRetriever:
                def __init__(self, vector_retriever, bm25_retriever, similarity_top_k=10):
                    self.vector_retriever = vector_retriever
                    self.bm25_retriever = bm25_retriever
                    self.similarity_top_k = similarity_top_k
                
                def retrieve(self, query_str):
                    # 执行向量检索
                    vector_results = self.vector_retriever.retrieve(query_str)
                    
                    # 执行BM25检索（返回(node, score)元组）
                    bm25_results = self.bm25_retriever.retrieve(query_str)
                    
                    # 合并结果（简单RRF融合）
                    all_results = {}
                    
                    # 处理向量检索结果 - 使用文本内容作为唯一标识
                    for i, node in enumerate(vector_results):
                        # 使用文本内容的哈希值作为唯一标识
                        node_key = hash(node.text[:200])
                        rrf_score = 1.0 / (i + 1)  # 倒数排名融合
                        all_results[node_key] = {
                            'node': node,
                            'score': rrf_score,
                            'source': 'vector'
                        }
                    
                    # 处理BM25检索结果（处理(node, score)元组格式）
                    for i, item in enumerate(bm25_results):
                        if isinstance(item, tuple):
                            node, bm25_score = item
                        else:
                            node = item
                            bm25_score = 0.0
                            
                        # 使用相同的节点标识方法
                        node_key = hash(node.text[:200])
                        rrf_score = 1.0 / (i + 1)
                        
                        if node_key in all_results:
                            # 合并分数 - 这里才是真正的混合检索
                            all_results[node_key]['score'] += rrf_score
                            all_results[node_key]['source'] = 'hybrid'
                        else:
                            all_results[node_key] = {
                                'node': node,
                                'score': rrf_score,
                                'source': 'bm25'
                            }
                    
                    # 按融合分数排序
                    sorted_results = sorted(all_results.values(), 
                                          key=lambda x: x['score'], reverse=True)
                    
                    # 返回带有临时分数信息的节点列表
                    final_results = []
                    for result in sorted_results[:self.similarity_top_k]:
                        node = result['node']
                        # 创建一个临时的结果对象，包含节点和分数信息
                        class NodeWithScore:
                            def __init__(self, node, score, source):
                                self.text = node.text
                                self.metadata = node.metadata.copy() if node.metadata else {}
                                self.metadata['hybrid_score'] = score
                                self.metadata['retrieval_source'] = source
                        
                        result_node = NodeWithScore(node, result['score'], result['source'])
                        final_results.append(result_node)
                    
                    return final_results
            
            hybrid_retriever = SimpleHybridRetriever(vector_retriever, bm25_retriever, similarity_top_k)
            return hybrid_retriever
            
        except Exception as e:
            print(f"⚠️ 混合检索器创建失败，使用向量检索: {str(e)}")
            return VectorIndexRetriever(index=self.vector_index, similarity_top_k=similarity_top_k)