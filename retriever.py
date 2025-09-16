"""
轻量级检索器 - 只负责查询，不负责建立索引
"""

import os
import pickle
from typing import List, Dict, Any

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


class DocumentRetriever:
    """轻量级文档检索器 - 负责文档搜索和相关性检索"""
    
    def __init__(self, document_database_path: str):
        self.document_database_root_path = os.path.expanduser(document_database_path)
        
        # 文档数据存储路径
        self.vector_database_path = os.path.join(self.document_database_root_path, "chroma_db")
        self.keyword_models_path = os.path.join(self.document_database_root_path, "keyword_model")
        
        # 核心检索组件
        self.chromadb_client = None
        self.vector_document_store = None
        self.vector_search_index = None
        self.bm25_keyword_search_model = None
        self.document_nodes_data = None
        self.text_embedding_model = None
        
        # 检索器状态管理
        self._is_data_loaded = False
        
    def is_document_search_available(self) -> bool:
        """检查文档检索系统是否可用且数据完整"""
        # 检查文档检索所需的必要文件是否存在
        essential_data_files = [
            self.vector_database_path,
            os.path.join(self.keyword_models_path, "bm25_model.pkl"),
            os.path.join(self.keyword_models_path, "nodes_data.pkl")
        ]
        
        for essential_file_path in essential_data_files:
            if not os.path.exists(essential_file_path):
                return False
                
        return True
        
        
    def _load_document_search_data_lazily(self):
        """延迟加载文档检索数据，只在需要时加载以提高性能"""
        if self._is_data_loaded:
            return
            
        print("🔄 初始化文档检索数据...")
        
        try:
            # 1. 初始化文本嵌入模型
            self._initialize_text_embedding_model()
            
            # 2. 加载向量文档存储
            self._load_vector_document_store()
            
            # 3. 加载BM25关键词检索模型
            self._load_bm25_keyword_search_model()
            
            self._is_data_loaded = True
            print("✅ 文档检索数据初始化完成")
            
        except Exception as initialization_error:
            print(f"❌ 文档检索数据加载失败: {initialization_error}")
            raise
            
    def _initialize_text_embedding_model(self):
        """初始化和设置文本嵌入向量化模型"""
        self.text_embedding_model = OllamaEmbedding(
            model_name="bge-m3:latest",
            base_url="http://localhost:11434",
            embed_batch_size=8,
            request_timeout=30,
        )
        Settings.embed_model = self.text_embedding_model
        
    def _load_vector_document_store(self):
        """加载向量文档存储和检索索引"""
        self.chromadb_client = chromadb.PersistentClient(path=self.vector_database_path)
        document_collection_name = "documents"
        chroma_document_collection = self.chromadb_client.get_or_create_collection(name=document_collection_name)
        
        self.vector_document_store = ChromaVectorStore(chroma_collection=chroma_document_collection)
        vector_storage_context = StorageContext.from_defaults(vector_store=self.vector_document_store)
        
        self.vector_search_index = VectorStoreIndex.from_vector_store(
            self.vector_document_store,
            storage_context=vector_storage_context,
            embed_model=self.text_embedding_model
        )
        
    def _load_bm25_keyword_search_model(self):
        """加载BM25关键词检索模型和文档节点数据"""
        bm25_model_file_path = os.path.join(self.keyword_models_path, "bm25_model.pkl")
        document_nodes_file_path = os.path.join(self.keyword_models_path, "nodes_data.pkl")
        
        with open(bm25_model_file_path, 'rb') as bm25_file:
            self.bm25_keyword_search_model = pickle.load(bm25_file)
            
        with open(document_nodes_file_path, 'rb') as nodes_file:
            serialized_nodes_data = pickle.load(nodes_file)
            
        # 重新构建文档节点对象
        self.document_nodes_data = []
        for node_data in serialized_nodes_data:
            document_node = TextNode(
                text=node_data['text'],
                metadata=node_data['metadata'],
                id_=node_data['node_id']
            )
            self.document_nodes_data.append(document_node)
            
    def search_documents(self, search_query: str, maximum_results: int = 9) -> List[Dict[str, Any]]:
        """执行混合文档检索 - 结合向量相似度和BM25关键词检索"""
        if not self.is_document_search_available():
            return []
            
        self._load_document_search_data_lazily()
        
        if not self._is_data_loaded:
            return []
            
        try:
            # 创建混合文档检索器 - 结合向量和BM25检索
            hybrid_document_retriever = self._create_hybrid_document_retriever(similarity_top_k=maximum_results * 2)
            
            # 执行混合文档检索
            retrieved_document_nodes = hybrid_document_retriever.retrieve(search_query)
            
            # 格式化检索结果
            formatted_search_results = []
            for document_node in retrieved_document_nodes:
                # 获取文档文件路径信息
                document_file_path = document_node.metadata.get('file_path', '未知路径')
                document_filename = os.path.basename(document_file_path) if document_file_path != '未知路径' else '未知文件'
                
                # 获取文档相关性分数和检索来源
                relevance_score = 0.0
                if hasattr(document_node, 'metadata') and document_node.metadata and 'hybrid_score' in document_node.metadata:
                    relevance_score = document_node.metadata['hybrid_score']
                    search_method = document_node.metadata.get('retrieval_source', 'hybrid')
                else:
                    relevance_score = getattr(document_node, 'score', 0.0)
                    search_method = 'vector'
                
                # 确保相关性分数是有效数值
                try:
                    relevance_score = float(relevance_score) if relevance_score is not None else 0.0
                    if relevance_score < 0:
                        relevance_score = 0.0
                except (ValueError, TypeError):
                    relevance_score = 0.0
                
                formatted_search_results.append({
                    'rank': len(formatted_search_results) + 1,
                    'content': document_node.text.strip(),
                    'score': relevance_score,
                    'file_path': document_file_path,
                    'filename': document_filename,
                    'content_length': len(document_node.text),
                    'retrieval_source': search_method
                })
            
            # 按相关性分数排序并限制返回数量
            formatted_search_results.sort(key=lambda x: x['score'], reverse=True)
            final_search_results = formatted_search_results[:maximum_results]
            
            # 重新计算排名
            for result_index, search_result in enumerate(final_search_results):
                search_result['rank'] = result_index + 1
                
            return final_search_results
            
        except Exception as search_error:
            print(f"❌ 文档检索失败: {search_error}")
            return []
            
            
    def _create_hybrid_document_retriever(self, similarity_top_k: int = 10):
        """创建混合文档检索器 - 结合向量相似度和BM25关键词检索"""
        try:
            
            # 1. 创建向量相似度检索器
            vector_similarity_retriever = VectorIndexRetriever(
                index=self.vector_search_index,
                similarity_top_k=similarity_top_k,
            )
            
            # 2. 创建BM25关键词检索器（需要所有文档节点）
            if not self.document_nodes_data:
                print("⚠️ 没有文档节点可用于BM25关键词检索，仅使用向量检索")
                return vector_similarity_retriever
                
            # 准备BM25关键词检索语料库（中文分词）
            tokenized_document_corpus = []
            for document_node in self.document_nodes_data:
                # 对文档节点进行中文分词
                document_tokens = list(jieba.cut(document_node.text, cut_all=False))
                tokenized_document_corpus.append(document_tokens)
            
            # 创建BM25关键词检索模型
            bm25_keyword_model = BM25Okapi(tokenized_document_corpus)
            
            # 创建自定义BM25关键词检索器
            class CustomBM25KeywordRetriever:
                def __init__(self, bm25_keyword_model, document_nodes, similarity_top_k=10):
                    self.bm25_keyword_search_model = bm25_keyword_model
                    self.document_nodes = document_nodes
                    self.maximum_similarity_results = similarity_top_k
                
                def retrieve(self, search_query_text):
                    # 对搜索查询进行中文分词
                    tokenized_search_query = list(jieba.cut(search_query_text, cut_all=False))
                    
                    # 执行BM25关键词检索
                    keyword_relevance_scores = self.bm25_keyword_search_model.get_scores(tokenized_search_query)
                    
                    # 获取最相关的文档索引
                    most_relevant_indices = keyword_relevance_scores.argsort()[-self.maximum_similarity_results:][::-1]
                    
                    # 返回文档节点和相关性分数
                    keyword_search_results = []
                    for document_index in most_relevant_indices:
                        if document_index < len(self.document_nodes):
                            relevant_document_node = self.document_nodes[document_index]
                            keyword_search_results.append((relevant_document_node, float(keyword_relevance_scores[document_index])))
                    
                    return keyword_search_results
            
            bm25_keyword_retriever = CustomBM25KeywordRetriever(bm25_keyword_model, self.document_nodes_data, similarity_top_k)
            
            # 3. 创建简化的混合文档检索器
            class SimpleHybridDocumentRetriever:
                def __init__(self, vector_similarity_retriever, bm25_keyword_retriever, similarity_top_k=10):
                    self.vector_similarity_retriever = vector_similarity_retriever
                    self.bm25_keyword_retriever = bm25_keyword_retriever
                    self.maximum_results = similarity_top_k
                
                def retrieve(self, search_query_text):
                    # 执行向量相似度检索
                    vector_similarity_results = self.vector_similarity_retriever.retrieve(search_query_text)
                    
                    # 执行BM25关键词检索（返回(node, score)元组）
                    bm25_keyword_results = self.bm25_keyword_retriever.retrieve(search_query_text)
                    
                    # 合并两种检索结果（使用RRF融合算法）
                    merged_hybrid_results = {}
                    
                    # 处理向量相似度检索结果 - 使用文档内容作为唯一标识
                    for result_rank, vector_node in enumerate(vector_similarity_results):
                        # 使用文档内容的哈希值作为唯一标识
                        document_key = hash(vector_node.text[:200])
                        reciprocal_rank_score = 1.0 / (result_rank + 1)  # 倒数排名融合
                        merged_hybrid_results[document_key] = {
                            'node': vector_node,
                            'score': reciprocal_rank_score,
                            'source': 'vector'
                        }
                    
                    # 处理BM25关键词检索结果（处理(node, score)元组格式）
                    for result_rank, bm25_item in enumerate(bm25_keyword_results):
                        if isinstance(bm25_item, tuple):
                            bm25_node, _ = bm25_item
                        else:
                            bm25_node = bm25_item
                            
                        # 使用相同的文档节点标识方法
                        document_key = hash(bm25_node.text[:200])
                        reciprocal_rank_score = 1.0 / (result_rank + 1)
                        
                        if document_key in merged_hybrid_results:
                            # 合并两种检索方法的分数 - 混合检索的核心
                            merged_hybrid_results[document_key]['score'] += reciprocal_rank_score
                            merged_hybrid_results[document_key]['source'] = 'hybrid'
                        else:
                            merged_hybrid_results[document_key] = {
                                'node': bm25_node,
                                'score': reciprocal_rank_score,
                                'source': 'bm25'
                            }
                    
                    # 按融合相关性分数排序
                    sorted_hybrid_results = sorted(merged_hybrid_results.values(), 
                                          key=lambda x: x['score'], reverse=True)
                    
                    # 返回带有混合分数信息的文档节点列表
                    final_hybrid_results = []
                    for hybrid_result in sorted_hybrid_results[:self.maximum_results]:
                        original_document_node = hybrid_result['node']
                        # 创建一个带有相关性分数的文档节点对象
                        class DocumentNodeWithHybridScore:
                            def __init__(self, document_node, hybrid_score, retrieval_source):
                                self.text = document_node.text
                                self.metadata = document_node.metadata.copy() if document_node.metadata else {}
                                self.metadata['hybrid_score'] = hybrid_score
                                self.metadata['retrieval_source'] = retrieval_source
                        
                        enriched_document_node = DocumentNodeWithHybridScore(
                            original_document_node, 
                            hybrid_result['score'], 
                            hybrid_result['source']
                        )
                        final_hybrid_results.append(enriched_document_node)
                    
                    return final_hybrid_results
            
            hybrid_document_retriever = SimpleHybridDocumentRetriever(
                vector_similarity_retriever, 
                bm25_keyword_retriever, 
                similarity_top_k
            )
            return hybrid_document_retriever
            
        except Exception as hybrid_retriever_error:
            print(f"⚠️ 混合文档检索器创建失败，降级使用单纯向量检索: {str(hybrid_retriever_error)}")
            return VectorIndexRetriever(index=self.vector_search_index, similarity_top_k=similarity_top_k)