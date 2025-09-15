#!/usr/bin/env python3
"""
RAG Demo - 简化的检索增强生成演示
功能：从data_test目录加载文档，实现查询检索，返回相似文本块和路径
"""

import os
import time
import pickle
import threading
from typing import List, Dict, Any
from pathlib import Path

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

# ChromaDB imports  
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


class RAGDemo:
    """简化的RAG演示系统"""
    
    def __init__(self, knowledge_base_path: str = "data_test", model_name: str = "bge-m3:latest"):
        """
        初始化RAG系统
        
        Args:
            knowledge_base_path: 知识库文档目录路径
            model_name: 嵌入模型名称
        """
        self.knowledge_base_path = knowledge_base_path
        self.model_name = model_name
        self.chroma_db_path = "./demo_chroma_db/chroma_db"
        self.bm25_cache_path = "./demo_chroma_db/models/bm25_model.pkl"
        self.nodes_cache_path = "./demo_chroma_db/models/nodes_data.pkl"
        
        # 确保目录存在
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        os.makedirs(self.chroma_db_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.bm25_cache_path), exist_ok=True)
        
        print(f"🚀 初始化RAG Demo")
        print(f"📁 知识库路径: {self.knowledge_base_path}")
        print(f"🤖 嵌入模型: {self.model_name}")
        print(f"🗃️ 向量数据库: {self.chroma_db_path}")
        
        # 初始化组件
        self._setup_embedding_model()
        self._setup_vector_store()
        self.index = None
        self.documents_metadata = []
        self.all_nodes = []  # 存储所有节点用于BM25检索
        
        # 文件监控相关
        self._monitoring = False
        self._monitor_thread = None
        self._known_files = set()  # 已知文件集合
        self._supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.doc'}
        
    def _setup_embedding_model(self):
        """设置嵌入模型"""
        print("🔧 配置嵌入模型...")
        self.embed_model = OllamaEmbedding(
            model_name=self.model_name,
            base_url="http://localhost:11434",
            embed_batch_size=8,
            request_timeout=30,
        )
        
        # 设置全局配置
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 300  # 稍大的块以获得更多上下文
        Settings.chunk_overlap = 50
        
    def _setup_vector_store(self):
        """设置ChromaDB向量存储"""
        print("🗃️ 配置ChromaDB...")
        print(f"🔍 数据库路径: {os.path.abspath(self.chroma_db_path)}")
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        
        # 获取或创建集合
        collection_name = "rag_demo"
        self.chroma_collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        # 创建向量存储
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
    
    def _check_document_exists(self, file_path: str) -> bool:
        """
        检查文档是否已在向量库中存在
        
        Args:
            file_path: 文件路径
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            # 查询ChromaDB中是否已存在该文档路径
            results = self.chroma_collection.get(
                where={"file_path": file_path},
                limit=1
            )
            
            exists = len(results['ids']) > 0
            if exists:
                filename = os.path.basename(file_path)
                print(f"📋 文档已存在于向量库: {filename}")
            
            return exists
            
        except Exception as e:
            print(f"⚠️ 检查文档存在性失败 {file_path}: {e}")
            return False
    
    def _get_existing_documents(self) -> List[str]:
        """
        获取已存在于向量库中的所有文档路径
        
        Returns:
            已存在文档的文件路径列表
        """
        try:
            # 查询所有文档的元数据
            results = self.chroma_collection.get(
                include=['metadatas']
            )
            
            existing_files = []
            for metadata in results['metadatas']:
                if metadata and 'file_path' in metadata:
                    existing_files.append(metadata['file_path'])
            
            return list(set(existing_files))  # 去重
            
        except Exception as e:
            print(f"⚠️ 获取已存在文档列表失败: {e}")
            return []
    
    def _save_bm25_model(self, bm25_model, nodes: List) -> None:
        """
        保存BM25模型和节点数据到磁盘
        
        Args:
            bm25_model: BM25模型对象
            nodes: 节点列表
        """
        try:
            # 确保缓存目录存在
            cache_dir = os.path.dirname(self.bm25_cache_path)
            os.makedirs(cache_dir, exist_ok=True)
            
            # 保存BM25模型
            with open(self.bm25_cache_path, 'wb') as f:
                pickle.dump(bm25_model, f)
            
            # 保存节点数据
            nodes_data = []
            for node in nodes:
                nodes_data.append({
                    'text': node.text,
                    'metadata': node.metadata,
                    'node_id': node.node_id
                })
            
            with open(self.nodes_cache_path, 'wb') as f:
                pickle.dump(nodes_data, f)
            
            print(f"💾 BM25模型和节点数据已保存到磁盘")
            
        except Exception as e:
            print(f"⚠️ 保存BM25模型失败: {e}")
    
    def _load_bm25_model(self):
        """
        从磁盘加载BM25模型和节点数据
        
        Returns:
            tuple: (bm25_model, nodes) 或 (None, None) 如果加载失败
        """
        try:
            # 检查文件是否存在
            if not (os.path.exists(self.bm25_cache_path) and os.path.exists(self.nodes_cache_path)):
                return None, None
            
            # 加载BM25模型
            with open(self.bm25_cache_path, 'rb') as f:
                bm25_model = pickle.load(f)
            
            # 加载节点数据
            with open(self.nodes_cache_path, 'rb') as f:
                nodes_data = pickle.load(f)
            
            # 重建节点对象
            from llama_index.core.schema import TextNode
            nodes = []
            for node_data in nodes_data:
                node = TextNode(
                    text=node_data['text'],
                    metadata=node_data['metadata'],
                    id_=node_data['node_id']
                )
                nodes.append(node)
            
            print(f"📂 从磁盘加载BM25模型和 {len(nodes)} 个节点")
            return bm25_model, nodes
            
        except Exception as e:
            print(f"⚠️ 加载BM25模型失败: {e}")
            return None, None
    
    def _update_bm25_model(self, existing_bm25, existing_nodes: List, new_nodes: List):
        """
        更新BM25模型，添加新文档
        
        Args:
            existing_bm25: 现有BM25模型
            existing_nodes: 现有节点列表
            new_nodes: 新节点列表
            
        Returns:
            tuple: (updated_bm25_model, all_nodes)
        """
        try:
            from rank_bm25 import BM25Okapi
            import jieba
            
            # 合并所有节点
            all_nodes = existing_nodes + new_nodes
            
            # 重新构建BM25模型（包含所有文档）
            print("🔄 更新BM25模型...")
            tokenized_corpus = []
            for node in all_nodes:
                tokens = list(jieba.cut(node.text))
                tokenized_corpus.append(tokens)
            
            bm25_model = BM25Okapi(tokenized_corpus)
            print(f"✅ BM25模型已更新，包含 {len(all_nodes)} 个文档")
            
            return bm25_model, all_nodes
            
        except Exception as e:
            print(f"⚠️ 更新BM25模型失败: {e}")
            return None, existing_nodes
    
    def create_hybrid_retriever(self, similarity_top_k: int = 10, num_queries: int = 4):
        """创建混合检索器：向量检索 + BM25关键词检索"""
        try:
            # 1. 创建向量检索器
            vector_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k,
            )
            
            # 2. 创建BM25检索器（需要所有节点）
            if not self.all_nodes:
                print("⚠️ 没有节点可用于BM25检索，仅使用向量检索")
                return vector_retriever
            
            from rank_bm25 import BM25Okapi
            import jieba
            
            # 准备BM25语料库（中文分词）
            tokenized_corpus = []
            for node in self.all_nodes:
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
                    
                    # 返回节点和分数信息，不直接修改节点
                    results = []
                    for idx in top_indices:
                        if idx < len(self.nodes):
                            node = self.nodes[idx]
                            # 创建节点副本或使用tuple存储分数信息
                            results.append((node, float(scores[idx])))
                    
                    return results
            
            bm25_retriever = CustomBM25Retriever(bm25, self.all_nodes, similarity_top_k)
            
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
                        # 使用文本内容的哈希值作为唯一标识，更可靠
                        node_key = hash(node.text[:200])  # 使用前200字符的哈希避免完全相同的长文本
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
                            print(f"🔥 发现混合检索节点: {node.text[:50]}...")  # 调试信息
                        else:
                            all_results[node_key] = {
                                'node': node,
                                'score': rrf_score,
                                'source': 'bm25'
                            }
                    
                    # 按融合分数排序
                    sorted_results = sorted(all_results.values(), 
                                          key=lambda x: x['score'], reverse=True)
                    
                    print(f"🔍 检索统计: Vector={len(vector_results)}, BM25={len(bm25_results)}, 总计={len(all_results)}")
                    
                    # 统计各类型结果数量
                    source_count = {'vector': 0, 'bm25': 0, 'hybrid': 0}
                    for result in sorted_results:
                        source_count[result['source']] += 1
                    print(f"📊 结果分布: {source_count}")
                    
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
            
            print(f"🔍 创建混合检索器: 向量检索 + BM25关键词检索")
            return hybrid_retriever
            
        except Exception as e:
            print(f"⚠️ 混合检索器创建失败，使用向量检索: {str(e)}")
            return VectorIndexRetriever(index=self.index, similarity_top_k=similarity_top_k)
        
    def load_documents(self) -> int:
        """
        从knowledge_base_path加载所有文档（带去重检查）
        
        Returns:
            加载的文档数量
        """
        print(f"📄 加载文档从: {self.knowledge_base_path}")
        start_time = time.time()
        
        # 使用SimpleDirectoryReader加载文档 - 逐个文件加载以避免目录模式问题
        try:
            documents = []
            supported_exts = [".txt", ".md", ".pdf", ".docx", ".doc"]
            
            # 遍历目录中的所有文件
            for root, _, files in os.walk(self.knowledge_base_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported_exts):
                        file_path = os.path.join(root, file)
                        try:
                            # 逐个文件加载
                            reader = SimpleDirectoryReader(input_files=[file_path])
                            file_docs = reader.load_data()
                            documents.extend(file_docs)
                            print(f"✅ 加载文件: {file} ({len(file_docs)} 个文档)")
                        except Exception as e:
                            print(f"❌ 加载文件失败 {file}: {e}")
            
            if not documents:
                print("❌ 没有找到支持的文档文件")
                return 0
                
            print(f"✅ 找到 {len(documents)} 个文档")
            
            # 检查哪些文档是新的
            print("🔍 检查文档是否已存在于向量库...")
            new_documents = []
            skipped_count = 0
            
            for doc in documents:
                file_path = doc.metadata.get('file_path', '未知路径')
                filename = os.path.basename(file_path)
                
                # 检查文档是否已存在
                if self._check_document_exists(file_path):
                    skipped_count += 1
                    continue
                else:
                    new_documents.append(doc)
                    print(f"   📄 新文档: {filename} ({len(doc.text)} 字符)")
            
            print(f"📊 统计: 新文档 {len(new_documents)} 个，跳过已存在 {skipped_count} 个")
            
            if not new_documents:
                print("✅ 所有文档都已存在于向量库中，无需重新处理")
                # 仍需要加载现有索引用于查询
                try:
                    self.index = VectorStoreIndex.from_vector_store(
                        self.vector_store,
                        storage_context=self.storage_context,
                        embed_model=self.embed_model
                    )
                    print("🔍 加载现有向量索引")
                    
                    # 尝试从磁盘加载BM25模型和节点
                    print("📝 加载BM25模型和节点...")
                    bm25_model, cached_nodes = self._load_bm25_model()
                    
                    if bm25_model is not None and cached_nodes is not None:
                        # 使用缓存的数据
                        self.all_nodes.extend(cached_nodes)
                        print(f"📂 从缓存加载 {len(cached_nodes)} 个节点用于BM25检索")
                    else:
                        # 缓存不存在，重新生成
                        print("📝 重新生成节点用于BM25检索...")
                        parser = SimpleNodeParser.from_defaults(
                            chunk_size=300,
                            chunk_overlap=50
                        )
                        all_nodes = parser.get_nodes_from_documents(documents)
                        self.all_nodes.extend(all_nodes)
                        print(f"📝 生成 {len(all_nodes)} 个节点用于BM25检索")
                        
                        # 创建并保存BM25模型
                        from rank_bm25 import BM25Okapi
                        import jieba
                        
                        tokenized_corpus = []
                        for node in all_nodes:
                            tokens = list(jieba.cut(node.text))
                            tokenized_corpus.append(tokens)
                        
                        bm25_model = BM25Okapi(tokenized_corpus)
                        self._save_bm25_model(bm25_model, all_nodes)
                    
                except Exception as e:
                    print(f"⚠️ 无法加载现有索引: {e}")
                
                return len(documents)
            
            # 保存所有文档元数据（包括已存在的）
            self.documents_metadata = []
            for doc in documents:
                file_path = doc.metadata.get('file_path', '未知路径')
                filename = os.path.basename(file_path)
                self.documents_metadata.append({
                    'filename': filename,
                    'file_path': file_path,
                    'content_length': len(doc.text)
                })
            
            # 只处理新文档
            print("✂️ 切分新文档...")
            parser = SimpleNodeParser.from_defaults(
                chunk_size=300,
                chunk_overlap=50
            )
            new_nodes = parser.get_nodes_from_documents(new_documents)
            print(f"📊 新生成 {len(new_nodes)} 个文本块")
            
            # 新节点已包含必要的元数据信息
            
            # 处理BM25模型的增量更新
            print("🔄 处理BM25模型...")
            existing_bm25, existing_nodes = self._load_bm25_model()
            
            if existing_bm25 is not None and existing_nodes is not None:
                # 更新现有BM25模型
                updated_bm25, all_nodes = self._update_bm25_model(existing_bm25, existing_nodes, new_nodes)
                if updated_bm25 is not None:
                    self.all_nodes = all_nodes
                    self._save_bm25_model(updated_bm25, all_nodes)
                    print(f"📝 BM25模型已更新，总共 {len(all_nodes)} 个节点")
                else:
                    # 更新失败，使用现有节点加新节点
                    self.all_nodes = existing_nodes + new_nodes
                    print(f"⚠️ BM25模型更新失败，使用现有模型，总共 {len(self.all_nodes)} 个节点")
            else:
                # 创建新的BM25模型
                print("🆕 创建新BM25模型...")
                all_nodes = new_nodes.copy()
                
                # 如果有其他已存在的文档，也要包含进来
                if skipped_count > 0:
                    print("📝 重新生成所有文档的节点...")
                    parser_all = SimpleNodeParser.from_defaults(
                        chunk_size=300,
                        chunk_overlap=50
                    )
                    all_document_nodes = parser_all.get_nodes_from_documents(documents)
                    all_nodes = all_document_nodes
                
                from rank_bm25 import BM25Okapi
                import jieba
                
                tokenized_corpus = []
                for node in all_nodes:
                    tokens = list(jieba.cut(node.text))
                    tokenized_corpus.append(tokens)
                
                bm25_model = BM25Okapi(tokenized_corpus)
                self._save_bm25_model(bm25_model, all_nodes)
                self.all_nodes = all_nodes
                print(f"📝 新BM25模型已创建，包含 {len(all_nodes)} 个节点")
            
            # 创建或更新向量索引
            print("🔄 更新向量索引...")
            embed_start = time.time()
            
            try:
                # 尝试从现有存储加载
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    storage_context=self.storage_context,
                    embed_model=self.embed_model
                )
                print("🔍 从现有ChromaDB加载索引")
                
                # 只添加新节点
                if new_nodes:
                    self.index.insert_nodes(new_nodes)
                    print(f"➕ 添加了 {len(new_nodes)} 个新文本块")
                
            except:
                # 创建新索引
                self.index = VectorStoreIndex(
                    new_nodes,
                    storage_context=self.storage_context, 
                    embed_model=self.embed_model
                )
                print("🆕 创建新的向量索引")
            
            embed_time = time.time() - embed_start
            total_time = time.time() - start_time
            
            print(f"✅ 向量化完成: {embed_time:.2f}秒")
            print(f"🎉 文档加载总耗时: {total_time:.2f}秒")
            
            return len(documents)
            
        except Exception as e:
            print(f"❌ 加载文档失败: {e}")
            return 0
    
    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        查询知识库
        
        Args:
            question: 查询问题
            top_k: 返回结果数量
            
        Returns:
            包含相似文本块和路径的结果列表
        """
        if not self.index:
            print("❌ 请先加载文档")
            return []
        
        print(f"🔍 查询: {question}")
        start_time = time.time()
        
        try:
            # 创建混合检索器
            retriever = self.create_hybrid_retriever(similarity_top_k=top_k * 2)  # 获取更多候选
            
            # 执行混合检索
            nodes = retriever.retrieve(question)
            
            # 处理结果，正确获取相似度得分
            results = []
            for node in nodes:
                # 获取文件路径
                file_path = node.metadata.get('file_path', '未知路径')
                filename = os.path.basename(file_path) if file_path != '未知路径' else '未知文件'
                
                # 获取分数：优先从metadata获取混合分数，否则使用节点原有分数
                score = 0.0
                if hasattr(node, 'metadata') and node.metadata and 'hybrid_score' in node.metadata:
                    # 使用混合检索分数
                    score = node.metadata['hybrid_score']
                    retrieval_source = node.metadata.get('retrieval_source', 'hybrid')
                else:
                    # 使用原始分数
                    score = getattr(node, 'score', 0.0)
                    retrieval_source = 'vector'
                
                # 确保score是浮点数并且在合理范围内
                try:
                    score = float(score) if score is not None else 0.0
                    if score < 0:
                        score = 0.0
                except (ValueError, TypeError):
                    score = 0.0
                
                results.append({
                    'rank': len(results) + 1,
                    'content': node.text.strip(),
                    'score': score,
                    'file_path': file_path,
                    'filename': filename,
                    'content_length': len(node.text),
                    'retrieval_source': retrieval_source  # 添加检索源信息
                })
            
            # 按相似度得分排序（降序）并限制返回数量
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:top_k]  # 限制返回数量
            
            # 重新分配排名
            for i, result in enumerate(results):
                result['rank'] = i + 1
            
            search_time = time.time() - start_time
            
            print(f"✅ 混合检索完成: {search_time:.3f}秒，找到 {len(results)} 个结果")
            
            return results
            
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return []
    
    def print_results(self, results: List[Dict[str, Any]], show_content: bool = True):
        """
        格式化打印查询结果
        
        Args:
            results: 查询结果
            show_content: 是否显示完整内容
        """
        if not results:
            print("🚫 没有找到相关结果")
            return
        
        print(f"\n📋 查询结果 (共 {len(results)} 条):")
        print("=" * 70)
        
        for result in results:
            # 获取检索来源和对应的图标
            source = result.get('retrieval_source', 'unknown')
            source_icons = {
                'vector': '🎯 向量检索',
                'bm25': '🔤 关键词检索', 
                'hybrid': '🔥 混合检索',
                'unknown': '❓ 未知来源'
            }
            source_display = source_icons.get(source, f'❓ {source}')
            
            print(f"🏷️ 排名: {result['rank']}")
            print(f"📄 文件: {result['filename']}")
            print(f"📁 路径: {result['file_path']}")
            print(f"⭐ 相似度: {result['score']:.4f}")
            print(f"🔍 检索来源: {source_display}")
            print(f"📊 长度: {result['content_length']} 字符")
            
            if show_content:
                content = result['content']
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"📝 内容: {content}")
            
            print("-" * 70)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        doc_count = len(self.documents_metadata)
        total_chars = sum(doc['content_length'] for doc in self.documents_metadata)
        
        # 估算文本块数量（基于平均块大小300字符）
        estimated_chunks = total_chars // 300 if total_chars > 0 else 0
        
        return {
            'documents_loaded': doc_count,
            'total_characters': total_chars,
            'estimated_chunks': estimated_chunks,
            'knowledge_base_path': self.knowledge_base_path,
            'model_name': self.model_name,
            'chunk_size': Settings.chunk_size,
            'chunk_overlap': Settings.chunk_overlap
        }
    
    def performance_test(self, test_queries: List[str] = None) -> Dict[str, float]:
        """
        性能测试
        
        Args:
            test_queries: 测试查询列表
            
        Returns:
            性能统计结果
        """
        if not test_queries:
            test_queries = [
                "什么是人工智能？",
                "机器学习的基本概念",
                "深度学习神经网络",
                "自然语言处理技术"
            ]
        
        print("⚡ 开始性能测试...")
        
        times = []
        for i, query in enumerate(test_queries, 1):
            print(f"🧪 测试查询 {i}: {query}")
            start_time = time.time()
            results = self.query(query, top_k=3)
            query_time = time.time() - start_time
            times.append(query_time)
            print(f"   ⏱️ 用时: {query_time:.3f}秒，结果数: {len(results)}")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📈 性能测试结果:")
        print(f"   - 平均查询时间: {avg_time:.3f}秒")
        print(f"   - 最快查询: {min_time:.3f}秒") 
        print(f"   - 最慢查询: {max_time:.3f}秒")
        
        return {
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'total_queries': len(test_queries)
        }
    
    def _scan_directory(self) -> set:
        """
        扫描知识库目录，返回所有支持的文件路径集合
        
        Returns:
            文件路径集合
        """
        files = set()
        try:
            for root, _, filenames in os.walk(self.knowledge_base_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext in self._supported_extensions:
                        # 使用绝对路径确保唯一性
                        files.add(os.path.abspath(file_path))
        except Exception as e:
            print(f"⚠️ 扫描目录失败: {e}")
        return files
    
    def _process_new_files(self, new_files: set) -> None:
        """
        处理新发现的文件
        
        Args:
            new_files: 新文件路径集合
        """
        if not new_files:
            return
            
        print(f"\n🔍 发现 {len(new_files)} 个新文件:")
        for file_path in new_files:
            filename = os.path.basename(file_path)
            print(f"   📄 {filename}")
        
        try:
            # 重新加载文档（会自动检测并只处理新文件）
            print("🔄 开始处理新文件...")
            old_doc_count = len(self.documents_metadata)  # 记录处理前的文档数
            new_count = self.load_documents()
            
            # 正确的逻辑：检查是否成功添加了所有新文件
            expected_count = old_doc_count + len(new_files)
            if new_count >= expected_count:
                print(f"✅ 新文件处理完成，知识库已更新 ({old_doc_count} → {new_count})")
            else:
                print(f"⚠️ 部分文件可能处理失败 (预期: {expected_count}, 实际: {new_count})")
                
            # 更新已知文件集合
            self._known_files.update(new_files)
            
        except Exception as e:
            print(f"❌ 处理新文件失败: {e}")
    
    def _monitor_directory(self) -> None:
        """
        监控目录变化的后台线程函数
        """
        print(f"📡 开始监控知识库目录: {self.knowledge_base_path}")
        print("⏰ 监控间隔: 5秒")
        
        # 初始扫描
        self._known_files = self._scan_directory()
        print(f"📊 初始文件数量: {len(self._known_files)}")
        
        while self._monitoring:
            try:
                time.sleep(5)  # 5秒检测间隔
                
                if not self._monitoring:
                    break
                
                # 扫描当前文件
                current_files = self._scan_directory()
                
                # 找出新文件
                new_files = current_files - self._known_files
                
                if new_files:
                    self._process_new_files(new_files)
                
                # 检测删除的文件（可选功能）
                deleted_files = self._known_files - current_files
                if deleted_files:
                    print(f"🗑️ 检测到 {len(deleted_files)} 个文件被删除")
                    # 注意：这里不处理删除，因为向量数据库中的数据保持不变
                    # 如需处理删除，可以添加相应逻辑
                    self._known_files = current_files
                
            except Exception as e:
                print(f"⚠️ 监控过程中出错: {e}")
                time.sleep(5)  # 出错后等待5秒继续
    
    def start_monitoring(self) -> None:
        """
        启动文件监控
        """
        if self._monitoring:
            print("⚠️ 文件监控已在运行中")
            return
        
        print("🚀 启动文件监控...")
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_directory,
            daemon=True  # 设为守护线程，主程序退出时自动结束
        )
        self._monitor_thread.start()
        print("✅ 文件监控已启动")
    
    def stop_monitoring(self) -> None:
        """
        停止文件监控
        """
        if not self._monitoring:
            print("⚠️ 文件监控未运行")
            return
        
        print("🛑 停止文件监控...")
        self._monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)  # 等待最多10秒
        
        print("✅ 文件监控已停止")
    
    def is_monitoring(self) -> bool:
        """
        检查是否正在监控
        
        Returns:
            True if monitoring is active, False otherwise
        """
        return self._monitoring


def main():
    """主函数演示"""
    print("🎯 RAG Demo 演示系统")
    print("=" * 50)
    
    # 创建RAG系统
    rag = RAGDemo(knowledge_base_path="data_test")
    
    # 加载文档
    doc_count = rag.load_documents()
    if doc_count == 0:
        print("❌ 没有加载任何文档，请在data_test目录中放入文档文件")
        return
    
    # 显示系统统计
    stats = rag.get_stats()
    print(f"\n📊 系统统计:")
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # 交互式查询
    print(f"\n🔍 交互式查询 (输入 'quit' 退出, 'test' 运行性能测试):")
    
    while True:
        try:
            query = input("\n💬 请输入查询问题: ").strip()
            
            if query.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
                
            if query.lower() == 'test':
                rag.performance_test()
                continue
                
            if not query:
                continue
                
            # 执行查询
            results = rag.query(query, top_k=3)
            rag.print_results(results, show_content=True)
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 查询出错: {e}")


if __name__ == "__main__":
    main()