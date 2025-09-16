#!/usr/bin/env python3
"""
完全静默的搜索脚本，专门为Ulauncher插件设计
"""

import sys
import json
import os
import io
from retriever import DocumentRetriever

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "缺少查询参数"}))
        sys.exit(1)
    
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    
    # 完全静默执行 - 重定向所有输出
    old_stdout = sys.stdout  
    old_stderr = sys.stderr
    
    try:
        # 重定向stdout和stderr到空设备
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        # 初始化（静默）
        # 使用相对于插件目录的路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'rag-systemd', 'demo_chroma_db')
        document_retriever = DocumentRetriever(data_path)
        
        # 恢复stdout用于输出结果
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        if not document_retriever.is_document_search_available():
            print(json.dumps({
                "error": "RAG数据不可用",
                "message": "请先启动后台服务处理文档"
            }))
            sys.exit(1)
        
        # 再次静默执行搜索
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        search_results = document_retriever.search_documents(query, maximum_results=top_k)
        
        # 恢复输出
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # 格式化文档搜索结果
        formatted_search_results = []
        for search_result in search_results:
            formatted_search_results.append({
                "filename": search_result["filename"],
                "content": search_result["content"],
                "score": search_result["score"],
                "file_path": search_result["file_path"],
                "retrieval_source": search_result.get("retrieval_source", "hybrid")
            })
        
        # 返回文档搜索JSON结果
        print(json.dumps({
            "success": True,
            "results": formatted_search_results,
            "total": len(formatted_search_results)
        }))
        
    except Exception as e:
        # 确保恢复输出
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        print(json.dumps({
            "error": f"搜索失败: {str(e)}",
            "query": query
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()