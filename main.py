#!/usr/bin/env python3
"""
RAG检索插件 - 轻量级版本
通过subprocess调用虚拟环境中的搜索脚本，避免依赖问题
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# 从Ulauncher 5.15.8导入API
from ulauncher.api.client.Extension import Extension
from ulauncher.api.client.EventListener import EventListener
from ulauncher.api.shared.event import KeywordQueryEvent, ItemEnterEvent
from ulauncher.api.shared.item.ExtensionResultItem import ExtensionResultItem
from ulauncher.api.shared.item.ExtensionSmallResultItem import ExtensionSmallResultItem
from ulauncher.api.shared.action.RenderResultListAction import RenderResultListAction
from ulauncher.api.shared.action.HideWindowAction import HideWindowAction
from ulauncher.api.shared.action.OpenAction import OpenAction
from ulauncher.api.shared.action.ExtensionCustomAction import ExtensionCustomAction


class DocumentSearchExtension(Extension):
    """文档搜索插件主类"""

    def __init__(self):
        super().__init__()
        self.subscribe(KeywordQueryEvent, DocumentSearchQueryListener())
        self.subscribe(ItemEnterEvent, SearchResultClickListener())
        
        # 文档搜索脚本路径（使用静默版本）
        self.document_search_script = os.path.join(os.path.dirname(__file__), "search_script_silent.py")
    
    @property
    def virtual_environment_python_path(self):
        """获取虚拟环境Python解释器路径（从preferences读取）"""
        venv_path = self.preferences.get('venv_python', 'venv/bin/python')
        # 如果是相对路径，则相对于插件目录
        if not os.path.isabs(venv_path):
            venv_path = os.path.join(os.path.dirname(__file__), venv_path)
        return venv_path
        

class DocumentSearchQueryListener(EventListener):
    """文档搜索查询事件监听器"""

    def on_event(self, event, extension):
        """处理关键词查询事件"""
        query = event.get_argument() or ""
        
        # 如果没有查询内容，显示提示
        if not query.strip():
            return RenderResultListAction([
                ExtensionResultItem(
                    icon='images/icon.png',
                    name='🔍 RAG文档检索',
                    description='请输入搜索关键词，例如: r 公司介绍',
                    on_enter=HideWindowAction()
                )
            ])
        
        try:
            # 通过subprocess调用搜索脚本
            max_results = int(extension.preferences.get('max_results', '9'))
            
            # 构建文档搜索命令
            search_command = [
                extension.virtual_environment_python_path,
                extension.document_search_script,
                query,
                str(max_results)
            ]
            
            # 执行文档搜索
            search_result = subprocess.run(
                search_command,
                cwd=os.path.dirname(extension.document_search_script),
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if search_result.returncode != 0:
                # 处理搜索错误
                try:
                    error_data = json.loads(search_result.stdout)
                    error_message = error_data.get('error', '未知错误')
                except:
                    error_message = f"搜索脚本退出码: {search_result.returncode}"
                
                return RenderResultListAction([
                    ExtensionResultItem(
                        icon='images/icon.png',
                        name='❌ 搜索失败',
                        description=error_message,
                        on_enter=HideWindowAction()
                    )
                ])
            
            # 解析文档搜索结果
            try:
                search_response_data = json.loads(search_result.stdout)
            except json.JSONDecodeError:
                return RenderResultListAction([
                    ExtensionResultItem(
                        icon='images/icon.png',
                        name='❌ 结果解析失败',
                        description='搜索脚本返回格式错误',
                        on_enter=HideWindowAction()
                    )
                ])
            
            if not search_response_data.get('success'):
                error_message = search_response_data.get('error', '搜索失败')
                return RenderResultListAction([
                    ExtensionResultItem(
                        icon='images/icon.png',
                        name='⚠️ RAG系统错误',
                        description=error_message,
                        on_enter=HideWindowAction()
                    )
                ])
            
            # 格式化文档搜索结果
            document_search_results = search_response_data.get('results', [])
            if document_search_results:
                formatted_result_items = self._format_search_results(document_search_results, query)
                return RenderResultListAction(formatted_result_items)
            else:
                return RenderResultListAction([
                    ExtensionResultItem(
                        icon='images/icon.png',
                        name='🚫 未找到相关结果',
                        description=f'查询 "{query}" 没有找到匹配的文档',
                        on_enter=HideWindowAction()
                    )
                ])
                
        except subprocess.TimeoutExpired:
            return RenderResultListAction([
                ExtensionResultItem(
                    icon='images/icon.png',
                    name='⏰ 搜索超时',
                    description='请稍后重试',
                    on_enter=HideWindowAction()
                )
            ])
        except Exception as e:
            return RenderResultListAction([
                ExtensionResultItem(
                    icon='images/icon.png',
                    name='❌ 系统错误',
                    description=f'错误: {str(e)}',
                    on_enter=HideWindowAction()
                )
            ])

    def _format_search_results(self, search_results, user_query):
        """格式化文档搜索结果为Ulauncher显示项"""
        ulauncher_result_items = []
        
        for result_index, search_result in enumerate(search_results):
            document_filename = search_result.get('filename', '未知文件')
            document_content = search_result.get('content', '')
            relevance_score = search_result.get('score', 0)
            document_file_path = search_result.get('file_path', '')
            retrieval_source = search_result.get('retrieval_source', 'hybrid')
            
            # 获取文件和来源图标
            document_type_icon = self._get_document_type_icon(document_filename)
            retrieval_source_icon = self._get_retrieval_source_icon(retrieval_source)
            
            # 格式化文档内容片段
            content_snippet = self._format_content_snippet(document_content, user_query, 100)
            
            # 构建结果项显示信息
            display_name = f"{retrieval_source_icon} {document_filename}"
            display_description = f"相关度: {relevance_score:.3f} | {content_snippet}"
            
            ulauncher_result_items.append(ExtensionResultItem(
                icon=document_type_icon,
                name=display_name,
                description=display_description,
                on_enter=OpenAction(document_file_path) if document_file_path else HideWindowAction()
            ))
        
        return ulauncher_result_items
    
    def _get_document_type_icon(self, document_filename):
        """根据文件名获取文档类型图标"""
        file_extension = Path(document_filename).suffix.lower()
        file_type_to_icon_mapping = {
            '.pdf': 'images/pdf-icon.png',
            '.txt': 'images/txt-icon.png', 
            '.md': 'images/md-icon.png',
            '.doc': 'images/doc-icon.png',
            '.docx': 'images/doc-icon.png'
        }
        return file_type_to_icon_mapping.get(file_extension, 'images/icon.png')
    
    def _get_retrieval_source_icon(self, retrieval_source):
        """根据检索来源获取对应图标"""
        retrieval_method_to_icon_mapping = {
            'hybrid': '🔥',
            'vector': '🎯', 
            'bm25': '🔤'
        }
        return retrieval_method_to_icon_mapping.get(retrieval_source, '🔍')
    
    def _format_content_snippet(self, document_content, search_query, maximum_snippet_length):
        """格式化文档内容片段，突出显示查询相关内容"""
        if len(document_content) <= maximum_snippet_length:
            return document_content
        
        # 尝试找到搜索关键词周围的内容
        search_query_lowercase = search_query.lower()
        content_lowercase = document_content.lower()
        
        if search_query_lowercase in content_lowercase:
            # 找到关键词位置，提取周围内容
            keyword_position = content_lowercase.find(search_query_lowercase)
            snippet_start_position = max(0, keyword_position - maximum_snippet_length // 2)
            snippet_end_position = min(len(document_content), snippet_start_position + maximum_snippet_length)
            content_snippet = document_content[snippet_start_position:snippet_end_position]
            
            if snippet_start_position > 0:
                content_snippet = "..." + content_snippet
            if snippet_end_position < len(document_content):
                content_snippet = content_snippet + "..."
                
            return content_snippet
        else:
            # 没找到关键词，取文档开头部分
            return document_content[:maximum_snippet_length] + "..."


class SearchResultClickListener(EventListener):
    """搜索结果点击事件监听器"""

    def on_event(self, event, extension):
        """处理项目点击事件"""
        # 简化处理，主要是打开文件
        return HideWindowAction()


if __name__ == '__main__':
    DocumentSearchExtension().run()