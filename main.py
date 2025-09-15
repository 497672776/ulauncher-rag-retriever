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


class RAGExtension(Extension):
    """RAG检索插件主类"""

    def __init__(self):
        super().__init__()
        self.subscribe(KeywordQueryEvent, KeywordQueryEventListener())
        self.subscribe(ItemEnterEvent, ItemEnterEventListener())
        
        # 搜索脚本路径（使用静默版本）
        self.search_script = os.path.join(os.path.dirname(__file__), "search_script_silent.py")
    
    @property
    def venv_python(self):
        """获取虚拟环境Python路径（从preferences读取）"""
        venv_path = self.preferences.get('venv_python', 'venv/bin/python')
        # 如果是相对路径，则相对于插件目录
        if not os.path.isabs(venv_path):
            venv_path = os.path.join(os.path.dirname(__file__), venv_path)
        return venv_path
        

class KeywordQueryEventListener(EventListener):
    """关键词查询事件监听器"""

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
            
            # 构建命令
            cmd = [
                extension.venv_python,
                extension.search_script,
                query,
                str(max_results)
            ]
            
            # 执行搜索
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(extension.search_script),
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # 处理错误
                try:
                    error_data = json.loads(result.stdout)
                    error_msg = error_data.get('error', '未知错误')
                except:
                    error_msg = f"搜索脚本退出码: {result.returncode}"
                
                return RenderResultListAction([
                    ExtensionResultItem(
                        icon='images/icon.png',
                        name='❌ 搜索失败',
                        description=error_msg,
                        on_enter=HideWindowAction()
                    )
                ])
            
            # 解析搜索结果
            try:
                search_data = json.loads(result.stdout)
            except json.JSONDecodeError:
                return RenderResultListAction([
                    ExtensionResultItem(
                        icon='images/icon.png',
                        name='❌ 结果解析失败',
                        description='搜索脚本返回格式错误',
                        on_enter=HideWindowAction()
                    )
                ])
            
            if not search_data.get('success'):
                error_msg = search_data.get('error', '搜索失败')
                return RenderResultListAction([
                    ExtensionResultItem(
                        icon='images/icon.png',
                        name='⚠️ RAG系统错误',
                        description=error_msg,
                        on_enter=HideWindowAction()
                    )
                ])
            
            # 格式化结果
            results = search_data.get('results', [])
            if results:
                items = self._format_results(results, query)
                return RenderResultListAction(items)
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

    def _format_results(self, results, query):
        """格式化搜索结果"""
        items = []
        
        for i, result in enumerate(results):
            filename = result.get('filename', '未知文件')
            content = result.get('content', '')
            score = result.get('score', 0)
            file_path = result.get('file_path', '')
            source = result.get('retrieval_source', 'hybrid')
            
            # 获取图标
            file_icon = self._get_file_icon(filename)
            source_icon = self._get_source_icon(source)
            
            # 格式化内容片段
            snippet = self._format_snippet(content, query, 100)
            
            # 构建结果项
            name = f"{source_icon} {filename}"
            description = f"分数: {score:.3f} | {snippet}"
            
            items.append(ExtensionResultItem(
                icon=file_icon,
                name=name,
                description=description,
                on_enter=OpenAction(file_path) if file_path else HideWindowAction()
            ))
        
        return items
    
    def _get_file_icon(self, filename):
        """获取文件图标"""
        ext = Path(filename).suffix.lower()
        icon_map = {
            '.pdf': 'images/pdf-icon.png',
            '.txt': 'images/txt-icon.png', 
            '.md': 'images/md-icon.png',
            '.doc': 'images/doc-icon.png',
            '.docx': 'images/doc-icon.png'
        }
        return icon_map.get(ext, 'images/icon.png')
    
    def _get_source_icon(self, source):
        """获取检索来源图标"""
        source_map = {
            'hybrid': '🔥',
            'vector': '🎯', 
            'bm25': '🔤'
        }
        return source_map.get(source, '🔍')
    
    def _format_snippet(self, content, query, max_length):
        """格式化内容片段"""
        if len(content) <= max_length:
            return content
        
        # 尝试找到查询关键词周围的内容
        query_lower = query.lower()
        content_lower = content.lower()
        
        if query_lower in content_lower:
            # 找到关键词位置，取周围内容
            pos = content_lower.find(query_lower)
            start = max(0, pos - max_length // 2)
            end = min(len(content), start + max_length)
            snippet = content[start:end]
            
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
                
            return snippet
        else:
            # 没找到关键词，取开头
            return content[:max_length] + "..."


class ItemEnterEventListener(EventListener):
    """项目点击事件监听器"""

    def on_event(self, event, extension):
        """处理项目点击事件"""
        # 简化处理，主要是打开文件
        return HideWindowAction()


if __name__ == '__main__':
    RAGExtension().run()