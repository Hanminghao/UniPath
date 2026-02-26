#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class PathologyVocabMatcher:
    """病理学词表匹配器"""
    
    def __init__(self, vocab_file: str):
        """
        初始化匹配器
        
        Args:
            vocab_file: 词表文件路径（支持JSONL和txt格式）
        """
        self.vocab_file = vocab_file
        self.vocab_list = self._load_vocab()
        # 按短语长度降序排序，优先匹配长短语
        self.vocab_list.sort(key=lambda x: len(x['phrase']), reverse=True)
        print(f"✓ 加载词表: {len(self.vocab_list)} 个词条")
    
    def _load_vocab(self) -> List[Dict]:
        """
        加载词表文件，支持两种格式：
        1. JSONL格式：每行一个JSON对象，包含phrase等字段
        2. TXT格式：每行一个词条（纯文本）
        """
        vocab_list = []
        file_ext = Path(self.vocab_file).suffix.lower()
        
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 先尝试按JSONL格式解析
                try:
                    vocab_item = json.loads(line)
                    vocab_list.append(vocab_item)
                except json.JSONDecodeError:
                    # 如果JSON解析失败，当作纯文本处理
                    vocab_list.append({
                        'phrase': line,
                        'freq': 1,  # 默认频率
                        'category': 'unknown'  # 默认类别
                    })
        
        return vocab_list
    
    def extract(
        self, 
        query: str, 
        case_sensitive: bool = False,
        categories: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        从查询中提取词表内容
        
        Args:
            query: 用户查询文本
            case_sensitive: 是否区分大小写
            categories: 限定类别列表（可选），如 ['morphology', 'tissue']
        
        Returns:
            匹配结果列表，每项包含词表信息和匹配位置
        """
        if not case_sensitive:
            query_search = query.lower()
        else:
            query_search = query
        
        matches = []
        matched_spans = []  # 记录已匹配的位置，避免重叠
        
        for vocab_item in self.vocab_list:
            # 应用过滤条件
            if categories and vocab_item.get('category') not in categories:
                continue
            
            phrase = vocab_item['phrase']
            search_phrase = phrase.lower() if not case_sensitive else phrase
            
            # 查找所有出现位置
            start = 0
            while True:
                pos = query_search.find(search_phrase, start)
                if pos == -1:
                    break
                
                end = pos + len(search_phrase)
                
                # 检查是否与已匹配位置重叠
                overlap = any(
                    not (end <= s or pos >= e) 
                    for s, e in matched_spans
                )
                
                if not overlap:
                    matches.append({
                        **vocab_item,
                        'matched_text': query[pos:end],
                        'start': pos,
                        'end': end
                    })
                    matched_spans.append((pos, end))
                
                start = pos + 1
        
        # 按位置排序
        matches.sort(key=lambda x: x['start'])
        return matches
    
    def extract_with_highlight(
        self, 
        query: str, 
        case_sensitive: bool = False,
        highlight_format: str = "【{text}】"
    ) -> Tuple[str, List[Dict]]:
        """
        提取词表内容并高亮显示
        
        Args:
            query: 用户查询文本
            case_sensitive: 是否区分大小写
            highlight_format: 高亮格式，{text}会被替换为匹配文本
        
        Returns:
            (高亮后的文本, 匹配列表)
        """
        matches = self.extract(query, case_sensitive)
        
        if not matches:
            return query, []
        
        # 构建高亮文本
        highlighted = ""
        last_pos = 0
        
        for match in matches:
            highlighted += query[last_pos:match['start']]
            highlighted += highlight_format.format(text=match['matched_text'])
            last_pos = match['end']
        
        highlighted += query[last_pos:]
        
        return highlighted, matches
    
    def get_stats(self, matches: List[Dict]) -> Dict:
        """
        获取匹配统计信息
        
        Args:
            matches: 匹配结果列表
        
        Returns:
            统计信息字典
        """
        if not matches:
            return {
                'total': 0,
                'by_category': {},
                'by_relevance': {}
            }
        
        stats = {
            'total': len(matches),
            'by_category': {},
            'by_relevance': {}
        }
        
        for match in matches:
            cat = match.get('category', 'unknown')
            rel = match.get('relevance', 'unknown')
            stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
            stats['by_relevance'][rel] = stats['by_relevance'].get(rel, 0) + 1
        
        return stats
    
    def print_matches(self, matches: List[Dict], verbose: bool = True):
        """
        打印匹配结果
        
        Args:
            matches: 匹配结果列表
            verbose: 是否显示详细信息
        """
        if not matches:
            print("未匹配到任何词条")
            return
        
        print(f"\n匹配到 {len(matches)} 个词条:\n")
        
        for i, match in enumerate(matches, 1):
            print(f"{i}. 【{match['phrase']}】")
            if verbose:
                print(f"   类别: {match.get('category', 'N/A')}")
                print(f"   相关性: {match.get('relevance', 'N/A')}")
                print(f"   位置: {match['start']}-{match['end']}")
                if 'reason' in match:
                    print(f"   说明: {match['reason']}")
            print()
