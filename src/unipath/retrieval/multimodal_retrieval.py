#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
import h5py
import numpy as np
import torch
from typing import List, Dict, Optional
import argparse
import sys
import os

from .vocab_query_matcher import PathologyVocabMatcher


class MultiModalRetriever:
    
    def __init__(
        self,
        h5_file: str,
        vocab_file: str,
        inverted_index_file: str,
        image_dir: Optional[str] = None,
        device: Optional[str] = None,
        load_conch: bool = True
    ):
        """
        初始化检索器
        
        Args:
            h5_file: H5数据文件路径
            vocab_file: 词表JSONL文件路径
            inverted_index_file: 倒排索引JSON文件路径
            image_dir: 图像目录路径（可选）
            device: 设备 ('cuda' 或 'cpu'，默认自动选择）
            load_conch: 是否加载CONCH模型（默认True）
        """
        print("Initializing MultiModal Retriever...")
        
        self.h5_file = h5_file
        self.image_dir = image_dir
        
        # 设备选择
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"\nUsing device: {self.device}")
        
        # 1. 加载CONCH模型（用于文本编码）
        if load_conch:
            self._load_conch_model()
        else:
            self.conch_model = None
            self.conch_tokenizer = None
            print("\n⚠️ Skipping CONCH model loading")
        
        # 2. 加载H5数据
        self._load_h5_data()
        
        # 3. 加载词表匹配器
        print(f"\nLoading vocabulary: {vocab_file}")
        self.vocab_matcher = PathologyVocabMatcher(vocab_file)
        
        # 4. 加载倒排索引
        print(f"\nLoading inverted index: {inverted_index_file}")
        with open(inverted_index_file, 'r', encoding='utf-8') as f:
            self.inverted_index = json.load(f)
        print(f"✓ Inverted index loaded: {len(self.inverted_index)} keywords")
        
        print("Initialization complete!")
        
    
    def _load_conch_model(self):
        """加载CONCH模型用于文本编码"""
        print(f"\nLoading CONCH model...")
        
        try:
            # 尝试导入CONCH
            try:
                from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
            except ImportError:
                conch_paths = [
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'conch')
                ]
                
                found = False
                for path in conch_paths:
                    if os.path.exists(path):
                        if path not in sys.path:
                            sys.path.insert(0, path)
                        found = True
                        break
                
                if not found:
                    raise ImportError(f"CONCH module path not found. Tried paths: {conch_paths}")
                
                from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
            
            # 加载模型
            model, preprocess = create_model_from_pretrained(
                'conch_ViT-B-16', 
                checkpoint_path='path/to/conch'
            )
            tokenizer = get_tokenizer()
            
            model = model.to(self.device)
            model.eval()
            
            self.conch_model = model
            self.conch_tokenizer = tokenizer
            self.conch_tokenize = tokenize
            
            print(f"✓ CONCH model loaded successfully (device: {self.device})")
            
        except Exception as e:
            print(f"✗ CONCH model loading failed: {e}")
            print(f"   Using random vectors as placeholders")
            self.conch_model = None
            self.conch_tokenizer = None
            self.conch_tokenize = None
    
    def _load_h5_data(self):
        print(f"\nLoading H5 file: {self.h5_file}")
        
        with h5py.File(self.h5_file, 'r') as f:
            print(f"H5 file contains keys: {list(f.keys())}")
            
            # 加载必要的数据
            self.sample_keys = f['sample_key'][:]
            if isinstance(self.sample_keys[0], bytes):
                self.sample_keys = [sk.decode('utf-8') for sk in self.sample_keys]
            
            # 加载特征向量（直接转为 GPU Tensor）
            self.conch_text_features = torch.from_numpy(f['conch_text_features'][:]).to(self.device, dtype=torch.float32)
            self.conch_visual_features = torch.from_numpy(f['conch_visual_features'][:]).to(self.device, dtype=torch.float32)
            self.uni2h_features = torch.from_numpy(f['uni2h_features'][:]).to(self.device, dtype=torch.float32)

            # 一次性 L2 归一化检索库特征（避免每步重复归一化）
            self.conch_text_features = torch.nn.functional.normalize(self.conch_text_features, p=2, dim=1, eps=1e-8)
            self.conch_visual_features = torch.nn.functional.normalize(self.conch_visual_features, p=2, dim=1, eps=1e-8)
            
            # 加载描述（用于显示）
            if 'gemini_description' in f:
                self.descriptions = f['gemini_description'][:]
                if isinstance(self.descriptions[0], bytes):
                    self.descriptions = [d.decode('utf-8', errors='ignore') for d in self.descriptions]
            else:
                self.descriptions = None
            
        
        self.n_samples = len(self.sample_keys)
        
        print(f"\n✓ H5 data loaded:")
        print(f"  - Number of samples: {self.n_samples}")
        # 打印张量形状
        print(f"  - conch_text_features shape: {list(self.conch_text_features.shape)} on {self.conch_text_features.device}")
        print(f"  - conch_visual_features shape: {list(self.conch_visual_features.shape)} on {self.conch_visual_features.device}")
        print(f"  - uni2h_features shape: {list(self.uni2h_features.shape)} on {self.uni2h_features.device}")
    
    def encode_text_with_conch(self, texts):
        """
        使用CONCH模型对文本进行编码（支持批量）
        
        Args:
            texts: 输入文本（str）或文本列表（List[str]）
            
        Returns:
            文本特征向量 (D,) 或特征矩阵 (N, D)
        """
        # 统一转为列表处理
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # 如果CONCH模型加载成功，使用真实编码
        if self.conch_model is not None and self.conch_tokenizer is not None:
            try:
                with torch.no_grad():
                    # 对文本进行tokenize（批量）
                    text_tokens = self.conch_tokenize(
                        texts=texts, 
                        tokenizer=self.conch_tokenizer
                    ).to(self.device)
                    
                    # 编码文本（批量），直接返回 GPU Tensor
                    text_features = self.conch_model.encode_text(
                        text_tokens,
                        normalize=True
                    )  # (N, D) on self.device, already L2-normalized

                    # 如果是单个文本，返回一维向量
                    return text_features[0] if is_single else text_features
                    
            except Exception as e:
                print(f"⚠️  CONCH encoding failed: {e}, using random vectors")
        
        # 如果模型未加载或编码失败，使用随机向量作为占位
        print(f"⚠️ Warning: Using random vectors for text encoding")
        
        # 返回与conch_text_features相同维度的随机向量（GPU Tensor）
        # 假设特征已加载为 numpy，可从其 shape 获取维度
        feature_dim = int(self.conch_text_features.shape[1]) if isinstance(self.conch_text_features, np.ndarray) else int(self.conch_text_features.size(1))
        random_vectors = torch.randn(len(texts), feature_dim, device=self.device, dtype=torch.float32)
        norms = torch.norm(random_vectors, dim=1, keepdim=True) + 1e-8
        random_vectors = random_vectors / norms
        return random_vectors[0] if is_single else random_vectors
    
    def text_vector_retrieval(
        self, 
        prompts, 
        top_k: int = 4
    ):
        """
        文本向量召回（支持批量）
        
        Args:
            prompts: 用户查询文本（str）或文本列表（List[str]）
            top_k: 返回top-k结果
            
        Returns:
            单个查询: [(sample_idx, score), ...] 列表
            批量查询: [[(sample_idx, score), ...], ...] 列表的列表
        """
        # 统一处理单个和批量输入
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        # 1. 对prompts进行批量编码（返回 GPU Tensor）
        query_features = self.encode_text_with_conch(prompts)  # (N, D) or (D,)
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)  # (1, D)

        # 2. 批量计算相似度矩阵 (N_queries, N_samples)
        # 说明：库特征在初始化时已 L2 归一化，这里仅需与已归一化的查询做内积
        similarities = torch.matmul(query_features, self.conch_text_features.t())  # (N, M)

        # 4. 对每个查询获取top-k（GPU 上完成）
        _, top_idx = torch.topk(similarities, k=top_k, dim=1, largest=True, sorted=True)

        # 直接返回索引（避免 Python for 循环），保持为 Python 列表
        batch_results = top_idx.tolist()
        
        # 如果是单个查询，返回单个结果
        return batch_results[0] if is_single else batch_results
    
    def visual_vector_retrieval(
        self, 
        prompts, 
        top_k: int = 4
    ):
        """
        视觉向量召回（跨模态，支持批量）
        
        Args:
            prompts: 用户查询文本（str）或文本列表（List[str]）
            top_k: 返回top-k结果
            
        Returns:
            单个查询: [(sample_idx, score), ...] 列表
            批量查询: [[(sample_idx, score), ...], ...] 列表的列表
        """
        # 统一处理单个和批量输入
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        # 1. 对prompts进行批量编码（返回 GPU Tensor）
        query_features = self.encode_text_with_conch(prompts)  # (N, D) or (D,)
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)  # (1, D)

        # 2. 批量计算相似度矩阵 (N_queries, N_samples)
        # 说明：库特征在初始化时已 L2 归一化
        similarities = torch.matmul(query_features, self.conch_visual_features.t())  # (N, M)

        # 4. 对每个查询获取top-k（GPU）
        _, top_idx = torch.topk(similarities, k=top_k, dim=1, largest=True, sorted=True)

        # 直接返回索引（避免 Python for 循环），保持为 Python 列表
        batch_results = top_idx.tolist()
        
        # 如果是单个查询，返回单个结果
        return batch_results[0] if is_single else batch_results
    
    def keyword_retrieval(
        self, 
        prompts,
        max_keywords: int = 4,
        samples_per_keyword: int = 2,
    ):
        """
        关键词选择性召回策略（支持批量）
        
        Args:
            prompts: 用户查询文本（str）或文本列表（List[str]）
            max_keywords: 最多选择的关键词数量（默认4）
            samples_per_keyword: 每个关键词返回的样本数（默认2）
            verbose: 是否打印详细信息
            
        Returns:
            单个查询: sample_idx 列表（去重）
            批量查询: [sample_idx列表, ...] 列表的列表
        """
        # 统一处理单个和批量输入
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        batch_results = []
        
        for prompt in prompts:
            
            # 1. 使用vocab_matcher提取关键词
            matches = self.vocab_matcher.extract(prompt)
            
            if not matches:
                batch_results.append([])
                continue
            
            # 2. 统计每个关键词在倒排索引中的文档数量
            keyword_info = []
            for match in matches:
                keyword = match['phrase'].lower()
                if keyword in self.inverted_index:
                    sample_indices = self.inverted_index[keyword]
                    keyword_info.append({
                        'keyword': keyword,
                        'match': match,
                        'doc_count': len(sample_indices),
                        'sample_indices': sample_indices
                    })
            
            if not keyword_info:
                batch_results.append([])
                continue
            
            # 3. 如果关键词数量超过max_keywords，选择倒排列表最少的关键词
            if len(keyword_info) > max_keywords:
                # 按doc_count升序排序，选择最少的max_keywords个
                keyword_info = sorted(keyword_info, key=lambda x: x['doc_count'])[:max_keywords]
            
            # 4. 从每个选中的关键词中召回samples_per_keyword个样本
            all_samples = []
            for kw in keyword_info:
                # 简单取前samples_per_keyword个
                samples = random.sample(kw['sample_indices'], min(samples_per_keyword, len(kw['sample_indices'])))
                all_samples.extend(samples)
                print(f"Keyword: {kw['keyword']}, Samples: {samples}")
            
            # 5. 去重并返回
            unique_samples = list(set(all_samples))
            batch_results.append(unique_samples)
        
        # 如果是单个查询，返回单个结果
        return batch_results[0] if is_single else batch_results
    
    def retrieve(
        self,
        prompts,
        top_k: int = 4,
        enable_text_recall: bool = True,
        enable_visual_recall: bool = True,
        enable_keyword_recall: bool = True,
        max_keywords: int = 4,
        samples_per_keyword: int = 2,
        max_samples: int = 16
    ):
        """
        多模态检索主函数（并集策略，支持批量）
        
        Args:
            prompts: 用户查询文本（str）或文本列表（List[str]）
            top_k: 文本和视觉召回的数量（默认4）
            enable_text_recall: 是否启用文本召回
            enable_visual_recall: 是否启用视觉召回
            enable_keyword_recall: 是否启用关键词召回
            max_keywords: 关键词召回最多选择的关键词数量（默认4）
            samples_per_keyword: 每个关键词返回的样本数（默认2）
            max_samples: 每个查询返回的最大样本数，不足则padding（默认16）
            verbose: 是否显示详细进度
            
        Returns:
            features: torch.Tensor，shape为[b, max_samples, d]，b为batch数量，d为uni2h_features维度
            mask: torch.Tensor，shape为[b, max_samples]，True表示真实样本，False表示padding
        """
        # 统一处理单个和批量输入
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        # 1. 文本向量召回（批量）
        text_results_batch = None
        if enable_text_recall:
            text_results_batch = self.text_vector_retrieval(prompts, top_k=top_k)
            print(f"Text Results: {text_results_batch}")
        
        # 2. 视觉向量召回（批量）
        visual_results_batch = None
        if enable_visual_recall:
            visual_results_batch = self.visual_vector_retrieval(prompts, top_k=top_k)
            print(f"Visual Results: {visual_results_batch}")
        
        # 3. 关键词召回（批量）
        keyword_results_batch = None
        if enable_keyword_recall:
            keyword_results_batch = self.keyword_retrieval(
                prompts,
                max_keywords=max_keywords,
                samples_per_keyword=samples_per_keyword,
            )
        
        # 获取uni2h_features的维度
        feature_dim = self.uni2h_features.shape[1]
        
        # 初始化返回的tensor和mask
        batch_features_list = []
        batch_mask_list = []
        
        for query_idx, prompt in enumerate(prompts):
            
            union = set()
            
            # 收集各路召回的结果集
            if text_results_batch is not None:
                # 现在 text_results_batch[query_idx] 为 List[int]
                text_set = set(text_results_batch[query_idx])
                union = union | text_set
            
            if visual_results_batch is not None:
                # 现在 visual_results_batch[query_idx] 为 List[int]
                visual_set = set(visual_results_batch[query_idx])
                union = union | visual_set
            
            if keyword_results_batch is not None:
                keyword_set = set(keyword_results_batch[query_idx])
                union = union | keyword_set
            
            if not union:
                # 全部padding（放在正确设备，mask 用 0/1 表示，0 为 padding）
                padded_features = torch.zeros(max_samples, feature_dim, device=self.device, dtype=self.uni2h_features.dtype)
                mask = torch.zeros(max_samples, dtype=torch.long, device=self.device)
                batch_features_list.append(padded_features)
                batch_mask_list.append(mask)
                continue
        
            
            # 获取并集序号对应的 uni2h_features，最多取max_samples个
            # 优化采样，避免多次转换且排序更高效
            union_indices = list(union)
            if len(union_indices) > max_samples:
                union_indices = random.sample(union_indices, max_samples)
            union_indices.sort()
            actual_num = len(union_indices)
            features_tensor = self.uni2h_features[union_indices]  # 已在正确设备上的张量
            
            # 创建mask（1 表示真实样本，0 表示 padding）
            mask = torch.zeros(max_samples, dtype=torch.long, device=self.device)
            if actual_num > 0:
                mask[:actual_num] = 1
            
            # Padding到max_samples
            if actual_num < max_samples:
                padding_size = max_samples - actual_num
                padding = torch.zeros(padding_size, feature_dim, dtype=features_tensor.dtype, device=features_tensor.device)
                features_tensor = torch.cat([features_tensor, padding], dim=0)
            
            batch_features_list.append(features_tensor)
            batch_mask_list.append(mask)
        
        # 堆叠成[b, max_samples, d]和[b, max_samples]
        batch_features = torch.stack(batch_features_list, dim=0)  # [b, max_samples, d]
        batch_mask = torch.stack(batch_mask_list, dim=0)  # [b, max_samples], int 0/1
     
        return batch_features, batch_mask
    
    def print_results(self, results: List[Dict], verbose: bool = True):
        """
        打印检索结果
        
        Args:
            results: 检索结果列表
            verbose: 是否显示详细信息
        """
        if not results:
            print("No retrieval results found")
            return
        
        print(f"\n{'='*80}")
        print(f"Retrieval Results (Top-{len(results)})")
        print(f"{'='*80}\n")
        
        for result in results:
            print(f"【{result['rank']}】 Sample: {result['sample_key']}")
            if 'score' in result:
                print(f"    Score: {result['score']:.4f}")
            
            if verbose:
                if 'wsi_id' in result:
                    print(f"   WSI ID: {result['wsi_id']}")
                if 'cluster_label' in result:
                    print(f"    Cluster Label: {result['cluster_label']}")
                if 'description' in result:
                    print(f"    Description: {result['description']}")
                if 'image_path' in result:
                    print(f"    Image Path: {result['image_path']}")
            
            print("")
    
    def print_batch_results(
        self, 
        batch_results: Dict[str, List[Dict]], 
        verbose: bool = True,
        max_results_per_query: int = 5
    ):
        """
        打印批量检索结果
        
        Args:
            batch_results: 批量检索结果字典
            verbose: 是否显示详细信息
            max_results_per_query: 每个查询最多显示多少个结果
        """
        if not batch_results:
            print("No retrieval results found")
            return
        
        print(f"\n{'='*80}")
        print(f"Batch Retrieval Results (Total {len(batch_results)} queries)")
        print(f"{'='*80}")
        
        for query_idx, (prompt, results) in enumerate(batch_results.items(), 1):
            print(f"\n{'─'*80}")
            print(f"Query [{query_idx}/{len(batch_results)}]:")
            print(f"  {prompt}")
            print(f"  Results count: {len(results)}")
            print(f"{'─'*80}")
            
            if not results:
                print("  ⚠️ No results found\n")
                continue
            
            # 只显示前几个结果
            display_results = results[:max_results_per_query]
            
            for result in display_results:
                print(f"\n  【{result['rank']}】 {result['sample_key']}")
                if 'score' in result:
                    print(f"        Score: {result['score']:.4f}")
                
                if verbose and 'description' in result:
                    desc = result['description'][:100] + "..." if len(result['description']) > 100 else result['description']
                    print(f"        Description: {desc}")
            
            if len(results) > max_results_per_query:
                print(f"\n  ... and {len(results) - max_results_per_query} more results")
            
            print("")
        
        print(f"{'='*80}")
        print(f"✅ Batch retrieval results displayed")
        print(f"{'='*80}\n")