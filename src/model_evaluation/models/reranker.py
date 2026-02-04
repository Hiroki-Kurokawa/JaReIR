#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reranker models using LLMs and Cross-Encoders.
"""

import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from sentence_transformers import CrossEncoder
import logging
import copy
import torch

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

logging.basicConfig(level=logging.INFO)
class LLMBaseReranker:
 
    def __init__(self, model_path_or_name, corpus, query_dict):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=True, cache_dir="/mnt/LSTA6/bay7/kurokawa/.cache")
        self.model_path_or_name = model_path_or_name

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_or_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir= os.getenv("TRANSFORMERS_CACHE"),
            device_map="auto"
        )
        self.model.eval()
        self.corpus = corpus
        self.query_dict = query_dict
        self.listwise_fail_count = 0
        self.token_id_a = self._get_token_ids("A")
        self.token_id_b = self._get_token_ids("B")
        self.token_id_true  = self._get_token_ids("true")
        self.token_id_false = self._get_token_ids("false")
        self.prp_prompt_template = """
質問と2つの文章（AとB）を与えられます。
どちらの文章が質問に対してより関連性が高いかを判断してください。

出力形式:
- "A" もし文章Aがより関連性が高い場合
- "B" もし文章Bがより関連性が高い場合


質問: {question}
文章A: {passage_a}
文章B: {passage_b}

出力:
"""

    def _get_token_ids(self, base_char):
        id_space = self._single_token_id(f" {base_char}")
        id_no_space = self._single_token_id(base_char)
        return id_space if id_space is not None else id_no_space

    def _single_token_id(self, s: str):
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def reranking(self, ranking_list, top_k = 20):
        """
        ranking list:
        {
         query_id : { doc_id : score, ... },
            ...
        }
        """
        
        res = copy.deepcopy(ranking_list)
        for qid, doc_dict in ranking_list.items():
            print("Processing query:", qid)
            res = self.__prp_sliding_k(doc_dict, qid, top_k, res)
        return res

    def __get_prp_scores(self, qid, psg1_id, psg2_id):
        psg1 = self.corpus[psg1_id]
        psg2 = self.corpus[psg2_id]

        prompt = self.prp_prompt_template.format(
            question=self.query_dict[qid],
            passage_a=psg1,
            passage_b=psg2,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        score_a = next_token_logits[self.token_id_a].item()
        score_b = next_token_logits[self.token_id_b].item()
        return score_a, score_b
    
    def __prp_sliding_k(self, doc_dict, qid, top_k, res, sliding_k = 10):


        sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        passage_ids = [doc_id for doc_id, _ in sorted_docs]
        n = len(passage_ids)
        for _ in range(sliding_k):
            for j in range(n - 1, 0, -1):
                pair_1_doc_id = passage_ids[j]
                pair_2_doc_id = passage_ids[j - 1]
                score_a, score_b = self.__get_prp_scores(qid, pair_1_doc_id, pair_2_doc_id)
                if score_a > score_b: # If the latter is better, swap
                    passage_ids[j], passage_ids[j - 1] = passage_ids[j - 1], passage_ids[j]
        for rank, doc_id in enumerate(passage_ids):
            res[qid][doc_id] = max(doc_dict.values()) + (top_k - rank)
        return res
    
                    

class CEReranker:
    def __init__(self, model_path_or_name, corpus, query_dict):
        self.model = CrossEncoder(model_path_or_name)
        self.corpus = corpus
        self.query_dict = query_dict
    
    def reranking(self, ranking_list, top_k = 20):
        """
        ranking list:
        {
         query_id : { doc_id : score, ... },
            ...
        }
        """
        
        res = copy.deepcopy(ranking_list)
        for qid, doc_dict in ranking_list.items():
            print("Processing query:", qid)
            sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            passage_ids = [doc_id for doc_id, _ in sorted_docs]
            for doc_id in passage_ids:
                res[qid][doc_id] = max(doc_dict.values())
            if len(passage_ids) < 2:
                continue
            scores = self.model.predict([(self.query_dict[qid], self.corpus[psg_id]) for psg_id in passage_ids])
            for idx, score in enumerate(scores):
                res[qid][passage_ids[idx]] += score
        return res
