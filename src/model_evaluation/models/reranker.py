#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["HF_HOME"] = "/mnt/LSTA6/bay7/kurokawa/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/LSTA6/bay7/kurokawa/.cache"
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from huggingface_hub import login
login(token="YOUR_HF_TOKEN_HERE")  # Replace with your actual token or use environment variable
from vllm import LLM, SamplingParams
import logging
import copy
import torch
import json
import re
from functools import cmp_to_key, partial
from itertools import combinations

logging.basicConfig(level=logging.INFO)
class Reranker:
 
    def __init__(self, model_path_or_name, corpus, query_dict, method="prp_allpair"):
        assert not (method == "pointwise" and model_path_or_name != "castorini/monot5-base-msmarco"), "Pointwise method is only supported with monot5-base-msmarco model."
        cfg = AutoConfig.from_pretrained(model_path_or_name, cache_dir="/mnt/LSTA6/bay7/kurokawa/.cache")
        if cfg.is_encoder_decoder:
            assert method == "pointwise", "Seq2Seq (e.g., MonoT5) supports only pointwise in this implementation."
        else:
            assert method != "pointwise", "CausalLM uses PRP/listwise; pointwise is for MonoT5 here."
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=True, cache_dir="/mnt/LSTA6/bay7/kurokawa/.cache")
        self.model_path_or_name = model_path_or_name
        self.method = method
        if cfg.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path_or_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="/mnt/LSTA6/bay7/kurokawa/.cache",
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_or_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="/mnt/LSTA6/bay7/kurokawa/.cache",
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
        if self.method == "pointwise":
            assert self.token_id_true is not None and self.token_id_false is not None
        if self.method.startswith("prp"):
            assert self.token_id_a is not None and self.token_id_b is not None

        self.prp_prompt_template = """
あなたは，質問と2つの文章（AとB）を与えられます．
どちらの文章が質問に対してより関連性が高いかを判断してください．

出力形式:
- "A" もし文章Aがより関連性が高い場合
- "B" もし文章Bがより関連性が高い場合


質問: {question}
文章A: {passage_a}
文章B: {passage_b}

出力:
"""
        self.pointwise_prompt_template = """質問: {question}
文章: {passage}
関連性:"""

        self.listwise_prompt_template = """
あなたは，検索システムの再ランキングモジュールです．

タスク:
ユーザが質問とN個の候補文章（1からNまでのインデックス付き）が与えられます．質問に最も役立つ順にランキングを出力してください．

出力形式は以下の制約を厳密に守ってください:
- 出力は，jsonオブジェクトでなければなりません。（例: {{"ranking": [...]}}）
- Jsonは正確に1つのキー "ranking" を持つ必要があります．
- "ranking" の各要素は，1からNまでの整数の順列でなければなりません（各インデックスは正確に一度だけ出現します）．
- 説明を加えず，余分なキーや文章を含めないでください．

質問:
{query}

文章候補:
{passage_list}

OUTPUT JSON:
{{"ranking":[...]}}
"""

    def _get_token_ids(self, base_char):
        id_space = self._single_token_id(f" {base_char}")
        id_no_space = self._single_token_id(base_char)
        return id_space if id_space is not None else id_no_space

    def _single_token_id(self, s: str):
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def reranking(self, ranking_list, context=None, top_k = 20):
        """
        ranking list:
        {
         query_id : { doc_id : score, ... },
            ...
        }
        """
        
        assert not (self.method == "pointwise" and context is not None), "Pointwise method does not support context."
        res = copy.deepcopy(ranking_list)
        output_count = {"A" : 0, "B" : 0}
        for qid, doc_dict in ranking_list.items():
            print("Processing query:", qid)
            if self.method == "prp_allpair":
                res, output_count = self.__prp_allpair(doc_dict, qid, context, top_k, res, output_count)
            elif self.method == "prp_sorting":
                res = self.__prp_sorting(doc_dict, qid, context, top_k, res)
            elif self.method == "prp_sliding_k":
                res = self.__prp_sliding_k(doc_dict, qid, context, top_k, res)
            elif self.method == "pointwise":
                res = self.__pointwise(doc_dict, qid, top_k, res)
            elif self.method == "listwise":
                res = self.__listwise(doc_dict, qid, context, top_k, res)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        if self.method == "listwise":
            print(f"Listwise reranking failed {self.listwise_fail_count} times.(out of {len(ranking_list)} queries)")
        print(output_count)
        return res

    def __get_prp_scores(self, context, qid, psg1_id, psg2_id):
        psg1 = self.corpus[psg1_id]
        psg2 = self.corpus[psg2_id]
        if context is not None:
            prompt = self.prp_prompt_template.format(
                question=self.query_dict[qid],
                passage_a=psg1,
                passage_b=psg2,
                context=context
            )
        else:
            prompt = self.prp_prompt_template.format(
                question=self.query_dict[qid],
                passage_a=psg1,
                passage_b=psg2,
                context=""
            )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        score_a = next_token_logits[self.token_id_a].item()
        score_b = next_token_logits[self.token_id_b].item()
        return score_a, score_b
    
    def _seq2seq_logprob(self, enc_inputs, target_text: str) -> float:
        labels = self.tokenizer(
            target_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.model.device)

        with torch.no_grad():
            out = self.model(**enc_inputs, labels=labels)

        # loss は平均なので、合計対数尤度に戻す
        return -out.loss.item() * labels.shape[1]
    
    def __get_pointwise_scores_seq2seq(self, qid, psg_id):
        prompt = self.pointwise_prompt_template.format(
            question=self.query_dict[qid],
            passage=self.corpus[psg_id],
        )
        enc_inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)

        lp_true  = self._seq2seq_logprob(enc_inputs, "true")
        lp_false = self._seq2seq_logprob(enc_inputs, "false")
        return lp_true, lp_false

    def __prp_allpair(self, doc_dict, qid, context, top_k, res, output_count):
        sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        passage_ids = [doc_id for doc_id, _ in sorted_docs]
        for doc_id in passage_ids:
            res[qid][doc_id] = max(doc_dict.values())
        for psg1_id, psg2_id in combinations(passage_ids, 2):
            score_a, score_b = self.__get_prp_scores(context, qid, psg1_id, psg2_id)
            if score_a > score_b:
                res[qid][psg1_id] += 1.0
                output_count["A"] += 1
            elif score_b > score_a:
                res[qid][psg2_id] += 1.0
                output_count["B"] += 1
        return res, output_count

    def __compare_base(self, context, qid, psg1_id, psg2_id):
        score_a, score_b = self.__get_prp_scores(context, qid, psg1_id, psg2_id)
        if score_a > score_b:
            return 1
        elif score_b > score_a:
            return -1
        else:
            return 0

    def __prp_sorting(self, doc_dict, qid, context, top_k, res):
        """
        A, B, C, D, E
        A, B, C -> B, A, C
        B, A, C, D, E

        """

        sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]

        passage_ids = [doc_id for doc_id, _ in sorted_docs]
        for doc_id in passage_ids:
            res[qid][doc_id] = max(doc_dict.values())
        compare_func = partial(self.__compare_base, context, qid)
        ranked_passage_ids = sorted(passage_ids, key=cmp_to_key(compare_func), reverse=True)
        for rank, doc_id in enumerate(ranked_passage_ids):
            res[qid][doc_id] += top_k - rank
        return res
    
    def __prp_sliding_k(self, doc_dict, qid, context, top_k, res, sliding_k = 10):
        """
        A, B, C, D, E
        A, B, C, E, D
        A, B, C, E, D
        A, C, B, E, D
        C, A, B, E, D
        """

        sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        passage_ids = [doc_id for doc_id, _ in sorted_docs]
        n = len(passage_ids)
        for _ in range(sliding_k):
            for j in range(n - 1, 0, -1):
                pair_1_doc_id = passage_ids[j]
                pair_2_doc_id = passage_ids[j - 1]
                score_a, score_b = self.__get_prp_scores(context, qid, pair_1_doc_id, pair_2_doc_id)
                if score_a > score_b: # If the latter is better, swap
                    passage_ids[j], passage_ids[j - 1] = passage_ids[j - 1], passage_ids[j]
        for rank, doc_id in enumerate(passage_ids):
            res[qid][doc_id] = max(doc_dict.values()) + (top_k - rank)
        return res
    
    
    def __pointwise(self, doc_dict, qid, top_k, res):
        sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        passage_ids = [doc_id for doc_id, _ in sorted_docs]

        for doc_id, score in sorted_docs:
            res[qid][doc_id] = score

        for psg_id in passage_ids:
            p_true, p_false = self.__get_pointwise_scores_seq2seq(qid, psg_id)
            res[qid][psg_id] += (p_true - p_false)  
        return res
        
    def _extract_last_valid_ranking_json(self, text: str):
       
        # {"ranking":[...]} っぽい塊を最後から探す
        candidates = re.findall(r'\{[^{}]*"ranking"[^{}]*\}', text)
        for s in reversed(candidates):
            try:
                obj = json.loads(s)
                if "ranking" in obj:
                    return obj
            except:
                pass
        return None

    def __listwise(self, doc_dict, qid, context, top_k, res):
        sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        passage_ids = [doc_id for doc_id, _ in sorted_docs]

        max_score = max(doc_dict.values())
        for doc_id, _ in sorted_docs:
            res[qid][doc_id] = max_score

        passage_list_str = ""
        for idx, psg_id in enumerate(passage_ids, start=1):
            passage = self.corpus[psg_id]
            passage_list_str += f"{idx}) {passage}\n\n"

        prompt = self.listwise_prompt_template.format(
            query=self.query_dict[qid],
            context=context or "",
            passage_list=passage_list_str
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_text = self.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        obj = self._extract_last_valid_ranking_json(gen_text)
        if obj is None:
            print("Failed to parse ranking JSON. gen_text:", gen_text)
            self.listwise_fail_count += 1
            return res

        ranking = obj["ranking"]
        n = len(passage_ids)

        # permutation検証（壊れてたら捨てる/リトライが理想）
        if not (isinstance(ranking, list) and len(ranking) == n and sorted(ranking) == list(range(1, n+1))):
            print("Invalid ranking:", ranking, "gen_text:", gen_text)
            return res

        # スコア付け（上位ほど加点）
        for rank, idx in enumerate(ranking):
            doc_id = passage_ids[idx - 1]
            res[qid][doc_id] += (top_k - rank)

        return res
                    

# class GroupReranker:
#     def __init__(self, model_path, sys_prompt, user_prompt) -> None:
#         # vllm offline inference
#         self.llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count(), max_model_len=32000)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.sampling_params = SamplingParams(temperature=0.3, top_p=0.8, max_tokens=8000, logprobs=10)
#         sys_prompt = '''Your task is to evaluate and rank documents based on how well they help answer the given query. Follow this evaluation priority:
# 1. PRIMARY: Usefulness & Helpfulness - Does the document provide actionable information, solutions, or direct answers that help address the user's needs?
# 2. SECONDARY: Relevance - Does the document contain information related to the query topic?

# Evaluation Process:
# 1. First, identify the user's core intent and what kind of help they need from the query
# 2. For each document, assess:
#    - How directly it addresses the user's intent
#    - What actionable information or answers it provides
#    - How much it helps solve the user's problem or need
# 3. Compare documents against each other to ensure proper ranking
# 4. Assign scores that reflect the relative usefulness ranking

# Scoring Scale (0-10):
# - 9-10: Extremely helpful, directly answers the query with actionable information
# - 7-8: Very helpful, provides substantial useful information for the query
# - 5-6: Moderately helpful, contains some useful information but incomplete
# - 3-4: Minimally helpful, limited useful information despite topic relevance
# - 1-2: Barely helpful, mentions related topics but provides little useful information
# - 0: Not helpful at all, cannot assist with answering the query
# '''

#     user_prompt = '''I will provide you {TOPK} documents, each indicated by a numerical identifier []. Score these documents based on their Usefulness and Relevance to the query.
# Query:
# {QUERY}

# Documents:
# {PASSAGES}

# ## Final Output Format
# You must structure your response in exactly two parts: provide your brief reasoning process first, then output final scores in JSON format like below, with document IDs as string keys and integer scores as values for all {TOPK} documents. 
# The reasoning process and answer are enclosed within <reason> </reason> and <answer> </answer> tags, respectively. Do NOT output anything outside the specified tags. Follow this exact format:
# <reason> 
# [Analyze each document's usefulness and relevance to the query, explaining your scoring rationale]
# </reason>
# <answer> 
# \```json
# {{"[1]": 5, "[2]": 3, "[3]": 8}}
# \``` 
# </answer>'''
#         self.group_system_prompt = sys_prompt
#         self.group_user_prompt = user_prompt

#     def rerank(self, query, doc_list):
#         docs_str = ''.join(["[{}]. {}\n\n".format(idx+1, doc_text) for idx, doc_text in enumerate(doc_list)])

#         group_texts = self.group_user_prompt.format(QUERY=query, PASSAGES=docs_str, TOPK=len(doc_list))

#         message = self.tokenizer.apply_chat_template(
#             [{'role': 'system', 'content': self.group_system_prompt}, 
#             {'role': 'user', 'content': group_texts}], tokenize=False, add_generation_prompt=True)

#         output = self.llm.generate(message, self.sampling_params, use_tqdm=True)

#         return output
