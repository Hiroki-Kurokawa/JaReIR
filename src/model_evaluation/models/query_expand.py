
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Query expansion using vLLM and BM25 retriever.
"""
import janome.tokenizer
from vllm import LLM, SamplingParams
from models.BM25_retriever import BM25Retriever

class QueryExpand:

    def __init__(self, model_path_or_name, corpus, index_dir):
        
        self.model = LLM(model=model_path_or_name, dtype="bfloat16", trust_remote_code=True, download_dir="/mnt/LSTA6/bay7/kurokawa/.vllm_cache/", max_model_len=32768)
        self.corpus = corpus
        self.retriever_model = BM25Retriever(corpus, index_dir=index_dir, force=False)

    def expand(self, query, round_nums=3, keep_passage_num=10, gen_num=5,
           reqeat_weight=3.0, accumulate=True, use_passage_filter=True):

        def extract_answer(text: str) -> str:
            return text.split("</think>\n")[-1].strip()

        def token_len_ja(text: str) -> int:
            t = janome.tokenizer.Tokenizer()
            return sum(1 for _ in t.tokenize(text))

        accumulated_expansions = []
        seen_passages = set()
        last_top_passages = []
        last_last_top_passages = []

        current_query = query

        for ridx in range(round_nums):
            # 1) retrieve
            ranking = self.retriever_model.retrieve({0: current_query}, top_k=1000)

            passages = [self.corpus[doc_id] for doc_id in ranking[0].keys()]

            # 2) pick top passages for prompting (filter optional)
            top_passages = []
            for p in passages:
                if use_passage_filter:
                    if p in seen_passages:
                        continue
                    if p in last_last_top_passages:
                        seen_passages.add(p)
                        continue
                top_passages.append(p)
                if len(top_passages) >= keep_passage_num:
                    break

            top_passages_str = "\n".join([f"{i+1}. {psg}" for i, psg in enumerate(top_passages)])

            prompt = f"""質問 "{query}" とその可能な回答パッセージ（これらのパッセージのほとんどは間違っています）が列挙されています:
{top_passages_str}

正しい回答パッセージを書いてください。例のパッセージだけでなく、自分の知識を使ってください。"""

            params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=256,
                n=gen_num,
            )
            res = self.model.generate(prompt, sampling_params=params)

            expansions = [extract_answer(o.text) for o in res[0].outputs]

            # 4) accumulate expansions
            if accumulate:
                accumulated_expansions.extend(expansions)
                used_expansions = accumulated_expansions
            else:
                used_expansions = expansions

            q_len = max(1, token_len_ja(query))
            e_len = max(1, token_len_ja("\n".join(used_expansions)))
            q_repeat = int(e_len / (q_len * reqeat_weight))
            q_repeat = max(q_repeat, 1)

            current_query = "\n".join([query] * q_repeat + used_expansions)
            last_last_top_passages = last_top_passages
            last_top_passages = top_passages

            print(f"[Round {ridx+1}] q_repeat={q_repeat}")
            print(f"[Round {ridx+1}] expansions[0]={used_expansions[0][:80]}...")
            print(f"[Round {ridx+1}] current_query(head)={current_query[:120]}...")

        return current_query
