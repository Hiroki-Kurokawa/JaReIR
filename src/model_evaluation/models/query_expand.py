
import faiss
import janome.tokenizer
import os
import logging
import subprocess
import json
from vllm import LLM, SamplingParams
from pyserini.search.lucene import LuceneSearcher
from multiprocessing import Pool, cpu_count
from models.BM25_retriever import BM25Retriever

class QueryExpand:
    """
    ThinkQEに基づくクエリ拡張モデル
    """
    def __init__(self, model_path_or_name, corpus, index_dir):
        
        self.model = LLM(model=model_path_or_name, dtype="bfloat16", trust_remote_code=True, download_dir="/mnt/LSTA6/bay7/kurokawa/.vllm_cache/", max_model_len=32768)
        self.corpus = corpus
        self.retriever_model = BM25Retriever(corpus, index_dir=index_dir, force=False)
    
    def exapnd(self, query, round_nums):
        
        context = []
        black_doc_ids = set()
        first_ranking_results = self.retriever_model.retrieve({0: query}, top_k=5)
        for i, (doc_id, _) in enumerate(first_ranking_results[0].items()):
            context.append(f"文章{i+1}:{self.corpus[doc_id]}")
            black_doc_ids.add(doc_id)
        context = "\n".join(context)
        
        last_round_expansion = query

        # --- プロンプトの修正箇所 ---
        
        # システムプロンプト的な役割を含む指示
        instruction = """
あなたは検索システムのクエリ拡張モジュールです。
与えられたクエリと参考文章から、検索精度を向上させるための「追加キーワード」を生成してください。

【制約事項】
1. 出力は**スペース区切りの単語リスト**のみにしてください。
2. 説明文、挨拶、記号、箇条書きは一切含めないでください。
3. 文法的に完全な文章（～です、～ます）は禁止です。
4. 元のクエリに含まれる重要な単語は残しつつ、類義語や関連語を追加してください。

【出力例】
入力クエリ: 富士山 高さ
出力: 富士山 高さ 標高 メートル 日本一 登山
"""

        first_prompt_template = f"""{instruction}
----------------
以下が入力データです。

クエリ:{query}
おそらく役立つ文章:{context}

拡張クエリ（キーワードのみ）:"""

        subsequent_prompt_template = f"""{instruction}
----------------
以下が入力データです。

クエリ: !!QUERY!!
おそらく役立つ文章: !!CONTEXT!!
前のラウンドで生成された拡張クエリ: !!LAST_ROUND_EXPANSION!!

上記を踏まえ、より検索に適したキーワード列に修正してください。
拡張クエリ（キーワードのみ）:"""
        
        # ------------------------

        for round_num in range(round_nums):
            if round_num == 0:
                prompt = first_prompt_template
            else:
                prompt = subsequent_prompt_template.replace("!!QUERY!!", query).replace("!!CONTEXT!!", context).replace("!!LAST_ROUND_EXPANSION!!", last_round_expansion)
            
            # max_tokensを小さく設定 (512 -> 128程度)
            response = self.model.generate(prompt, sampling_params=SamplingParams(temperature=0.0, top_p=0.9, max_tokens=128),)
            
            # 改行コードなどを削除して整形
            last_round_expansion = response[0].outputs[0].text.strip().replace("\n", " ")
            
            # (オプション) もし文章になってしまった場合、Janomeで名詞だけ抽出する処理を挟むと堅牢になります
            # last_round_expansion = self.extract_nouns(last_round_expansion) 

            print("Round", round_num + 1, "Expansion:", last_round_expansion)
            
            context = []
            ranking_results = self.retriever_model.retrieve({0: last_round_expansion}, top_k=10000)
            for i, (doc_id, _) in enumerate(ranking_results[0].items()):
                if doc_id in black_doc_ids:
                    continue
                context.append(f"文章{i+1}:{self.corpus[doc_id]}")
                black_doc_ids.add(doc_id)
                if len(context) >= 5:
                    break
            context = "\n".join(context)
            
        return last_round_expansion

    # 補助関数: どうしても文章になってしまう場合の保険
    def extract_nouns(self, text):
        t = janome.tokenizer.Tokenizer()
        words = []
        for token in t.tokenize(text):
            # 名詞のみ、かつ非自立語（代名詞など）を除外するなど調整
            if token.part_of_speech.startswith('名詞'):
                words.append(token.surface)
        return " ".join(words)
