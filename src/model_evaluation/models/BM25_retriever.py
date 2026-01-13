#!/usr/bin/env python
# -*- coding: utf-8 -*-

import janome.tokenizer
import os
import logging
import subprocess
import json
from pyserini.search.lucene import LuceneSearcher
from multiprocessing import Pool, cpu_count

class BM25Retriever:

    def __init__(self, corpus, index_dir, force = False):
        self.ix = None
        self.index_dir = index_dir
        self.tokenizer = janome.tokenizer.Tokenizer()
        self.build_index(corpus, index_dir=index_dir, force=force)

    def build_index(self, corpus, index_dir, force = False):
        """
        BM25のインデックスを構築する
        """
        pyserini_input_dir = os.path.join(index_dir, "pyserini_jsonl")
        os.makedirs(pyserini_input_dir, exist_ok=True)
        index_path = os.path.join(index_dir, "bm25_pyserini")
        if not os.path.isdir(index_path):
            os.mkdir(index_path)
        elif not force:
            logging.info(f"Loading existing BM25 index from {index_path}...")
            self.ix = LuceneSearcher(index_path)
            self.ix.set_language('ja')
            return
        
        with open(f"{pyserini_input_dir}/wiki_psg_pyserini.jsonl", "w", encoding="utf-8") as f:
            for doc_id, psg in corpus.items():
                saved_item = {"id" : doc_id, "contents" : psg}
                f.write(json.dumps(saved_item, ensure_ascii=False) + "\n")
        subprocess.run(["python", "-m", "pyserini.index.lucene",
                        "--input", pyserini_input_dir,
                        "--index", index_path,
                        "--collection", "JsonCollection",
                        "--generator", "DefaultLuceneDocumentGenerator",
                        "--threads", str(cpu_count()),
                        "--language", "ja",
                        "--storePositions", "--storeDocvectors", "--storeRaw"])
        self.ix = LuceneSearcher(index_path)
        self.ix.set_language('ja')

    def retrieve(self, queries, top_k = 100):
        """
        検索を行う関数．スコアが高い順に top_k 件を返す
        """
        responses = {}
        for qid, query_text in queries.items():
            hits = self.ix.search(query_text, k=top_k)
            responses[qid] = {hit.docid : hit.score for hit in hits}

        return responses
    
