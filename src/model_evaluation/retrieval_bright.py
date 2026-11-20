#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Retrieval evaluation for BRIGHT dataset.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import json
import orjson
from tqdm import tqdm
from src.model_evaluation.models.dense_retriever import SemanticRetriever
from src.model_evaluation.models.BM25_retriever import BM25Retriever
from src.model_evaluation.models.query_expand import QueryExpand
from src.model_evaluation.models.reranker import LLMBaseReranker, CEReranker
from src.utils.configs import load_configs

from ranx import compare, Qrels, Run, fuse
import torch
import gc
import pickle
CFG = load_configs()


compare_model_name_list = {
    "bm25" : BM25Retriever,
    "tohoku-nlp/bert-base-japanese-v3" : SemanticRetriever,
    "AQ-MedAI/Diver-Retriever-0.6B" : SemanticRetriever,
    "sbintuitions/sarashina-embedding-v2-1b" : SemanticRetriever,
}

def load_worker(jsonl_path):
    passages = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = orjson.loads(line)
            psg_id = data["id"]
            psg = data["title"] + "\n" + data["text"].strip()
            passages[psg_id] = psg
    return passages

def retreive(all_passages, queries, split_name):
    all_results = {}
    for model_name, model_class in compare_model_name_list.items():
        print(f"Creating model: {model_name}")

        if model_class == SemanticRetriever:
            model_instance = model_class(
                model_path_or_name=model_name,
                corpus=all_passages,
                index_dir= os.path.join("bright_index", f"faiss_index_{model_name.replace('/', '_')}_{split_name}"),
                force=False
            )
        else:  # BM25Retriever
            model_instance = model_class(
                corpus=all_passages,
                index_dir= os.path.join("/bright_index", f"bm25_index_{split_name}"),
                force=False
            )
        print(f"Evaluating model: {model_name}")
        ranking_results = model_instance.retrieve(queries, top_k=10000)
        all_results[model_name] = ranking_results

        del model_instance
        torch.cuda.empty_cache()
        gc.collect()

    return all_results


def run(all_passages, queries, qrels, split_name):
    qe_cache_path = os.path.join(f"bright_index/query_expand_{split_name}_cache", "pseudo_expanded_queries.pkl")
    if not os.path.exists(qe_cache_path):
        query_expander = QueryExpand(
            model_path_or_name="Qwen/Qwen3-30B-A3B-Thinking-2507",
            corpus=all_passages,
            index_dir = os.path.join("bright_index", f"bm25_index_{split_name}"),
        )
        expanded_queries = {}
        for qid, query in tqdm(queries.items()):
            expanded_query = query_expander.expand(query, round_nums=5)
            expanded_queries[qid] = expanded_query

        torch.cuda.empty_cache()
        gc.collect()
        del query_expander
        os.makedirs(f"bright_index/query_expand_{split_name}_cache", exist_ok=True)


        with open(qe_cache_path, "wb") as file:
            pickle.dump(expanded_queries, file)
    with open(qe_cache_path, "rb") as file:
        expanded_queries = pickle.load(file)

    all_results = retreive(all_passages, queries, split_name)
    expanded_results = retreive(all_passages, expanded_queries, split_name)
    expanded_results = {k + "_QE": v for k, v in expanded_results.items()}
    all_results.update(expanded_results)
    reranker_method_and_model_name = [
        {"model_path_or_name": "google/gemma-3-12b-it", "method": "prp_sliding_k"},
        {"model_path_or_name": "hotchpotch/japanese-reranker-base-v2", "method": "neural"},
    ]
    tmp_results = {}
    for model_name, results in all_results.items():
        if "bm25" in model_name:
            continue
        if "QE" not in model_name: # Rerank only for QE models
            continue
        for item in reranker_method_and_model_name:
            if item["method"] == "neural":
                reranker_instance = CEReranker(model_path_or_name=item["model_path_or_name"], corpus=all_passages, query_dict=queries)
            else:
                reranker_instance = LLMBaseReranker(model_path_or_name=item["model_path_or_name"], corpus=all_passages, query_dict=queries)
            reranked_result = reranker_instance.reranking(results, top_k=20)
            reranker_name = item["method"]
            tmp_results[f"{model_name}_Reranker_{reranker_name}"] = reranked_result
            del reranker_instance
            torch.cuda.empty_cache()
            gc.collect()
        
    all_results.update(tmp_results)
    os.makedirs(f"bright_index/retrieval_results_{split_name}", exist_ok=True)
    save_path = os.path.join(f"bright_index/retrieval_results_{split_name}", "pseudo_all_retrieval_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)
    with open(save_path, "rb") as f:
        all_results = pickle.load(f)
    
    runs = []
    for model_name, result in all_results.items():
        run_dict = {}
        for qid, psg_dict in result.items():
            sorted_psgs = sorted(psg_dict.items(), key=lambda x: x[1], reverse=True)
            run_dict[qid] = {psg_id: score for psg_id, score in sorted_psgs}
        runs.append(Run(run_dict, name=model_name))
    qrels_for_ranx = Qrels(qrels)
    results = compare(qrels=qrels_for_ranx, runs=runs, metrics=["ndcg@10", "mrr@10", "hits@10", "hit_rate@10", "recall@1000"])
    print(results.to_dataframe())
    os.makedirs("report_bright", exist_ok=True)
    results.to_dataframe().to_csv(f"report_bright/{split_name}.csv")


def main():
    
    with open("BRIGHT_translate/qrels.json", "r") as f:
        qrels = json.load(f)
    with open("BRIGHT_translate/queries_ja.json", "r") as f:
        queries = json.load(f)
    with open("BRIGHT_translate/corpus_ja.json", "r") as f:
        all_passages = json.load(f)
    print("all passages loaded.", len(all_passages))


    print(f"Loaded {len(all_passages)}")
    split_names = sorted(set(qid.split(":")[0] for qid in queries))
    print(f"Query splits: {split_names}")
    for split_name in split_names:
        if split_name not in ["stackoverflow"]:
            continue
        split_queries = {qid: queries[qid] for qid in queries if qid.startswith(split_name + ":")}
        split_qrels = {qid: qrels[qid] for qid in qrels if qid.startswith(split_name + ":")}
        split_passages = {did: all_passages[did] for did in all_passages.keys() if did.startswith(split_name + ":")}
        print(f"Split: {split_name}, Queries: {len(split_queries)}, Qrels: {len(split_qrels)}, Passages: {len(split_passages)}")
        run(split_passages, split_queries, split_qrels, split_name)

if __name__ == "__main__":
    main()
