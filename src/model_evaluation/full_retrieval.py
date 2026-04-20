#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation of retrieval performance on the document.
"""
from dotenv import load_dotenv
load_dotenv()

from collections import defaultdict
import os
import json
import orjson
import re
import mwparserfromhell
from tqdm import tqdm
import glob
from src.model_evaluation.models.dense_retriever import SemanticRetriever
from src.model_evaluation.models.BM25_retriever import BM25Retriever
from src.model_evaluation.models.query_expand import QueryExpand
from src.model_evaluation.models.reranker import LLMBaseReranker, CEReranker
from ranx import compare, Qrels, Run, fuse
from multiprocessing import Pool, cpu_count 
import torch
import gc
import pickle
from dotenv import load_dotenv
from src.utils.configs import load_configs
CFG = load_configs()
load_dotenv()

compare_model_name_list = {
    "bm25" : BM25Retriever,
    #"tohoku-nlp/bert-base-japanese-v3" : SemanticRetriever,
    #"AQ-MedAI/Diver-Retriever-0.6B" : SemanticRetriever,
    "sbintuitions/sarashina-embedding-v2-1b" : SemanticRetriever,
}

def get_query(all_query_dir):
    queries = {}
    qrels = {}
    with open(os.path.join(all_query_dir, "qrels.json"), "r", encoding="utf-8") as f:
        qrels = json.load(f)
    with open(os.path.join(all_query_dir, "query.json"), "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(os.path.join(all_query_dir, "query_difficulty.json"), "r", encoding="utf-8") as f:
        query_difficulty = json.load(f)
    
    for qid, diff in query_difficulty.items():
        if diff == "exclude_query":
            if qid in queries:
                del queries[qid]
            if qid in qrels:
                del qrels[qid]
    for qid, values in qrels.items():
        qrels[qid] = {str(psg_id): score for psg_id, score in values.items()}
    
    for qid, values in qrels.items():
        qrels[qid] = {psg_id.split("_")[0] : score for psg_id, score in values.items()}
   
    print(f"Loaded {len(queries)} queries and {len(qrels)} qrels from {all_query_dir}")
    return queries, qrels, {}, query_difficulty

def load_worker(jsonl_path):
    passages = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = orjson.loads(line)
            psg_id = data["id"]
            psg = data["title"] + "\n" + data["text"].strip()
            passages[psg_id] = psg
    return passages

def retrieve(all_passages, queries):
    all_results = {}
    for model_name, model_class in compare_model_name_list.items():
        print(f"Creating model: {model_name}")

        if model_class == SemanticRetriever:
            model_instance = model_class(
                model_path_or_name=model_name,
                corpus=all_passages,
                index_dir= os.path.join("full_wiki_index", f"faiss_index_{model_name.replace('/', '_')}"),
                force=False
            )
        else:  # BM25Retriever
            model_instance = model_class(
                corpus=all_passages,
                index_dir= os.path.join("full_wiki_index", "bm25_index/"),
                force=False
            )
        print(f"Evaluating model: {model_name}")
        ranking_results = model_instance.retrieve(queries, top_k=10000)
        all_results[model_name] = ranking_results

        del model_instance
        torch.cuda.empty_cache()
        gc.collect()

    return all_results

def worker_for_load_jsonl(jsonl_path):
    passages = {}
    with open(jsonl_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            psg_id = data["id"]
            title = " ".join(data["section_titles"])
            title = re.sub(r'_dup\d+', '', title)
            text = data["text"]

            text = mwparserfromhell.parse(text).strip_code()
            text = re.sub(r'[\ud800-\udfff]', '', text)
            text = text.replace("\x00", "")
            text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
            all_text = title + "\n" + text

            passages[str(psg_id)] = all_text
    return passages

def load_passages(data_dir):
    jsonl_paths = glob.glob(os.path.join(data_dir, "*.jsonl"))
    passages = {}
    with Pool(min(cpu_count(), len(jsonl_paths))) as pool:
        for passage_dict  in tqdm(pool.imap_unordered(worker_for_load_jsonl, jsonl_paths)):
            passages.update(passage_dict)
    print(f"Loaded {len(passages)} passages from {data_dir}")
    return passages

def main():
    
    queries, qrels, excluded_dids, query_difficulty = get_query(CFG["data_config"]["paths"]["benchmark_path"])
    all_passages = load_passages(CFG["data_config"]["paths"]["wikipedia_psg_path"])
    os.makedirs(os.path.dirname("wikipedia_data/wiki_psg_full_cache.pkl"), exist_ok=True)
    with open("wikipedia_data/wiki_psg_full_cache.pkl", "wb") as f:
        pickle.dump(all_passages, f)
    with open("wikipedia_data/wiki_psg_full_cache.pkl", "rb") as f:
        all_passages = pickle.load(f)
    print("all passages loaded.", len(all_passages))


    tmp = defaultdict(list)
    for pid, passage in all_passages.items():
        doc_id, sec_id = pid.split("_", 1)
        tmp[doc_id].append((int(sec_id), passage))

    all_passages = {
        doc_id: "\n".join([psg for _, psg in sorted(items, key=lambda x: x[0])])
        for doc_id, items in tmp.items()
    }
    print(f"Loaded {len(all_passages)} documents.")

    query_expander = QueryExpand(
        model_path_or_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        corpus=all_passages,
        index_dir = os.path.join("full_wiki_index", "bm25_index")
    )
    expanded_queries = {}
    for qid, query in tqdm(queries.items()):
        expanded_query = query_expander.expand(query, keep_passage_num=3, round_nums=5)
        expanded_queries[qid] = expanded_query

    torch.cuda.empty_cache()
    gc.collect()
    del query_expander
    os.makedirs("wikipedia_data/query_expand_cache", exist_ok=True)
    with open("wikipedia_data/query_expand_cache/expanded_queries.pkl", "wb") as file:
        pickle.dump(expanded_queries, file)
    with open("wikipedia_data/query_expand_cache/expanded_queries.pkl", "rb") as file:
        expanded_queries = pickle.load(file)

    all_results = retrieve(all_passages, queries)
    expanded_results = retrieve(all_passages, expanded_queries)
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
    os.makedirs("wikipedia_data/retrieval_results", exist_ok=True)
    with open("wikipedia_data/retrieval_results/all_retrieval_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    with open("wikipedia_data/retrieval_results/all_retrieval_results.pkl", "rb") as f:
        all_results = pickle.load(f)

    query_type_results = {}
    for query_type in ["all", "easy_query", "hard_query"]:
        if query_type == "all":
            filtered_queries = queries
            filtered_qrels = qrels
        else:
            filtered_queries = {qid: queries[qid] for qid in queries if query_difficulty[qid] == query_type}
            filtered_qrels = {qid: qrels[qid] for qid in qrels if query_difficulty[qid] == query_type}
        assert len(filtered_queries) == len(filtered_qrels), f"Mismatch in queries and qrels for {query_type}, ({len(filtered_queries)} vs {len(filtered_qrels)})"
        print(f"Evaluating for query type: {query_type} with {len(filtered_queries)} queries.")
        partial_results = {}
        runs = []
        bm25_run = None
        expanded_bm25_run = None
        for model_name, ranking_results in all_results.items():
            partial_result = {}
            for qid, rankings in ranking_results.items():
                if qid not in filtered_queries:
                    continue
                partial_result[qid] = rankings
            partial_results[model_name] = partial_result
            runs.append(Run(partial_result, name=model_name))
            if model_name == "bm25":
                bm25_run = Run(partial_result, name=model_name)
            if "bm25" in model_name and model_name.endswith("_QE"):
                expanded_bm25_run = Run(partial_results[model_name], name=model_name)
            
        ## rrf fusion
        for model_name, partial_result in partial_results.items():
            if model_name == "bm25" or ("bm25" in model_name and model_name.endswith("_QE")) or ("Reranker" in model_name):
                continue
            if "_QE" in model_name:
                fused_run = fuse([expanded_bm25_run, Run(partial_result, name=model_name)],  method="rrf", params = {"k":60, "name":f"RRF_{model_name}"})
            else:
                fused_run = fuse([bm25_run, Run(partial_result, name=model_name)],  method="rrf", params = {"k":60, "name":f"RRF_{model_name}"})
            runs.append(fused_run)
        print("N queries:", len(filtered_queries))
        print("N qrels:", len(filtered_qrels))
        print("N bm25 run:", len(partial_results["bm25"]))
        print("missing qids:", len(set(filtered_qrels) - set(partial_results["bm25"])))
        qrels_ranx = Qrels(filtered_qrels)
        results = compare(qrels_ranx, runs, metrics=["ndcg@10", "mrr@10", "hits@10", "hit_rate@10", "recall@1000"])
        print(f"Results for query type: {query_type}")
        print(results.to_dataframe())
        query_type_results[query_type] = results
    for query_type, results in query_type_results.items():
        os.makedirs("report_full", exist_ok=True)
        results.to_dataframe().to_csv(os.path.join("report_full", f"retrieval_results_{query_type}.csv"))


if __name__ == "__main__":
    main()
