

import os
os.environ["IR_DATASETS_HOME"] = "/mnt/LSTA6/bay1/kurokawa/ir_datasets_cache"
os.environ["HF_HOME"] = "/mnt/LSTA6/bay7/kurokawa/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/LSTA6/bay7/kurokawa/.cache"
os.environ["XDG_CACHE_HOME"] = "/mnt/LSTA6/bay7/kurokawa/.cache"
os.environ["HF_HUB_CACHE"] = "/mnt/LSTA6/bay7/kurokawa/.cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/LSTA6/bay7/kurokawa/.cache"
os.environ["VLLM_ASSETS_CACHE"] = "/mnt/LSTA6/bay7/kurokawa/.vllm_cache/"
os.environ["FLASHINFER_WORKSPACE_BASE"] = "/mnt/LSTA6/bay7/kurokawa/flashinfer_cache/"
os.environ["FLASHINFER_CUBIN_DIR"] = "/mnt/LSTA6/bay7/kurokawa/flashinfer_cache/cubin/"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import json
import orjson
from tqdm import tqdm
import glob
from models.diver_retriever import SemanticRetriever
from models.BM25_retriever import BM25Retriever
from models.query_expand import QueryExpand
from models.reranker import Reranker
from ranx import compare, Qrels, Run, fuse
from multiprocessing import Pool, cpu_count 
import torch
import gc
import pickle

compare_model_name_list = {
    "bm25" : BM25Retriever,
    #"tohoku-nlp/bert-base-japanese-v3" : SemanticRetriever,
    "AQ-MedAI/Diver-Retriever-0.6B" : SemanticRetriever,
    "sbintuitions/sarashina-embedding-v2-1b" : SemanticRetriever,
    # "cl-nagoya/ruri-v3-130m" : SemanticRetriever,
    #"Qwen/Qwen3-Embedding-4B" : SemanticRetriever,
    #"llm-jp/llm-jp-3.1-13b-instruct4" : SemanticRetriever
}

def get_query(all_query_dir):
    queries = {}
    qrels = {}
    with open(os.path.join(all_query_dir, "qrels.json"), "r", encoding="utf-8") as f:
        qrels = json.load(f)
    with open(os.path.join(all_query_dir, "query.json"), "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(os.path.join(all_query_dir, "excluded_dids.json"), "r", encoding="utf-8") as f:
        excluded_dids = set(json.loads(f.read()))
    with open(os.path.join(all_query_dir, "query_difficulty.json"), "r", encoding="utf-8") as f:
        query_difficulty = json.load(f)
    
    for qid, diff in query_difficulty.items():
        if diff == "exclude_query":
            if qid in queries:
                del queries[qid]
            if qid in qrels:
                del qrels[qid]
   
    print(f"Loaded {len(queries)} queries and {len(qrels)} qrels from {all_query_dir}")
    return queries, qrels, excluded_dids, query_difficulty

def load_worker(jsonl_path):
    passages = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = orjson.loads(line)
            psg_id = data["id"]
            psg = data["title"] + "\n" + data["text"].strip()
            passages[psg_id] = psg
    return passages

def create_model(data_dir, all_passages):
    # jsonl_paths = glob.glob(os.path.join(data_dir, "*.jsonl"))
    # all_passages = []
    # with Pool(cpu_count()) as p:
    #     for passage_dict in tqdm(p.imap_unordered(load_worker, jsonl_paths, chunksize=10), total=len(jsonl_paths)):
    #         all_passages.append(passage_dict)
    # all_passages = {k: v for d in all_passages for k, v in d.items()}
    # with open("/mnt/LSTA6/bay7/kurokawa/wiki_all_passages.pkl", "wb") as f:
    #     pickle.dump(all_passages, f)
    
    # new_all_passages = {}
    # for did in need_passages:
    #     new_all_passages[did] = all_passages[did]
    # all_passages = new_all_passages
    # print(f"Filtered to {len(all_passages)} passages needed for qrels.")

    # semantic_retriever = SemanticRetriever(
    #     model_path_or_name="AQ-MedAI/Diver-Retriever-4B",
    #     corpus=all_passages,
    #     index_dir = "/mnt/LSTA6/bay1/kurokawa/faiss_index/",
    #     force=False
    # )
    # bm25_retriever = BM25Retriever(
    #     corpus = all_passages,
    #     index_dir = "/mnt/LSTA6/bay1/kurokawa/bm25_index/",
    #     force=False
    # )
    models = {}
    for model_name, model_class in compare_model_name_list.items():
        print(f"Creating model: {model_name}")
        if model_class == SemanticRetriever:
            model_instance = model_class(
                model_path_or_name=model_name,
                corpus=all_passages,
                index_dir=f"/mnt/LSTA6/bay1/kurokawa/faiss_index_{model_name.replace('/', '_')}/",
                force=True
            )
        else:  # BM25Retriever
            model_instance = model_class(
                corpus=all_passages,
                index_dir=f"/mnt/LSTA6/bay1/kurokawa/bm25_index/",
                force=True
            )
        models[model_name] = model_instance

    return models, all_passages

def retreive(all_passages, queries):
    all_results = {}
    for model_name, model_class in compare_model_name_list.items():
        print(f"Creating model: {model_name}")

        if model_class == SemanticRetriever:
            model_instance = model_class(
                model_path_or_name=model_name,
                corpus=all_passages,
                index_dir=f"/mnt/LSTA6/bay1/kurokawa/faiss_index_{model_name.replace('/', '_')}/",
                force=False
            )
        else:  # BM25Retriever
            model_instance = model_class(
                corpus=all_passages,
                index_dir=f"/mnt/LSTA6/bay1/kurokawa/bm25_index/",
                force=False
            )
        print(f"Evaluating model: {model_name}")
        ranking_results = model_instance.retrieve(queries, top_k=1000)
        all_results[model_name] = ranking_results

        del model_instance
        torch.cuda.empty_cache()
        gc.collect()
    return all_results


def main():
    
    queries, qrels, excluded_dids, query_difficulty = get_query("data/j-bright_v1")
    #documents = load_documents("/mnt/LSTA6/bay1/kurokawa/wiki_docs/")
    with open("/mnt/LSTA6/bay7/kurokawa/wiki_all_passages.pkl", "rb") as f:
        all_passages = pickle.load(f)
    print("all passages loaded.", len(all_passages))
    # new_all_passages = {}
    # for did in need_passages:
    #     new_all_passages[did] = all_passages[did]
    # all_passages = new_all_passages
    for did in excluded_dids:
        if did in all_passages:
            del all_passages[did]

    print(f"Loaded {len(all_passages)}")
    query_expander = QueryExpand(
        model_path_or_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        corpus=all_passages,
        index_dir = "/mnt/LSTA6/bay7/kurokawa/bm25_index/"
    )
    expanded_queries = {}
    for qid, query in tqdm(queries.items()):
        expanded_query = query_expander.exapnd(query, round_nums=5)
        expanded_queries[qid] = expanded_query
    torch.cuda.empty_cache()
    gc.collect()
    del query_expander
    #models, all_passages = create_model("/mnt/LSTA6/bay1/kurokawa/wikipedia_data/wiki_psg_vec/", all_passages)
    # all_results = retreive(all_passages, queries)
    # expanded_results = retreive(all_passages, expanded_queries)
    # expanded_results = {k + "_QE": v for k, v in expanded_results.items()}
    # all_results.update(expanded_results)
    # reranker_method_and_model_name = [
    #     {"model_path_or_name": "google/gemma-3-12b-it", "method": "prp_allpair"},
    #     {"model_path_or_name": "google/gemma-3-12b-it", "method": "prp_sorting"},
    #     {"model_path_or_name": "google/gemma-3-12b-it", "method": "prp_sliding_k"},
    #     {"model_path_or_name": "castorini/monot5-base-msmarco", "method": "pointwise"},
    #     {"model_path_or_name": "google/gemma-3-12b-it", "method": "listwise"},
    # ]
    # tmp_results = {}
    # for model_name, results in all_results.items():
    #     for item in reranker_method_and_model_name:
    #         reranker_instance = Reranker(model_path_or_name=item["model_path_or_name"], corpus=all_passages, query_dict=queries, method=item["method"])
    #         reranked_result = reranker_instance.reranking(results, context=None, top_k=20)
    #         reranker_name = item["method"]
    #         tmp_results[f"{model_name}_Reranker_no_Context_{reranker_name}"] = reranked_result
    #         del reranker_instance
    #         torch.cuda.empty_cache()
    #         gc.collect()
    # for model_name, results in expanded_results.items():
    #     for item in reranker_method_and_model_name:
    #         reranker_instance = Reranker(model_path_or_name=item["model_path_or_name"], corpus=all_passages, query_dict=expanded_queries, method=item["method"])
    #         reranked_result = reranker_instance.reranking(results, context=None, top_k=20)
    #         reranker_name = item["method"]
    #         tmp_results[f"{model_name}_QE_Reranker_no_Context_{reranker_name}"] = reranked_result
    #         del reranker_instance
    #         torch.cuda.empty_cache()
    #         gc.collect()
    # all_results.update(tmp_results)
    # with open("all_retrieval_results.pkl", "wb") as f:
    #     pickle.dump(all_results, f)
    with open("all_retrieval_results.pkl", "rb") as f:
        all_results = pickle.load(f)
    query_type_results = {}
    for query_type in ["all", "easy_query", "hard_query"]:
        if query_type == "all":
            filtered_queries = queries
            filtered_expanded_queries = expanded_queries
            filtered_qrels = qrels
        else:
            filtered_queries = {qid: queries[qid] for qid in queries if query_difficulty[qid] == query_type}
            filtered_expanded_queries = {qid: expanded_queries[qid] for qid in expanded_queries if query_difficulty[qid] == query_type}
            filtered_qrels = {qid: qrels[qid] for qid in qrels if query_difficulty[qid] == query_type}
        assert len(filtered_queries) == len(filtered_qrels), f"Mismatch in queries and qrels for {query_type}, ({len(filtered_queries)} vs {len(filtered_qrels)})"
        print(f"Evaluating for query type: {query_type} with {len(filtered_queries)} queries.")
        partial_results = {}
        partial_query_expanded_results = {}
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
            
            
        for model_name, partial_result in partial_results.items():
            if model_name == "bm25" or ("bm25" in model_name and model_name.endswith("_QE")) or ("Reranker" in model_name):
                continue
            if "_QE" in model_name:
                fused_run = fuse([expanded_bm25_run, Run(partial_result, name=model_name)],  method="rrf", params = {"k":60, "name":f"RRF_{model_name}"})
            else:
                fused_run = fuse([bm25_run, Run(partial_result, name=model_name)],  method="rrf", params = {"k":60, "name":f"RRF_{model_name}"})
            runs.append(fused_run)

        qrels_ranx = Qrels(filtered_qrels)
        results = compare(qrels_ranx, runs, metrics=["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@100", "map", "recall@1000"])
        print(f"Results for query type: {query_type}")
        print(results.to_dataframe())
        query_type_results[query_type] = results
    for query_type, results in query_type_results.items():
        results.to_dataframe().to_csv(f"retrieval_results_{query_type}.csv")



if __name__ == "__main__":
    main()
