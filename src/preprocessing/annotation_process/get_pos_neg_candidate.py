#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Get negative candidate passage for annotation from Yahoo! Japan QA pairs.
"""

import os
from matplotlib.pyplot import title
from sentence_transformers import SentenceTransformer
import urllib
import urllib.parse
import re
import mwparserfromhell

import numpy as np
import json
from tqdm import tqdm
import glob

import faiss
import janome.tokenizer
import torch
import subprocess
from pyserini.search.lucene import LuceneSearcher
import pickle
from multiprocessing import Pool, cpu_count 
from src.utils.configs import load_configs
CFG = load_configs("configs")

MODEL = SentenceTransformer("sbintuitions/sarashina-embedding-v2-1b", cache_folder=CFG["model_config"]["paths"]["model_cache_dir"], device="cuda")
TOKENIZER = janome.tokenizer.Tokenizer()

def get_query(all_query_dir):
    queries = {}

    paths = glob.glob(os.path.join(all_query_dir, "*.json"))
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
            query_id = data["qid"]
            queries[query_id] = data
    print(f"Loaded {len(queries)} queries from {all_query_dir}")
    return queries

def tokenize_text(task):
    doc_id, text = task
    return doc_id, [token for token in TOKENIZER.tokenize(text, wakati=True) if token not in [" ", "\n"]]

def process_text(data):
    title = " ".join(data["section_titles"])
    title = re.sub(r'_dup\d+', '', title)
    text = data["text"]
    text = mwparserfromhell.parse(text).strip_code()
    text = re.sub(r'[\ud800-\udfff]', '', text)
    text = text.replace("\x00", "")
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    text = title + "\n" + text.strip()
    return text

def pyserini_load_text(jsonl_path):
    save_pyserini_data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = json.loads(line)
            psg_id = data["id"]
            psg = process_text(data)
            if len(psg) > 1094 or len(psg) < 29:
                continue
            save_pyserini_data.append({"id": psg_id, "contents": psg})
    return save_pyserini_data

def build_bm25_index_by_pyserini(jsonl_paths):
    index_path = os.path.join(CFG['data_config']['paths']['wikipedia_index_dir'], "wiki_psg_bm25_pyserini")
    pyserini_input_dir = os.path.join(CFG['data_config']['paths']['wikipedia_index_dir'], "wiki_pyserini_jsonl")
    os.makedirs(index_path, exist_ok=True)
    os.makedirs(pyserini_input_dir, exist_ok=True)
    save_pyserini_data = []
    with Pool(cpu_count()) as p:
        for passage_list in tqdm(p.imap_unordered(pyserini_load_text, jsonl_paths), total=len(jsonl_paths)):
            save_pyserini_data.extend(passage_list)
                
    with open(f"{pyserini_input_dir}/wiki_psg_pyserini.jsonl", "w", encoding="utf-8") as f:
        for item in save_pyserini_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    subprocess.run(["python", "-m", "pyserini.index.lucene",
                    "--input", pyserini_input_dir,
                    "--index", index_path,
                    "--collection", "JsonCollection",
                    "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", str(cpu_count()),
                    "--language", "ja",
                    "--storePositions", "--storeDocvectors", "--storeRaw"])

def build_faiss_index(jsonl_paths):

    faiss_id_converter = {}
    count = 0
    dim = 1792
    faiss_index = faiss.IndexHNSWFlat(dim, 32)
    faiss_index.hnsw.efConstruction = 200
    faiss_index.hnsw.efSearch = 64
  
    BATCH_SIZE = 10 ** 4
    batch_texts = []

    for jsonl_path in tqdm(jsonl_paths):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                data = json.loads(line)
                psg_id = data["id"]
                text = process_text(data)
                if len(text) > 1094 or len(text) < 29:
                    continue
                batch_texts.append(text)
                faiss_id_converter[str(count)] = psg_id
                count += 1

                if len(batch_texts) >= BATCH_SIZE:
                    try:
                        batch_embeddings = MODEL.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=128, normalize_embeddings=True)
                    except Exception as e:
                        print(f"Error encoding batch: {e}")
                        for txt in batch_texts:
                            with open("tmp.txt", "w", encoding="utf-8") as tmpf:
                                tmpf.write(txt)
                            MODEL.encode(txt, convert_to_numpy=True, show_progress_bar=False, batch_size=1, normalize_embeddings=True)
                        raise e
                    arr = np.ascontiguousarray(np.vstack(batch_embeddings).astype("float32"))
                    faiss_index.add(arr)
                    batch_texts = []

    if batch_texts:
        batch_embeddings = MODEL.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=8, normalize_embeddings=True)
        arr = np.ascontiguousarray(np.vstack(batch_embeddings).astype("float32"))
        faiss_index.add(arr)
    #cpu_index_from_gpu = faiss.index_gpu_to_cpu(gpu_index)
    os.makedirs(CFG["data_config"]["paths"]["wikipedia_index_dir"], exist_ok=True)
    faiss.write_index(faiss_index, os.path.join(CFG["data_config"]["paths"]["wikipedia_index_dir"], "wiki_psg_faiss_sb.index"))
    #faiss.write_index(cpu_index_from_gpu, f"{INDEX_DIR}/wiki_psg_faiss.index")
    print(f"FAISS index with {faiss_index.ntotal} vectors created and saved.")
    with open(os.path.join(CFG["data_config"]["paths"]["wikipedia_index_dir"], "wiki_faiss_sb_id_converter.json"), "w", encoding="utf-8") as f:
        json.dump(faiss_id_converter, f, ensure_ascii=False, indent=4)

def load_indexes(data_dir):
    jsonl_paths = glob.glob(os.path.join(data_dir, "*.jsonl"))
    #build_bm25_index_by_pyserini(jsonl_paths)
    print("BM25 index built and saved.")
    #build_faiss_index(jsonl_paths)
    print("FAISS index built and saved.")
    bm25 = LuceneSearcher(os.path.join(CFG["data_config"]["paths"]["wikipedia_index_dir"], "wiki_psg_bm25_pyserini"))
    bm25.set_language('ja')
    doc_ids = []
    
    faiss_index = faiss.read_index(os.path.join(CFG["data_config"]["paths"]["wikipedia_index_dir"], "wiki_psg_faiss_sb.index"))
    with open(os.path.join(CFG["data_config"]["paths"]["wikipedia_index_dir"], "wiki_faiss_sb_id_converter.json"), "r", encoding="utf-8") as f:
        faiss_id_converter = json.load(f)

    passages = {}
    for jsonl_path in tqdm(jsonl_paths):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                data = json.loads(line)
                psg_id = data["id"]
                psg = data["title"] + "\n" + data["text"].strip()
                passages[psg_id] = {"title": data["title"], "psg": psg}
    print(f"Loaded {len(passages)} passages from {data_dir}")
    
    return doc_ids, bm25, faiss_id_converter, faiss_index, passages

def get_positive_documents(wiki_title, documents):
    res = []
    wiki_title_normalized = wiki_title.replace("_", " ")
    for doc_id, doc_info in documents.items():
        if doc_info["title"].startswith(wiki_title_normalized):
            res.append({"doc_id" : doc_id, "info" : doc_info})
    return res

def get_negative_ranking_list(query, bm25=None, dense=None):
    ranking_list = []
    if bm25:
        hits = bm25.search(query, k=10000)
        for i in range(len(hits)):
            doc_id = hits[i].docid
            if doc_id == -1:
                print(f"Warning: doc_id -1 found in BM25 results :{hits[i]} (index {i})")
                break
            ranking_list.append((i+1, doc_id))
    if dense:
        faiss_index, faiss_id_converter = dense
        with torch.no_grad():
            query_vector = MODEL.encode(query, convert_to_tensor=True, show_progress_bar=False)
            query_vector = query_vector.cpu().numpy().astype("float32")
            qnorm = np.linalg.norm(query_vector)
            if qnorm > 0:
                query_vector = query_vector / qnorm
            query_vector = np.ascontiguousarray(query_vector[None, :])
        D, I = faiss_index.search(query_vector, 10000)
        for i, idx in enumerate(I[0]):
            if idx == -1:
                print(f"Warning: idx -1 found in FAISS results (index {i})")
                break
            doc_id = faiss_id_converter[str(idx)]
            ranking_list.append((i, doc_id))

    return ranking_list

def main():
    
    queries = get_query(CFG["data_config"]["paths"]["yahoo_json_query_dir"])
    doc_ids, bm25, faiss_id_converter, faiss_index, passages = load_indexes(CFG['data_config']['paths']['wikipedia_psg_path'])
    os.makedirs(CFG["data_config"]["paths"]["annotation_pos_neg_json"], exist_ok=True)
    for query_id, query_body in tqdm(queries.items()):
        query = query_body["q_title"] + query_body["q_body"]
        query_wiki_url = query_body["ans_url"]
        wiki_title = urllib.parse.unquote(query_wiki_url.split("/")[-1])
        print(query_wiki_url)
        print(wiki_title)
        print(query)
        if len(wiki_title.strip()) == 0:
            print("Empty wiki title, skipping...")
            continue
        if "#" in wiki_title:   
            wiki_title = wiki_title.split("#")[0]
        if " " in wiki_title and "(" in wiki_title:
            wiki_title = wiki_title.replace(" ", "_")
        wiki_title = wiki_title.split()[0]
        positive_doc_id_and_info = get_positive_documents(wiki_title, passages)
        
        negative_ranking_list_bm = get_negative_ranking_list(query, bm25=bm25, dense=None)
        negative_ranking_list_faiss = get_negative_ranking_list(query, bm25=None, dense=(faiss_index, faiss_id_converter))
        with open(os.path.join(CFG["data_config"]["paths"]["annotation_pos_neg_json"], f"{query_id}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "query_id": query_id,
                "query_title": query_body["q_title"],
                "query_url": query_wiki_url,
                "query": query,
                "positive_docs": positive_doc_id_and_info,
                "negative_docs_bm25": negative_ranking_list_bm,
                "negative_docs_faiss": negative_ranking_list_faiss,
                "answer_body": query_body["ans_body"]
            }, f, ensure_ascii=False, indent=4)
        print("-"*50)


if __name__ == "__main__":
    main()
