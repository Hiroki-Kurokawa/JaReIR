#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Get Yahoo! Japan QA pairs that contain a single Wikipedia URL in the answer body.
"""

import pandas as pd
from urlextract import URLExtract
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
import os
from JaReIR.src.utils.configs import load_configs
CFG = load_configs("configs")
os.makedirs(CFG["data_config"]["paths"]["yahoo_json_query_dir"], exist_ok=True)

def init_worker():
    global url_extractor
    url_extractor = URLExtract()

def worker(row):
    global url_extractor
    qid = row["qid"]
    q_title = row["q_title"]
    q_body = row["q_body"]
    ans_id = row["ans_id"]
    ans_body = row["ans_body"]
    category = row["category"]
    
    if pd.isna(q_body) or pd.isna(ans_body):
        return None
    if not row["ans_status"]:
        return None
    if str(row["has_image_y"]) == "1":
        return None 

    q_body_urls = url_extractor.find_urls(q_body)
    q_title_urls = url_extractor.find_urls(q_title)
    if len(q_body_urls) > 0 or len(q_title_urls) > 0:
        return None

    ans_urls = url_extractor.find_urls(ans_body)
    if len(ans_urls) != 1:
        return None
    
    ans_url = ans_urls[0]
    if ans_url.count("http") > 1:
        return None
    ans_url = ans_url.split()[0].rstrip("。、，．）】」』>")
    if ans_url == ans_body.strip():
        return None

    if "ja.m.wikipedia.org" in ans_url or "ja.wikipedia.org" in ans_url:
        output_path = os.path.join(CFG["data_config"]["paths"]["yahoo_json_query_dir"], f"{qid}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "qid": qid,
                "category": category,
                "category_path": row['category_path'],
                "q_title": q_title,
                "q_body": q_body,
                "ans_id": ans_id,
                "ans_body": ans_body,
                "ans_url": ans_url
            }, f, ensure_ascii=False, indent=4)
    return None

if __name__ == "__main__":
    query_df = pd.read_csv(os.path.join(CFG["data_config"]["paths"]["yahoo_data_path"], "all_query_df.csv"))
    answer_df = pd.read_csv(os.path.join(CFG["data_config"]["paths"]["yahoo_data_path"], "all_answer_df.csv"))
    query_answer_pair_df = pd.merge(query_df, answer_df, on="qid", how="inner")
    
    os.makedirs("yahoo_json_query", exist_ok=True)
    
    rows = [row for _, row in query_answer_pair_df.iterrows()]
    
    with Pool(cpu_count(), initializer=init_worker) as p:
        list(tqdm(p.imap_unordered(worker, rows, chunksize=2000), total=len(rows)))
