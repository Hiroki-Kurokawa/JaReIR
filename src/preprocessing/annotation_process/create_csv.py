#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create CSV files for annotation tasks from positive and negative passage lists.
"""

import json
import glob
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import orjson
import re
from multiprocessing import Pool, cpu_count
from JaReIR.src.utils.configs import load_configs
CFG = load_configs("configs")
np.random.seed(1)

def add_item(all_data, item, is_postive):
    if item[1] in all_data["did"]:
        print("Dupplicated did found in the same file:", item[1], item[0])
        return 

    all_data["qid"].append(item[0])
    all_data["did"].append(item[1])
    all_data["topic"].append(item[2])
    all_data["psg"].append(item[3])
    all_data["is_positive"].append(is_postive)
    return all_data

def task_for_load_jsonl(jsonl_path):
    passages = {}
    with open(jsonl_path, "rb") as f:
        for line in tqdm(f):
            data = orjson.loads(line)
            psg_id = data["id"]
            psg = data["title"].strip() + "\n" + data["text"].strip()
            passages[psg_id] = {"title": data["title"], "psg": psg}
    return passages

def main():
    pos_neg_json_paths = glob.glob(os.path.join(CFG["data_config"]["paths"]["annotation_pos_neg_json"], "*.json"))
    passages = {}
    psg_jsonl_paths = glob.glob(os.path.join(CFG["data_config"]["paths"]["wikipedia_psg_path"], "*.jsonl"))
    with Pool(cpu_count()) as pool:
        for passage_dict in pool.imap_unordered(task_for_load_jsonl, psg_jsonl_paths):
            passages.update(passage_dict)
    query_count = 0
    stats = {"total": 0, "skipped_due_to_no_positive": 0, "skipped_due_to_insufficient_data": 0}
    example_pairs = set()
    example_pairs.add(("934056_1", "10235767330"))
    example_pairs.add(("65265_10", "11263424416"))

    os.makedirs(CFG["data_config"]["paths"]["first_annotation_csv_paths"], exist_ok=True)
    for path in tqdm(pos_neg_json_paths):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data = {"qid": [], "did": [], "topic": [], "psg": [], "is_positive": []}
            positive_docs = data["positive_docs"]
            if len(positive_docs) == 0:
                stats["skipped_due_to_no_positive"] += 1
                continue
            np.random.shuffle(positive_docs)
            new_positive_docs = []
            pos_count = 0
            for doc in positive_docs:
                psg = passages[doc["doc_id"]]["psg"]
                psg = re.sub(r'\{\|.*?\|\}', '', psg, flags=re.DOTALL)
                if len(psg) == 0:
                    continue
                if (doc["doc_id"], data["query_id"]) in example_pairs:
                    continue
                new_positive_docs.append(doc)
                pos_count += 1
                if pos_count == CFG["data_config"]["annotation"]["num_max_positive_passages"]:
                    break
            positive_docs = new_positive_docs
            if len(positive_docs) == 0:
                stats["skipped_due_to_no_positive"] += 1
                continue

            if len(positive_docs) < CFG["data_config"]["annotation"]["num_max_positive_passages"]:
                less_count = CFG["data_config"]["annotation"]["num_max_positive_passages"] - len(positive_docs)
            else:
                less_count = 0
            positive_doc_ids = [doc["doc_id"] for doc in positive_docs]
            assert len(positive_doc_ids) == len(set(positive_doc_ids)), f"Duplicate positive doc ids found: {positive_doc_ids}"
            psg_doc_ids = set(positive_doc_ids)
            positive_doc_items = [(data["query_id"], doc["doc_id"], data["query"], re.sub(r'\{\|.*?\|\}', '', doc["info"]["psg"], flags=re.DOTALL)) for doc in positive_docs]
            count = 0
            negative_doc_items = []
            did_neg_pos = set()
            for neg_doc_infos in data["negative_docs_bm25"]:
                if neg_doc_infos[1] in psg_doc_ids:
                    did_neg_pos.add(neg_doc_infos[1])
                    continue
                if (neg_doc_infos[1], data["query_id"]) in example_pairs:
                    continue

                neg_doc_id = neg_doc_infos[1]
                neg_doc_item = passages[neg_doc_id]["psg"]
                neg_doc_item = re.sub(r'\{\|.*?\|\}', '', neg_doc_item, flags=re.DOTALL)
                if len(neg_doc_item) == 0:
                    continue
                negative_doc_items.append((data["query_id"], neg_doc_id, data["query"], neg_doc_item))
                psg_doc_ids.add(neg_doc_id)
                count += 1
                if count == CFG["data_config"]["annotation"]["num_max_negative_passages"] + less_count // 2:
                    break
            count = 0
            
            for neg_doc_infos in data["negative_docs_faiss"]:
                if neg_doc_infos[1] in psg_doc_ids:
                    did_neg_pos.add(neg_doc_infos[1])
                    continue
                if (neg_doc_infos[1], data["query_id"]) in example_pairs:
                    continue
                # judge = judge_text(data["query"], passages[neg_doc_infos[1]]["psg"])
                # if not judge:
                #     continue
                neg_doc_id = neg_doc_infos[1]
                neg_doc_item = passages[neg_doc_id]["psg"]
                neg_doc_item = re.sub(r'\{\|.*?\|\}', '', neg_doc_item, flags=re.DOTALL)
                if len(neg_doc_item) == 0:
                    continue
                negative_doc_items.append((data["query_id"], neg_doc_id, data["query"], neg_doc_item))
                psg_doc_ids.add(neg_doc_id)
                count += 1
                if count == CFG["data_config"]["annotation"]["num_max_negative_passages"] + (less_count - less_count // 2):
                    break

            for item in positive_doc_items:
                tmp = add_item(all_data, item, 1)
                if tmp is not None:
                    all_data = tmp
            for item in negative_doc_items:
                tmp = add_item(all_data, item, 0)
                if tmp is not None:
                    all_data = tmp
            
            all_data_df = pd.DataFrame(all_data)
            if len(all_data_df) < CFG["data_config"]["annotation"]["num_max_negative_passages"] * 2 + CFG["data_config"]["annotation"]["num_max_positive_passages"]:
                print("Not enough data found in the file:", path, len(all_data_df))
                stats["skipped_due_to_insufficient_data"] += 1
                continue
            assert len(all_data_df) == CFG["data_config"]["annotation"]["num_max_negative_passages"] * 2 + CFG["data_config"]["annotation"]["num_max_positive_passages"], f"Data is not equal to TOPK*2 and pos...{len(all_data_df)} in {path}, positive: {len(positive_doc_items)}, negative: {len(negative_doc_items)}"
            all_data_df = all_data_df.sample(frac=1, random_state=1).reset_index(drop=True) # shuffle
            for neg_id in did_neg_pos:
                all_data_df.loc[all_data_df['did'] == neg_id, 'is_positive'] = -1

            all_data_df.to_csv(os.path.join(CFG["data_config"]["annotation"]["first_annotation_csv_paths"], f"{query_count}.csv"), index=False)
            query_count += 1
            stats["total"] += 1
    print("Processing stats:", stats)

if __name__ == "__main__":
    main()
