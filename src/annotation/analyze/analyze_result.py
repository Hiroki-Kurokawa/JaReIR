#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze annotation results for inter-rater reliability.
"""


import numpy as np
import pandas as pd
from collections import defaultdict
import json
import os
import glob
from src.utils.configs import load_configs
CFG = load_configs()
np.random.seed(1)
def irr_5way(y1, y2, labels=None):
    assert len(y1) == len(y2)
    if labels is None:
        labels = sorted(set(y1) | set(y2))
    idx = {l:i for i, l in enumerate(labels)}
    k = len(labels)
    n = len(y1)

    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(y1, y2):
        cm[idx[a], idx[b]] += 1

    po = np.trace(cm) / n  # % agreement

    p1 = cm.sum(axis=1) / n
    p2 = cm.sum(axis=0) / n
    pe_kappa = float((p1 * p2).sum())
    kappa = (po - pe_kappa) / (1 - pe_kappa) if (1 - pe_kappa) > 0 else np.nan

    # Gwet AC1 (nominal)
    pi = (p1 + p2) / 2.0
    pe_ac1 = float((pi * (1 - pi)).sum() / (k - 1)) if k > 1 else 0.0
    ac1 = (po - pe_ac1) / (1 - pe_ac1) if (1 - pe_ac1) > 0 else np.nan

    return {
        "labels": labels,
        "confusion_matrix": cm,
        "percent_agreement": po,
        "cohen_kappa": kappa,
        "gwet_ac1": ac1,
    }

def main():
    results = pd.read_csv("analyzer/data/annotation_result.csv")
    print("Total annotations:", len(results))

    query_body_paths = glob.glob(os.path.join(CFG["data_config"]["paths"]["annotation_pos_neg_json"], "*.json"))
    qid_did_to_org = {}
    for path in query_body_paths:
        with open(path, "r") as f:
            data = json.load(f)
            qid = str(data["query_id"])
            did_pos = set()
            for pos in data["positive_docs"]:
                did = str(pos["doc_id"])
                did_pos.add(did)
            TOP_K = min(10, len(did_pos))
            for i, neg in enumerate(data["negative_docs_bm25"]):
                if i >= TOP_K // 2:
                    break
    
                did = str(neg[1])
                if did in did_pos:
                    did_pos.remove(did)
            for i, neg in enumerate(data["negative_docs_faiss"]):
                if i >= (TOP_K + 1) // 2:
                    break
                did = str(neg[1])
                if did in did_pos:
                    did_pos.remove(did)
            for did in did_pos:
                qid_did_to_org[(qid, did)] = True

    qid_did_groups = results.groupby(['qid', 'did'])
    labels_by_worker = {}   # worker_id -> {unit_id: label}
    qiddid_to_id = {}
    count = 0
    label_pairs = defaultdict(lambda : {"count" : 0, "examples" : []})

    already_done = defaultdict(dict)
    worker_id = {w: i for i, w in enumerate(results['worker_id'].unique())}
    print(results)
    print("Unique worker count:", len(worker_id))
    task_per_worker_counts = {}
    for worker in worker_id:
        task_per_worker = results[results['worker_id'] == worker]
        task_per_worker_counts[worker] = len(task_per_worker)
    print("Average tasks per worker:", np.mean(list(task_per_worker_counts.values())))
    match_counts = []
    y1, y2 = [], []
  
    is_positive_counts = {"positive": 0, "negative": 0}
    for (qid, did), group in qid_did_groups:

        unit_key = f"{qid}_{did}"
        if unit_key not in qiddid_to_id:
            qiddid_to_id[unit_key] = count
            count += 1
        unit_id = qiddid_to_id[unit_key]

        labels_list = []
        workers_list = []
        tmp_label_name = []
        for _, row in group.iterrows():
            worker = row['worker_id']
            lab = row['label']
            tmp_label_name.append(lab)
            
          
            if lab == "not_descriptive":
                labv = 0
            elif lab == "not_relevant":
                labv = 1
            elif lab == "doc_inappropriate":
                labv = 2
            elif lab == "direct_answering":
                labv = 4
            else:
                labv = 5
                if (str(qid), str(did)) in qid_did_to_org:
                    is_positive_counts["positive"] += 1
                else:
                    is_positive_counts["negative"] += 1
            # store
            if worker not in labels_by_worker:
                labels_by_worker[worker] = {}
            labels_by_worker[worker][unit_id] = labv

            labels_list.append(labv)
            workers_list.append(worker)
        if len(labels_list) != 2:
            continue

        if labels_list[0] == labels_list[1]:
            match_counts.append(1)
        else:
            match_counts.append(0)
        tmp_label_name.sort()
        label_key = ":".join(tmp_label_name)
        label_pairs[label_key]["count"] += 1
        if len(label_pairs[label_key]["examples"]) < 5:
            label_pairs[label_key]["examples"].append(f"{qid},{did}")
        already_done[qid][did] = (labels_list, workers_list)
        y1.append(labels_list[0])
        y2.append(labels_list[1])


    print("Total units considered:", count)

    print("Is positive counts:", is_positive_counts)
    print("Cohen's kappa and Gwet AC1 for all pairs:")
    irr_results = irr_5way(y1, y2)
    print(irr_results)
    print("Aggregate agreement rate:", np.mean(match_counts))
    # other stats from original
    avg_time = results["time"].mean()
    std_time = results["time"].std()
    print("Average annotation time (seconds):", avg_time)
    print("Standard deviation of annotation time (seconds):", std_time)
    print("Annotation time range (seconds):", (results["time"].min(), results["time"].max()))
    print("Cost Estimation", avg_time * 496 * 2 * 1100 / 3600 ) # 496: query
    with open("analyzer/label_pair_details.json", "w", encoding="utf-8") as f:
        json.dump(label_pairs, f, indent=4, ensure_ascii=False)
    # unanimous count
    already_count = 0
    for qid in already_done:
        for did in already_done[qid]:
            labs, workers_list = already_done[qid][did]
            if len(set(labs)) == 1:
                already_count += 1
    print("Number of queries with unanimous agreement:", already_count)

if __name__ == "__main__":
    main()
