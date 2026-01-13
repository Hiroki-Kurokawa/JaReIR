import pandas as pd
import os
import glob
import json
from collections import defaultdict
INPUT_PATH = "analyzer/data/annotation_result.csv"
SAVE_DIR = "analyzer/data/j-bright"
os.makedirs(SAVE_DIR, exist_ok=True)
qrel_path = os.path.join(SAVE_DIR, "qrels.json")
query_path = os.path.join(SAVE_DIR, "query.json")



df = pd.read_csv(INPUT_PATH)
print(df)
qid_to_query = {}
query_body_paths = glob.glob("/mnt/LSTA6/bay1/kurokawa/neg_pos_list/json_data_1212/*.json")
for path in query_body_paths:
    with open(path, "r") as f:
        data = json.load(f)
        qid_to_query[str(data["query_id"])] = data["query"]

qrels = defaultdict(dict)
queries = {}
excluded_dids = set()
for qid_did, group in df.groupby(["qid", "did"]):
    qid = str(qid_did[0])
    did = str(qid_did[1])
    labels = group["label"].tolist()
    if len(labels) != 2:
        continue
    if labels[0] == labels[1]:
        label = labels[0]
        if label == "relevant":
            qrels[qid][did] = 2
        elif label == "direct_answering":
            qrels[qid][did] = 1
        elif label == "doc_inappropriate":
            excluded_dids.add(did)
            continue
        else:
            qrels[qid][did] = 0
        queries[qid] = qid_to_query[qid]

assert len(queries) == len(qrels), f"Mismatch in queries and qrels ({len(queries)} vs {len(qrels)})"

queries_diff = {}
for qid in qrels.keys():
    if 1 in qrels[qid].values():
        queries_diff[qid] = "easy_query"
    elif 2 in qrels[qid].values():
        queries_diff[qid] = "hard_query"
    else:
        queries_diff[qid] = "exclude_query"

for qid, dids in qrels.items():
    for did in list(dids.keys()):
        if qrels[qid][did] == 2:
            qrels[qid][did] = 1

with open(qrel_path, "w") as f:
    json.dump(qrels, f, indent=4, ensure_ascii=False)
with open(query_path, "w") as f:
    json.dump(queries, f, indent=4, ensure_ascii=False)

with open(os.path.join(SAVE_DIR, "query_difficulty.json"), "w") as f:
    json.dump(queries_diff, f, indent=4, ensure_ascii=False)
with open(os.path.join(SAVE_DIR, "excluded_dids.json"), "w") as f:
    json.dump(list(excluded_dids), f, indent=4, ensure_ascii=False)
