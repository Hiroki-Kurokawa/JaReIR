#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Remove queries for annotation based on manual inspection.
"""
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import mwparserfromhell
import json
from JaReIR.src.utils.configs import load_configs
CFG = load_configs("configs")

json_mode = False
remove_qids = set()
already_done = set()
with open("data/remove_qids.txt", "r") as f:
    for line in f:
        if line.strip() == "":
            continue
        remove_qids.add(int(line.strip()))
with open("data/already_done.txt", "r") as f:
    for line in f:
        #print(line.strip(), line)
        if line.strip() == "":
            continue
        #print(line)
        already_done.add(line.strip())

paths = glob.glob("data/wiki_psg_tasks_1214/*.csv")
for path in tqdm(paths):
    df = pd.read_csv(path)
    qid = df["qid"][0]
    if str(qid) in already_done or qid in remove_qids:
        continue
    qid = df["qid"][0]
    topic = df["topic"][0]
    print(f"qid: {qid}, topic: {topic}")
    user_input = input("このクエリを除外しますか？ (y/n): ")
    if user_input.lower() == 'y':
        remove_qids.add(qid)
    with open("data/already_done.txt", "a") as f:
        f.write(f"{qid}\n")
    with open("data/remove_qids.txt", "w") as f:
        for rqid in remove_qids:
            f.write(f"{rqid}\n")
print(f"除外するクエリID: {remove_qids}")
new_saved_dir = CFG["data_config"]["paths"]["final_annotation_csv_path"]
os.makedirs(new_saved_dir, exist_ok=True)
counter = 0
for path in tqdm(paths):
    df = pd.read_csv(path)
    qid = df["qid"][0]
    if qid in remove_qids:
        continue
    df["psg"] = df["psg"].apply(lambda x : mwparserfromhell.parse(x).strip_code())
    df.to_csv(os.path.join(new_saved_dir, f"{counter}.csv"), index=False)
    counter += 1
print(f"クエリ除外後のデータを {new_saved_dir} に保存しました。")
