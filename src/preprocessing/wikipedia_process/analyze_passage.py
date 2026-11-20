#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze Wikipedia passages dataset.
"""


import json
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
from src.preprocessing.wikipedia_process.passage_text import (
    MAX_PASSAGE_LENGTH, MIN_PASSAGE_LENGTH, format_passage_text,
)
from src.utils.configs import load_configs
CFG = load_configs()

def worker_for_load_jsonl(jsonl_path):
    tmp_length = []
    tmp2_length = []
    stats = {"total": 0, "short": 0, "long": 0, "total_length": 0, "articles" : set(), "filtered_articles": set()}
    with open(jsonl_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            psg_id = data["id"]
            article_id = psg_id.split("_", 1)[0]
            text_length = len(format_passage_text(data))
            if text_length > MAX_PASSAGE_LENGTH: # 95 percentile
                stats["long"] += 1
            elif text_length < MIN_PASSAGE_LENGTH: # 5 percentile
                stats["short"] += 1
            else:
                stats["filtered_articles"].add(article_id)
                tmp2_length.append(text_length)
            tmp_length.append(text_length)
            stats["total"] += 1
            stats["total_length"] += text_length
            stats["articles"].add(article_id)
    print(dict(stats, articles=len(stats["articles"]), filtered_articles=len(stats["filtered_articles"])))
    return tmp_length, tmp2_length, stats

def print_passage_lengths(passage_lengths):
    print("Total passages:", len(passage_lengths))
    if not passage_lengths:
        return
    print("Average passage length (in characters):", np.mean(passage_lengths))
    print("Max passage length (in characters):", np.max(passage_lengths))
    print("Min passage length (in characters):", np.min(passage_lengths))
    print("Passage length percentiles (in characters):")
    for percentile in [5, 25, 50, 75, 90, 95, 99]:
        print(f"  {percentile}th percentile: {np.percentile(passage_lengths, percentile)}")

def main():
    pattern = os.path.join(CFG['data_config']['paths']['wikipedia_psg_path'], "*.jsonl")
    print(pattern)
    data_paths = glob.glob(pattern)
    passage_lengths = []
    filtered_passage_lengths = []
    stats = {"total": 0, "short": 0, "long": 0, "total_length": 0}
    articles = set()
    filtered_articles = set()
    with Pool(cpu_count()) as pool:
        for lengths, filtered_lengths, file_stats in pool.imap_unordered(worker_for_load_jsonl, data_paths):
            passage_lengths.extend(lengths)
            filtered_passage_lengths.extend(filtered_lengths)
            for key in stats:
                stats[key] += file_stats[key]
            articles.update(file_stats["articles"])
            filtered_articles.update(file_stats["filtered_articles"])
    stats["articles"] = len(articles)
    stats["filtered_articles"] = len(filtered_articles)
    print_passage_lengths(passage_lengths)
    print("Passage stats:", stats)
    print("After filtering short and long passages:")
    print_passage_lengths(filtered_passage_lengths)
    print("Filtered passage stats:", dict(
        stats, total=len(filtered_passage_lengths), total_length=sum(filtered_passage_lengths),
        short=0, long=0, articles=len(filtered_articles),
    ))

if __name__ == "__main__":
    main()
