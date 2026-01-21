import json
import glob
import os
import re
from tqdm import tqdm
import mwparserfromhell
import orjson
from multiprocessing import Pool, cpu_count
import numpy as np
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
            title = " ".join(data["section_titles"])
            title = re.sub(r'_dup\d+', '', title)
            text = data["text"]
        
            text = mwparserfromhell.parse(text).strip_code()
            text = re.sub(r'[\ud800-\udfff]', '', text)
            text = text.replace("\x00", "")  # remove null characters
            text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
            all_text = title + "\n" + text
            text_length = len(all_text)
            if text_length > 1094: # ../passage_split/src/stats.py の95パーセンタイルに基づく
                stats["long"] += 1
            elif text_length < 29: # ../passage_split/src/stats.py の5パーセンタイルに基づく
                stats["short"] += 1
            else:
                if not isinstance(text, str) or not isinstance(title, str):
                    print(title, type(title))
                    print(text, type(text))
                    raise ValueError("Invalid text or title")
                stats["filtered_articles"].add(psg_id.split("_")[0])
                tmp2_length.append(text_length)
            tmp_length.append(text_length)
            stats["total"] += 1
            stats["total_length"] += text_length
            stats["articles"].add(psg_id.split("_")[0])
    stats["articles"] = len(stats["articles"])
    stats["filtered_articles"] = len(stats["filtered_articles"])
    print(stats)
    return tmp_length, tmp2_length, stats

print(os.path.join(CFG['data_config']['paths']['wikipedia_psg_path'], "*.jsonl"))
data_paths = glob.glob(os.path.join(CFG['data_config']['paths']['wikipedia_psg_path'], "*.jsonl"))
passage_lengths = []
stats = {"total": 0, "short": 0, "long": 0, "total_length": 0, "articles" : 0, "filtered_articles": 0}
filtered_passage_lengths = []
with Pool(cpu_count()) as pool:
    for length_li, length_li2, stats in pool.imap_unordered(worker_for_load_jsonl, data_paths):
        passage_lengths.extend(length_li)
        filtered_passage_lengths.extend(length_li2)
        for key in stats:
            stats[key] += stats[key]

print("Total passages:", len(passage_lengths))
print("Average passage length (in characters):", np.mean(passage_lengths))
print("Max passage length (in characters):", np.max(passage_lengths))
print("Min passage length (in characters):", np.min(passage_lengths))
print("Passage length percentiles (in characters):")
for percentile in [5, 25, 50, 75, 90, 95, 99]:
    print(f"  {percentile}th percentile: {np.percentile(passage_lengths, percentile)}")
print("Passage stats:", stats)

print("After filtering short and long passages:")
print("Total passages:", len(filtered_passage_lengths))
print("Average passage length (in characters):", np.mean(filtered_passage_lengths))
print("Max passage length (in characters):", np.max(filtered_passage_lengths))
print("Min passage length (in characters):", np.min(filtered_passage_lengths))
print("Passage length percentiles (in characters):")
for percentile in [5, 25, 50, 75, 90, 95, 99]:
    print(f"  {percentile}th percentile: {np.percentile(filtered_passage_lengths, percentile)}")
print("Filtered passage stats:", stats)
