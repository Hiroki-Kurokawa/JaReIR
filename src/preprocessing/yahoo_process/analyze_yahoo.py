import pandas as pd
import numpy as np

query_df = pd.read_csv("/mnt/LSTA6/bay1/kurokawa/yahoo_df/all_query_df.csv")
answer_df = pd.read_csv("/mnt/LSTA6/bay1/kurokawa/yahoo_df/all_answer_df.csv")
print(query_df.columns)
print(answer_df.columns)
number_of_qids = set()
query_lengths = []
answer_lengths = []
category = set()
for i, row in query_df.iterrows():
    number_of_qids.add(row["qid"])
    q_title = row["q_title"]
    q_body = row["q_body"]
    if pd.isna(q_body):
        q_body = ""
    query_lengths.append(len(q_title) + len(q_body))
    category.add(row["category_path"].split(">")[-1])

for i, row in answer_df.iterrows():
    ans_body = row["ans_body"]
    answer_lengths.append(len(ans_body))

print("Number of unique qids:", len(number_of_qids))
print("Average query length (in characters):", np.mean(query_lengths))
print("Max query length (in characters):", np.max(query_lengths))
print("Min query length (in characters):", np.min(query_lengths))
print("Average answer length (in characters):", np.mean(answer_lengths))
print("Max answer length (in characters):", np.max(answer_lengths))
print("Min answer length (in characters):", np.min(answer_lengths))
print("Number of unique categories:", len(category))
