
import json
import pandas as pd
import glob

result_paths = glob.glob("analyzer/lancers_extracted_data/*.json")
reject_worker_names = ["temp", ]
result_dict = {"worker_id" : [], "qid" : [], "did": [], "label": [], "reason": [], "time" : []}
times = {"qid" : [], "length" : [], "time" : []}
task_paths = glob.glob("data/wiki_psg_tasks/*.csv")
qid_to_path = {}
for task_path in task_paths:
    df = pd.read_csv(task_path)
    qid = str(df.iloc[0, 0])
    qid_to_path[qid] = task_path
for path in result_paths:
    with open(path, "r") as f:
        results = json.load(f)
    if results["worker_id"] in reject_worker_names:
        continue
    for result in results["results"]:
        
        if len(result["qid"]) == 0:
            task_csv_name = results["csv_name"]
            print("Searching for QID for task csv name:", task_csv_name)
            for qid in qid_to_path.keys():
                if "/" + task_csv_name in qid_to_path[qid]:
                    result["qid"] = qid
                    break
        assert len(result["qid"]) > 0, f"QID not found in result: {result}, file: {path}"

        result_dict["worker_id"].append(results["worker_id"])
        result_dict["qid"].append(result["qid"])
        result_dict["did"].append(result["did"])
        if result["relevance"] == "related" and result["can_describe"] == "yes":
            result_dict["label"].append("relevant")
        elif result["relevance"] == "related" and result["can_describe"] == "direct_answering":
            result_dict["label"].append("direct_answering")
        elif result["relevance"] == "related":
            result_dict["label"].append("not_descriptive")
        elif result["relevance"] == "doc_inappropriate":
            result_dict["label"].append("doc_inappropriate")
        else:
            result_dict["label"].append("not_relevant")
        if len(result["reason_text"]) == 0:
            result_dict["reason"].append("No reason")
        else:
            result_dict["reason"].append(result["reason_text"])
        result_dict["time"].append(results["effective_elapsed_seconds"])
df = pd.DataFrame(result_dict)
print(df)
df.to_csv("analyzer/data/annotation_result.csv", index=False)
