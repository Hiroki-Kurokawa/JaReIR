# JaReIR: A Reasoning-intensive Information Retrieval Benchmark in Japanese using Q\&A Data and Wikipedia Passages

## Overview
This repository contains the code and data for JaReIR, a reasoning-intensive information retrieval benchmark in Japanese. JaReIR is designed to evaluate the performance of IR models on complex queries that require reasoning over multiple pieces of information.
## Contents
- `configs` : Configuration files for data path.
- `data` : Contains the JaReIR dataset. Due to Yahoo! Chiebukuro data usage restrictions, the Yahoo! Chiebukuro data is not included in this repository. Please refer to the original source to obtain it. And also, Wikipedia passages are not included due to their large size.
- `report` : Contains the report and analysis of the benchmark results.
- `src` : Source code for data processing, annotation and model evaluation.
  - `annoation` : Code for annotating the dataset.
  - `model_evaluation` : Code for evaluating IR models on the JaReIR benchmark.
  - `preprocessing` : Code for preprocessing the Yahoo! Chiebukuro data and Wikipedia passages.
    - `annotation_process` : Code for processing annotated data.
    - `wikipedia_process` : Code for processing Wikipedia passages.
    - `yahoo_process` : Code for processing Yahoo! Chiebukuro data.
- `utils` : For loading configs from yaml files.
## Process for annotation data creation
1. Preprocess Yahoo! Chiebukuro data using the code in `src/preprocessing/yahoo_process`.
2. Preprocess Wikipedia passages using the code in `src/preprocessing/wikipedia_process`.
3. Create positive-negative passage pairs for annotation using the code in `src/preprocessing/annotation_process/get_pos_neg_candidate.py`.
4. Remove inappropriate questions using the code in `src/preprocessing/annotation_process/remove_query.py`.
5. Create annotation data using the code in `src/preprocessing/annotation_process/create_csv.py`.

## Annotation Process
The annotation process involves several steps to ensure the quality and relevance of the dataset. The code for annotation is located in the `src/annotation` directory. 
- `server.py` : Code for setting up the annotation server.
- `templates/` : HTML templates for the annotation interface, including guidelines and examples.
- `lancers/` : Contains files related to the Lancers annotation platform.
- `analye/` : Code for analyzing the annotation results.

## Model Evaluation
- The model evaluation code is located in the `src/model_evaluation` directory.
- The main evaluation script is `retrieval.py`, which evaluates various IR models on the JaReIR benchmark.
- `models/` : Contains implementations of different IR models, including BM25, Semantic Retriever. Also code includes code for query expansion using LLMs and reranking.
 
## TODO:
src/preprocessing/wikipedia_processでconvert_to_vecの文章カットの処理がないので追加すること
