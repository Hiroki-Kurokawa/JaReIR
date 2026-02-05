# JaReIR: A Reasoning-intensive Information Retrieval Benchmark in Japanese using Q\&A Data and Wikipedia Passages

## Overview
This repository contains the code and data for JaReIR, a reasoning-intensive information retrieval benchmark in Japanese. JaReIR is designed to evaluate the performance of IR models on complex queries that require reasoning over multiple pieces of information.
## Contents
- `Dockerfile` `pyproject.toml` : For setting up the development environment using Docker and uv.
- `configs` : Configuration files for data path. You can set the paths for Yahoo! Chiebukuro data, Wikipedia dump data, cache directory, results save directory, and so on.
- `data` : Contains the JaReIR dataset. Due to Yahoo! Chiebukuro data usage restrictions, 
the Yahoo! Chiebukuro data is not included in this repository. Please refer to the original source to obtain it. And also, Wikipedia passages are not included due to their large size.
  - `JaReIR-data` : JaReIR dataset version 1.
    - `excluded_dids.json` : List of document IDs excluded from the dataset.
    - `qrels.json` : Relevance judgments for the dataset.
    - `query_difficulty.json` : Difficulty levels of the queries.
        - The difference in query count versus target annotation count is due to exact match filtering.
        - Difficulty levels are categorized as "easy", "medium", and "hard". 
    - Also, `queries.json` does not exist here due to Yahoo! Chiebukuro data usage restrictions.
- `report` : Contains the report and analysis of the benchmark results.
- `src` : Source code for data processing, annotation and model evaluation.
  - `annoation` : Code for annotating the dataset.
  - `model_evaluation` : Code for evaluating IR models on the JaReIR benchmark.
  - `preprocessing` : Code for preprocessing the Yahoo! Chiebukuro data and Wikipedia passages.
    - `annotation_process` : Code for processing annotated data.
    - `wikipedia_process` : Code for processing Wikipedia passages.
    - `yahoo_process` : Code for processing Yahoo! Chiebukuro data.
- `utils` : For loading configs from yaml files.
## Process for Annotation Data Creation
0. Get Yahoo! Chiebukuro data and Wikipedia dump data from their original sources.
  - 0.1 Yahoo! Chiebukuro data: https://www.nii.ac.jp/dsc/idr/yahoo/yahoo-user.html
    - ⚠️ Warning: Yahoo! Chiebukuro data was terminated in 2025/12. Please check the usage policy before using it.
  - 0.2 Wikipedia dump data: https://dumps.wikimedia.org/jawiki/
    - We use the dump data as of 2025/08/01.
  - 0.3 Set the URL for annotation server.
    - You need to set the URL for the annotation server in `src/annotation/lancers/lancers.html` and `.env` file.
    - `.env` file example:
      ```
      ANNOTATION_BASE_URL=https://your_annotation_server_url
      ```
  - 0.4 Create a virtual environment using Docker and Poetry.
    - Build the Docker image:
      ```bash
      docker build -t jareir_env .
      ```
    - Run the Docker container:
      ```bash
      docker run -it -v YOUR_LOCAL_PATH:/app --ipc=host --gpus=all jareir_env /bin/bash
      ```
    - Inside the Docker container, install dependencies using uv:
      ```bash
      uv sync
      ```
    - Then, you can run the code using uv run command.

1. Preprocess Yahoo! Chiebukuro data using the code in `src/preprocessing/yahoo_process`.
  - Preprocessing steps include extracting question-answer pairs, filtering based on URL patterns, and removing image-based questions.
  - `analyze_yahoo.py` : Analyze the Yahoo! Chiebukuro dataset to understand its characteristics.
  - `collect_yahoo_qa_pairs.py` : Collect question-answer pairs from the Yahoo! Chiebukuro dataset.
2. Preprocess Wikipedia passages using the code in `src/preprocessing/wikipedia_process`.
  - Preprocessing steps include extracting passages from Wikipedia dump data and cleaning the text.
  - `get_wikipedia_passage.py` : Extract passages from Wikipedia dump data.
  - `analyze_passage.py` : Analyze the extracted Wikipedia passages to understand their characteristics.
3. Create positive-negative passage pairs for annotation using the code in `src/preprocessing/annotation_process/get_pos_neg_candidate.py`.
4. Create annotation data using the code in `src/preprocessing/annotation_process/create_csv.py`.
5. Remove inappropriate questions using the code in `src/preprocessing/annotation_process/remove_query.py`.

## Annotation Process
The annotation process involves several steps to ensure the quality and relevance of the dataset. The code for annotation is located in the `src/annotation` directory. 
- `server.py` : Code for setting up the annotation server.
  - The server allows annotators to log in, view tasks, and submit their annotations.
  - Run the server using the command:
    ```bash
    uv run gunicorn -w 9 --threads 2 -b 0.0.0.0:5001 src.annotation.server:app
    ```
- `templates/` : HTML templates for the annotation interface, including guidelines and examples.
- `lancers/` : Contains files related to the Lancers annotation platform.
- `analyze/` : Code for analyzing the annotation results.
  - If you get the annotation result in `output/` directory, you can analyze it using the code in `src/annotation/analyze`.
    - `parse_result.py` : Parse the annotation results from JSON files to a structured format.
    - `analyze_result.py` : Analyze the parsed results to compute inter-annotator agreement and other statistics.
    - `create_qrel.py` : Create qrel files from the annotation results for evaluation purposes.
- The annotation process can be executed using the following steps:
  1. Start the annotation server (use `server.py`).
  2. Set up tasks on the Lancers platform using the provided HTML templates (See `lancers/SETTINGS.md`).
  3. Annotators complete the tasks on the Lancers platform.
  3. Collect the annotation results from the `output/` directory.
    - Our annotation outputs are not included in this repository due to privacy concerns.
  4. Analyze the results using the provided analysis code.

## Model Evaluation
- The model evaluation code is located in the `src/model_evaluation` directory.
- The main evaluation script is `retrieval.py`, which evaluates various IR models on the JaReIR benchmark.
  - If you run:
    ```bash
    uv run -m src.model_evaluation.retrieval
    ```
  - it will evaluate different models and save the results in the specified directory. 
    
- `models/` : Contains implementations of different IR models, including BM25, Semantic Retriever. Also code includes code for query expansion using LLMs and reranking.
