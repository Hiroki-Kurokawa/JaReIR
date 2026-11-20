#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dense retriever using SentenceTransformers and FAISS.
"""

import json
from pathlib import Path
from filelock import FileLock
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv
from src.utils.configs import load_configs
CFG = load_configs()
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

class SemanticRetriever:

    def __init__(self, model_path_or_name, corpus, index_dir: str, force: bool = False):
        self.index_dir = index_dir
        self.model = SentenceTransformer(model_path_or_name, device="cuda", cache_folder=CFG["model_config"]["paths"]["model_cache_dir"])
        self.model.half().eval()
        self.index = None
        self.doc_ids = []
        self.build_index(corpus, index_dir, force)

    def build_index(self, corpus, index_dir, force = False):

        os.makedirs(index_dir, exist_ok=True)
        save_path = os.path.join(index_dir, "faiss.index")
        ids_path = Path(index_dir) / "doc_ids.json"
        with FileLock(save_path + ".lock"):
            if not force and os.path.isfile(save_path) and self._load_index(corpus, save_path, ids_path):
                return
            self._build_new_index(corpus, save_path, ids_path)

    def _load_index(self, corpus, save_path, ids_path):
        try:
            with ids_path.open(encoding="utf-8") as f:
                doc_ids = json.load(f)
            if not isinstance(doc_ids, list) or len(doc_ids) != len(corpus) or set(doc_ids) != corpus.keys():
                return False
        except (FileNotFoundError, ValueError, TypeError):
            return False
        index = faiss.read_index(save_path)
        if index.ntotal != len(doc_ids):
            return False
        self.index = index
        self.doc_ids = doc_ids
        return True

    def _build_new_index(self, corpus, save_path, ids_path):
        print("Building model index... at", save_path)
        dim = self.model.get_sentence_embedding_dimension()
        faiss_index = faiss.IndexHNSWFlat(dim, 32)
        faiss_index.hnsw.efConstruction = 200
        faiss_index.hnsw.efSearch = 128
        doc_ids = list(corpus)
        batch_size = max(1, 2 ** 16 * torch.cuda.device_count())
        for start in tqdm(range(0, len(doc_ids), batch_size)):
            batch = [corpus[doc_id] for doc_id in doc_ids[start:start + batch_size]]
            with torch.no_grad():
                embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
                embeddings = embeddings.astype("float32")
            faiss_index.add(embeddings)

        # An interrupted write must never leave the previous mapping attached to new vectors.
        ids_path.unlink(missing_ok=True)
        faiss.write_index(faiss_index, save_path)
        with ids_path.open("w", encoding="utf-8") as f:
            json.dump(doc_ids, f, ensure_ascii=False)
        self.index = faiss_index
        self.doc_ids = doc_ids
        print("Model index built successfully.")
    

    def retrieve(self, queries, top_k = 100):
        responses = {}
        query_texts = [query for _, query in queries.items()]
        query_embeddings = np.vstack([self.__get_embedding(text) for text in query_texts]).astype("float32")
  
        distances, indices = self.index.search(query_embeddings, top_k)
      
        for i, (dists, inds) in enumerate(zip(distances, indices)):
            qid = list(queries.keys())[i]
            result_dict = {}
            for dist, ind in zip(dists, inds):
                if ind == -1:
                    continue
                doc_id = self.doc_ids[ind]
                dist = 1 / (1 + dist) 
                result_dict[doc_id] = dist.item()
            responses[qid] = result_dict
        return responses
    
    def __get_embedding(self, text):
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding
