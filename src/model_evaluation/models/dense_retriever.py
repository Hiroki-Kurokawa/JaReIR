#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dense retriever using SentenceTransformers and FAISS.
"""

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
        if os.path.exists(save_path) and not force:
            self.index = faiss.read_index(save_path)
            self.doc_ids = [doc_id for doc_id in corpus.keys()]
            return
        print("Building model index... at", save_path)
        dim = self.model.get_sentence_embedding_dimension()
        faiss_index = faiss.IndexHNSWFlat(dim, 32)
        faiss_index.hnsw.efConstruction = 200
        faiss_index.hnsw.efSearch = 128
        batch_size = 2 ** 16
        batch = []
        for doc_id, doc in tqdm(corpus.items()):
            self.doc_ids.append(doc_id)
            batch.append(doc)
            if len(batch) >= batch_size * torch.cuda.device_count():
                with torch.no_grad():
                    embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
                    embeddings = embeddings.astype("float32")
                    faiss_index.add(embeddings)
                batch = []

        if len(batch) > 0:
            with torch.no_grad():
                embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
                embeddings = embeddings.astype("float32")
            faiss_index.add(embeddings)
        
        faiss.write_index(faiss_index, save_path)
        self.index = faiss_index
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
                doc_id = self.doc_ids[ind]
                dist = 1 / (1 + dist) 
                result_dict[doc_id] = dist.item()
            responses[qid] = result_dict
        return responses
    
    def __get_embedding(self, text):
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding
