#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
import faiss
import os
import logging
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from huggingface_hub import login
os.environ["HF_HOME"] = "/mnt/LSTA6/bay7/kurokawa/.cache"
login(token="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")
logging.basicConfig(level=logging.INFO)
class SemanticRetriever:

    def __init__(self, model_path_or_name, corpus, index_dir: str, force: bool = False):
        self.index_dir = index_dir
        self.model = SentenceTransformer(model_path_or_name, device="cuda", cache_folder="/mnt/LSTA6/bay7/kurokawa/.cache")
        self.model.half().eval()
        self.index = None
        self.doc_ids = []
        self.build_index(corpus, index_dir, force)

    def build_index(self, corpus, index_dir, force = False):
        """
        BERTのインデックスを構築する
        """

        os.makedirs(index_dir, exist_ok=True)
        save_path = os.path.join(index_dir, "faiss.index")
        if os.path.exists(save_path) and not force:
            self.index = faiss.read_index(save_path)
            self.doc_ids = [doc_id for doc_id in corpus.keys()]
            logging.info(f"Loaded existing BERT index from {save_path}.")
            return
        logging.info(f"Building BERT index at {save_path}...")
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
                    embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=True, batch_size=128, normalize_embeddings=True)
                    embeddings = embeddings.astype("float32")
                    faiss_index.add(embeddings)
                batch = []

        if len(batch) > 0:
            with torch.no_grad():
                embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=True, batch_size=128, normalize_embeddings=True)
                embeddings = embeddings.astype("float32")
            faiss_index.add(embeddings)
        
        faiss.write_index(faiss_index, save_path)
        self.index = faiss_index
        logging.info("Model index built successfully.")
    

    def retrieve(self, queries, top_k = 100):
        """
        検索を行う関数．スコアが高い順に top_k 件を返す
        """
        responses = {}
        query_texts = [query for _, query in queries.items()]
        query_embeddings = np.vstack([self.__get_embedding(text) for text in query_texts]).astype("float32")
        if None in query_embeddings:
            logging.error("One or more query embeddings could not be computed.")
            return None
        distances, indices = self.index.search(query_embeddings, top_k)
        #print(distances, indices)
        for i, (dists, inds) in enumerate(zip(distances, indices)):
            qid = list(queries.keys())[i]
            result_dict = {}
            for dist, ind in zip(dists, inds):
                doc_id = self.doc_ids[ind]
                dist = 1 / (1 + dist)  # 距離をスコアに変換
                result_dict[doc_id] = dist.item()
            responses[qid] = result_dict
        return responses
    
    def __get_embedding(self, text):
        try:
            with torch.no_grad():
                embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        except RuntimeError as e:
            logging.error(f"Error encoding text: {e}")
            return None
        return embedding
