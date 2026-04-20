#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Translation script for BRIGHT dataset.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import pickle
import random
import re
from collections.abc import Mapping

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


# =========================
# Configuration
# =========================
SAVE_DIR = "./BRIGHT_translate/"
MODEL_NAME = "cyberagent/CAT-Translate-0.8b"
MODEL_MAX_LEN = 8192

USER_TEMPLATE = "Translate the following English text into Japanese.\n\n{}"

SAFETY_MARGIN = 128

QUERY_MAX_BATCH_ITEMS = 1024
QUERY_MAX_BATCH_PROMPT_TOKENS = 128_000

DOC_MAX_BATCH_ITEMS = 2 ** 16
DOC_MAX_BATCH_PROMPT_TOKENS = 10 ** 18

QUERY_MAX_OUTPUT_TOKENS = 2048

_SPLIT_PARAGRAPH_RE = re.compile(r"\n\s*\n")



def seed_everything(seed = 1) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_bright_dataset():
    """
    Load BRIGHT dataset using Hugging Face Datasets library.
    """

    examples = load_dataset("xlangai/BRIGHT", "examples")
    documents = load_dataset("xlangai/BRIGHT", "documents")
    return examples, documents

def build_queries_and_qrels(examples_ds):
    """
    Build queries and qrels from the examples dataset.
    """

    queries = {}
    qrels = {}

    for split_name, ds in examples_ds.items():
        for ex in ds:
            qid = f"{split_name}:{ex['id']}"
            gold_ids = [f"{split_name}:{gid}" for gid in ex["gold_ids"]]

            queries[qid] = ex["query"]
            qrels[qid] = {gid: 1 for gid in gold_ids}

    return queries, qrels


def build_corpus(documents_ds):
    corpus = {}

    for split_name, ds in documents_ds.items():
        for doc in ds:
            doc_id = f"{split_name}:{doc['id']}"
            corpus[doc_id] = doc["content"]

    return corpus

def make_messages(text):
    return [
        {
            "role": "user",
            "content": USER_TEMPLATE.format(text),
        }
    ]


def _normalize_token_ids(ids):
    """
    Normalize tokenizer output into a flat Python ``list[int]``.
    """

    if hasattr(ids, "tolist") and not isinstance(ids, list):
        ids = ids.tolist()

    if isinstance(ids, Mapping):
        if "input_ids" in ids:
            ids = ids["input_ids"]
        elif "prompt_token_ids" in ids:
            ids = ids["prompt_token_ids"]

    if isinstance(ids, tuple):
        ids = list(ids)

    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])

    normalized = []
    for tok in ids:
        if isinstance(tok, str):
            if tok.lstrip("-").isdigit():
                tok = int(tok)
        elif hasattr(tok, "item") and callable(getattr(tok, "item")):
            tok = tok.item()
        normalized.append(int(tok))

    return normalized


def build_prompt_token_ids(tokenizer, text):
    ids = tokenizer.apply_chat_template(
        make_messages(text),
        add_generation_prompt=True,
    )
    return _normalize_token_ids(ids)


def prompt_overhead_tokens(tokenizer):
    return len(build_prompt_token_ids(tokenizer, ""))


def _decode_ids(tokenizer, ids):
    ids = _normalize_token_ids(list(ids))
    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

def split_long_text_fast(tokenizer, text, max_source_tokens):
    """
    Split source text into chunks using paragraph packing.
    """
    stripped = text.strip()
    if not stripped:
        return [""]

    paragraphs = [p.strip() for p in _SPLIT_PARAGRAPH_RE.split(text) if p.strip()]
    if not paragraphs:
        paragraphs = [stripped]

    sep = "\n\n"
    sep_len = len(tokenizer.encode(sep, add_special_tokens=False))

    chunks = []
    current_parts = []
    current_len = 0

    def flush() -> None:
        nonlocal current_parts, current_len
        if current_parts:
            chunks.append(sep.join(current_parts))
            current_parts = []
            current_len = 0

    for para in paragraphs:
        para_ids = tokenizer.encode(para, add_special_tokens=False)
        para_len = len(para_ids)

        if para_len > max_source_tokens:
            flush()
            start = 0
            while start < para_len:
                end = min(start + max_source_tokens, para_len)
                piece = _decode_ids(tokenizer, para_ids[start:end]).strip()
                if piece:
                    chunks.append(piece)
                start = end
            continue

        add_len = para_len if not current_parts else (sep_len + para_len)
        if current_parts and current_len + add_len > max_source_tokens:
            flush()

        current_parts.append(para)
        current_len += para_len if len(current_parts) == 1 else add_len

    flush()
    return chunks


def iter_prompt_items(
    texts,
    tokenizer,
    *,
    max_source_tokens,
    max_prompt_tokens,
):
    for owner_idx, text in enumerate(texts):
        for initial_chunk in split_long_text_fast(
            tokenizer,
            text,
            max_source_tokens=max_source_tokens,
        ):
            stack = [initial_chunk]

            while stack:
                cur = stack.pop()
                prompt_ids = build_prompt_token_ids(tokenizer, cur)

                if len(prompt_ids) <= max_prompt_tokens:
                    yield owner_idx, prompt_ids
                    continue

                raw_ids = tokenizer.encode(cur, add_special_tokens=False)

                mid = len(raw_ids) // 2
                left = _decode_ids(tokenizer, raw_ids[:mid]).strip()
                right = _decode_ids(tokenizer, raw_ids[mid:]).strip()

                if right:
                    stack.append(right)
                if left:
                    stack.append(left)

def _atomic_pickle_dump(obj, path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def _atomic_json_dump(obj, path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, path)


def translate_texts(
    texts,
    cache_prefix,
    *,
    tokenizer,
    llm,
    max_source_tokens,
    max_output_tokens,
    max_batch_items,
    max_batch_prompt_tokens,
):
    os.makedirs(SAVE_DIR, exist_ok=True)
    cache_dir = os.path.join(SAVE_DIR, f"cache_{cache_prefix}")
    os.makedirs(cache_dir, exist_ok=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_output_tokens,
    )

    all_outputs = [[] for _ in range(len(texts))]

    batch = []
    batch_prompt_tokens = 0
    batch_idx = 0

    progress = tqdm(desc=f"Translating ({cache_prefix})", unit="chunk")

    def flush_batch() -> None:
        nonlocal batch, batch_prompt_tokens, batch_idx

        if not batch:
            return

        shard_path = os.path.join(cache_dir, f"batch_{batch_idx:06d}.pkl")

        if os.path.exists(shard_path):
            with open(shard_path, "rb") as f:
                translated_batch = pickle.load(f)
        else:
            prompt_token_ids = [_normalize_token_ids(prompt_ids) for _, prompt_ids in batch]

            first_bad = next(
                (
                    (batch_i, tok_i, type(tok).__name__, tok)
                    for batch_i, ids in enumerate(prompt_token_ids)
                    for tok_i, tok in enumerate(ids)
                    if not isinstance(tok, int)
                ),
                None,
            )

            outputs = llm.generate(
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                use_tqdm=True,
            )

            translated_batch = [
                (out.outputs[0].text.strip() if out.outputs else "")
                for out in outputs
            ]
            _atomic_pickle_dump(translated_batch, shard_path)

        for (owner_idx, _), translated in zip(batch, translated_batch):
            all_outputs[owner_idx].append(translated)

        progress.update(len(batch))
        batch = []
        batch_prompt_tokens = 0
        batch_idx += 1

    for owner_idx, prompt_ids in iter_prompt_items(
        texts,
        tokenizer,
        max_source_tokens=max_source_tokens,
        max_prompt_tokens=MODEL_MAX_LEN - max_output_tokens - SAFETY_MARGIN,
    ):
        prompt_len = len(prompt_ids)

        if batch and (
            len(batch) >= max_batch_items
            or batch_prompt_tokens + prompt_len > max_batch_prompt_tokens
        ):
            flush_batch()

        batch.append((owner_idx, prompt_ids))
        batch_prompt_tokens += prompt_len

    flush_batch()
    progress.close()

    merged_outputs = ["\n\n".join(parts).strip() for parts in all_outputs]
    assert len(merged_outputs) == len(texts)
    return merged_outputs


def translate_mapping(
    src_mapping,
    cache_prefix,
    *,
    tokenizer,
    llm,
    max_source_tokens,
    max_output_tokens,
    max_batch_items,
    max_batch_prompt_tokens,
):
    sorted_items = sorted(src_mapping.items(), key=lambda x: x[0])
    src_texts = [text for _, text in sorted_items]

    translated_texts = translate_texts(
        src_texts,
        cache_prefix=cache_prefix,
        tokenizer=tokenizer,
        llm=llm,
        max_source_tokens=max_source_tokens,
        max_output_tokens=max_output_tokens,
        max_batch_items=max_batch_items,
        max_batch_prompt_tokens=max_batch_prompt_tokens,
    )

    return {
        sorted_items[i][0]: translated_texts[i]
        for i in range(len(sorted_items))
    }



def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    examples_ds, documents_ds = load_bright_dataset()
    queries, qrels = build_queries_and_qrels(examples_ds)
    corpus = build_corpus(documents_ds)

    _atomic_json_dump(qrels, os.path.join(SAVE_DIR, "qrels.json"))

    prompt_overhead = prompt_overhead_tokens(tokenizer)
    usable_context = MODEL_MAX_LEN - prompt_overhead - SAFETY_MARGIN

    query_max_output_tokens = min(QUERY_MAX_OUTPUT_TOKENS, usable_context // 2)
    query_max_source_tokens = usable_context - query_max_output_tokens

    doc_max_output_tokens = usable_context // 2
    doc_max_source_tokens = usable_context - doc_max_output_tokens

    queries_ja = translate_mapping(
        queries,
        cache_prefix="queries",
        tokenizer=tokenizer,
        llm=llm,
        max_source_tokens=query_max_source_tokens,
        max_output_tokens=query_max_output_tokens,
        max_batch_items=QUERY_MAX_BATCH_ITEMS,
        max_batch_prompt_tokens=QUERY_MAX_BATCH_PROMPT_TOKENS,
    )
    _atomic_json_dump(queries_ja, os.path.join(SAVE_DIR, "queries_ja.json"))

    corpus_ja = translate_mapping(
        corpus,
        cache_prefix="corpus",
        tokenizer=tokenizer,
        llm=llm,
        max_source_tokens=doc_max_source_tokens,
        max_output_tokens=doc_max_output_tokens,
        max_batch_items=DOC_MAX_BATCH_ITEMS,
        max_batch_prompt_tokens=DOC_MAX_BATCH_PROMPT_TOKENS,
    )
    _atomic_json_dump(corpus_ja, os.path.join(SAVE_DIR, "corpus_ja.json"))


if __name__ == "__main__":
    seed_everything(1)

    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.92,
    )
    tokenizer = llm.get_tokenizer()
    main()
