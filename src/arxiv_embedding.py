from __future__ import annotations

import json
import os
import mlx.core as mx
from typing import Optional, List, Any
from qwen3_embedding import load_qwen3_for_embeddings
import sqlite3

# Insert metadata into SQLite DB
def preprocessing(limit: Optional[int] = None):
	processed = 0
	skipped = 0
	failed_inserts = []
	inserted = 0

	db_path = "data/arxiv_metadata.db"
	conn = sqlite3.connect(db_path)
	cur = conn.cursor()
	cur.execute("""
	CREATE TABLE IF NOT EXISTS arxiv_metadata (
		id INTEGER PRIMARY KEY,
		doc_id TEXT,
		title TEXT,
		authors TEXT,
		abstract TEXT
	)
	""")

	with open("data/arxiv-metadata-oai-snapshot.json", "r", encoding="utf-8") as inf:
		for line in inf:
			if limit is not None and processed >= limit:
				break
			processed += 1

			line = line.strip()
			if not line:
				skipped += 1
				continue

			try:
				rec = json.loads(line)
			except Exception:
				skipped += 1
				continue

			doc_id = rec.get("id")
			title = (rec.get("title") or "").strip()
			abstract = (rec.get("abstract") or "").strip()
			authors = (rec.get("authors") or "").strip()

			# Insert into SQLite DB
			try:
				cur.execute(
					"INSERT OR REPLACE INTO arxiv_metadata (doc_id, title, authors, abstract) VALUES (?, ?, ?, ?)",
					(doc_id, title, authors, abstract)
				)
				inserted += 1
			except Exception as e:
				failed_inserts.append((doc_id, str(e)))
				print(f"SQLite insert error for doc_id {doc_id}: {e}")

			if processed % 100000 == 0:
				print(f"Processed {processed:,} lines, skipped {skipped:,}, inserted {inserted:,}")

	conn.commit()
	conn.close()
	print(f"Finished. Processed {processed:,} lines.")
	print(f"Total skipped records: {skipped}")
	print(f"Total failed inserts: {len(failed_inserts)}")
	if failed_inserts:
		print("Failed insert details (first 10):")
		for doc_id, err in failed_inserts[:10]:
			print(f"doc_id: {doc_id}, error: {err}")

def embed_batch(model, tokenizer, batch_docs: List[str]):
    # Tokenize
    batch = tokenizer(batch_docs, max_length=256, padding=True, truncation=True, return_tensors="mlx")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Get hidden states
    hidden_states = model(input_ids)

    # Pool last token
    seq_lens = mx.sum(attention_mask, axis=1) - 1
    bsz = hidden_states.shape[0]
    last_token_indices = mx.maximum(seq_lens, 0)
    pooled = hidden_states[mx.arange(bsz), last_token_indices]

    # Normalize
    norm = mx.linalg.norm(pooled, ord=2, axis=-1, keepdims=True)
    normalized = pooled / mx.maximum(norm, 1e-9)

    mx.eval(normalized)

    return normalized


# Embed documents
def embed_documents(limit: Optional[int] = None, batch_size: int = 128, output_prefix: str = "data/doc_embeddings/emb"):
    print("\nBatch processing with last token pooling (saving MLX tensors)")
    model_name = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    model, tokenizer = load_qwen3_for_embeddings(model_name)

    processed = 0
    embedded = 0
    docs: List[str] = []
    batch_idx = 0

    db_path = "data/arxiv_metadata.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, doc_id, title, authors, abstract FROM arxiv_metadata")

    def _clean(s: str) -> str:
        return s.replace("\t", " ").replace("\r", " ").replace("\n", " ")

    ids = []
    for row in cur:
        if limit is not None and processed >= limit:
            break
        processed += 1
        id, _, title, authors, abstract = row
        ids.append(id)
        metadata = (
            f"authors: {_clean(authors)}\n"
            f"title: {_clean(title)}\n"
            f"abstract: {_clean(abstract)}\n\n"
        )
        docs.append(metadata)
        # If batch is full, process it
        if len(docs) >= batch_size:
            normalized = embed_batch(model, tokenizer, docs)
            embedded += len(docs)
            docs = []
            # Save MLX tensor
            save_key = f"{output_prefix}_b{batch_idx}"
            try:
                mx.save(save_key, mx.concatenate([mx.array(ids, dtype=mx.bfloat16).reshape(-1, 1), normalized], axis=1))
                print(f"Saved MLX tensor for batch {batch_idx} -> key: {save_key}")
            except Exception as e:
                print(f"Warning: failed to mx.save batch {batch_idx}: {e}")
            print(f"Processed batch of {batch_size} docs -> embeddings shape: {normalized.shape}")
            batch_idx += 1
            ids = []
    # Process any remaining docs
    if docs:
        normalized = embed_batch(model, tokenizer, docs)
        embedded += len(docs)
        save_key = f"{output_prefix}_b{batch_idx}"
        try:
            mx.save(save_key, mx.concatenate([mx.array(ids, dtype=mx.bfloat16).reshape(-1, 1), normalized], axis=1))
            print(f"Saved MLX tensor for batch {batch_idx} -> key: {save_key}")
        except Exception as e:
            print(f"Warning: failed to mx.save batch {batch_idx}: {e}")
        print(f"Processed final batch of {len(docs)} docs -> embeddings shape: {normalized.shape}")
    
    conn.close()
    print(f"Finished embedding. Processed {processed} input rows, embedded {embedded} documents.")

if __name__ == "__main__":
    # preprocessing()
    embed_documents(limit=1000000)