#Using Qwen3 to implement dense retrieval for arxiv paper search.

import os
import mlx.core as mx
import heapq
import sqlite3


from arxiv_embedding import embed_batch
from qwen3_embedding import load_qwen3_for_embeddings

def semantic_search(query: str, top_k: int = 5):
    model_name = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    model, tokenizer = load_qwen3_for_embeddings(model_name)
    
    # Batch embed queries
    query_embedding = embed_batch(model, tokenizer, [query])
    
    heap = []

    batch_number = 0
    while True:
        np_path = f"data/doc_embeddings/emb_b{batch_number}.npy"
        if os.path.exists(np_path):
            index_embeddings:mx.array = mx.load(np_path)
            ids = index_embeddings[:, 0]
            embeddings = index_embeddings[:, 1:]
        else:
            break
        
        # Use cosine similarity to find top-k
        scores = ((query_embedding @ embeddings.T) * 100).reshape(-1)
        mx.eval(scores)
        for idx, score in enumerate(scores):
            item = (score, ids[idx])
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            else:
                if score > heap[0][0]:
                    heapq.heappushpop(heap, item)
        batch_number += 1
        
    if not heap:
        print("No embedding batches found.")
        return []
    top_candidates = sorted(heap, key=lambda x: x[0], reverse=True)
    print("Top-k results:")
    
    conn = sqlite3.connect("data/arxiv_metadata.db")
    cur = conn.cursor()
    
    with open("data/arxiv_concatenated.jsonl", "r", encoding="utf-8") as file:
        for score, table_id in top_candidates:
            print(f"Score: {score:.4f}, Table ID: {int(table_id)}")
            cur.execute("SELECT title, abstract FROM arxiv_metadata WHERE id = ?", (int(table_id),))
            print(cur.fetchone(), "\n")
    
    conn.close()
    return top_candidates
        

if __name__ == "__main__":
    # semantic_search(f"""query: Support vector machine""", top_k=5)
    # semantic_search(f"""query: Papers on machine learning""", top_k=5)
