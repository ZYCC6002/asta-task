import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Type, Optional, List, Any
import importlib
from transformers import AutoTokenizer
from mlx_lm.utils import load_model, get_model_path

# cloned from https://gist.github.com/mzbac/67d92c2cfe8bcf75579ac55144d1174f
def get_qwen3_embedding_classes(config: dict) -> Tuple[Type[nn.Module], Type]:

    model_type = config.get("model_type", "").lower()

    if model_type not in ["qwen3"]:
        raise ValueError(f"This loader only supports qwen3 models, got: {model_type}")

    # Import the appropriate module
    try:
        qwen_module = importlib.import_module(f"mlx_lm.models.{model_type}")
    except ImportError:
        raise ImportError(
            f"Could not import module for model type '{model_type}'. "
            "Ensure mlx_lm.models.qwen3 is available."
        )

    # Create embedding model class
    class Qwen3EmbeddingModel(qwen_module.Model):
        def __init__(self, args):
            super().__init__(args)
            # Remove lm_head for embeddings
            if hasattr(self, "lm_head"):
                delattr(self, "lm_head")

        def __call__(
            self,
            inputs: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        ) -> mx.array:
            """Return hidden states instead of logits."""
            return self.model(inputs, mask, cache)

    return Qwen3EmbeddingModel, qwen_module.ModelArgs


def load_qwen3_for_embeddings(
    model_path: str,
) -> Tuple[nn.Module, Any]:

    # Get the model path
    model_path_resolved = get_model_path(model_path)

    # Load model with custom embedding classes
    model, _ = load_model(
        model_path=model_path_resolved,
        get_model_classes=get_qwen3_embedding_classes,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )

    return model, tokenizer


# Example usage
if __name__ == "__main__":
    # Example 1: Basic loading and embedding extraction
    print("Example 1: Loading Qwen3 model for embeddings")

    model_name = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    model, tokenizer = load_qwen3_for_embeddings(model_name)

    # Tokenize text
    text = "This is a sample text for embedding extraction."
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    input_ids = mx.array(inputs["input_ids"])

    # Get embeddings
    hidden_states = model(input_ids)
    print(f"Hidden states shape: {hidden_states.shape}")

    # Example 2: Batch processing with pooling
    print("\nExample 2: Batch processing with last token pooling")

    texts = [
        "First sentence for embedding.",
        "Second sentence with different length.",
        "Third one.",
    ]

    # Tokenize batch
    batch = tokenizer(
        texts, max_length=128, padding=True, truncation=True, return_tensors="mlx"
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Get embeddings
    hidden_states = model(input_ids)

    # Pool last token
    sequence_lengths = mx.sum(attention_mask, axis=1) - 1
    batch_size = hidden_states.shape[0]
    last_token_indices = mx.maximum(sequence_lengths, 0)
    pooled = hidden_states[mx.arange(batch_size), last_token_indices]

    # Normalize
    norm = mx.linalg.norm(pooled, ord=2, axis=-1, keepdims=True)
    normalized = pooled / mx.maximum(norm, 1e-9)

    print(f"Final embeddings shape: {normalized.shape}")

    # Example 3: Similarity calculation with instruct queries
    print("\nExample 3: Semantic similarity with instruction-based queries")

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    # Test similarity calculation
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "how much protein should a female eat"),
        get_detailed_instruct(task, "summit define"),
    ]

    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]

    # Process all texts together
    all_texts = queries + documents
    batch = tokenizer(
        all_texts, max_length=128, padding=True, truncation=True, return_tensors="mlx"
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Get embeddings
    hidden_states = model(input_ids)

    # Pool last token
    sequence_lengths = mx.sum(attention_mask, axis=1) - 1
    batch_size = hidden_states.shape[0]
    last_tokens = hidden_states[mx.arange(batch_size), sequence_lengths]

    # Normalize
    norm = mx.linalg.norm(last_tokens, ord=2, axis=-1, keepdims=True)
    embeddings = last_tokens / mx.maximum(norm, 1e-9)
    mx.eval(embeddings)

    # Split queries and documents using MLX arrays directly
    query_embeddings = embeddings[: len(queries)]
    doc_embeddings = embeddings[len(queries) :]

    # Calculate similarity scores using MLX
    scores = (query_embeddings @ doc_embeddings.T) * 100
    mx.eval(scores)

    print("\nSimilarity scores (scaled by 100):")
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}.")
        for j, doc in enumerate(documents):
            print(f"  Doc {j+1}: {float(scores[i, j]):.2f} - {doc[:60]}...")