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