"""Build a tiny random Llama locally (no network) for shard/pipeline tests."""

from __future__ import annotations

from pathlib import Path

TINY_MODEL_DIR = Path("/tmp/loom-tiny-llama")


def ensure_tiny_model(path: Path = TINY_MODEL_DIR, num_layers: int = 6) -> Path:
    """Create the model once; reuse it on later runs."""
    if (path / "model.safetensors").exists():
        return path
    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig

    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    _save_byte_tokenizer(path)
    return path


def _save_byte_tokenizer(path: Path) -> None:
    """A 256-symbol byte-level tokenizer matching the model's vocab_size."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=None,
        eos_token=None,
        unk_token=None,
        pad_token=None,
    )
    tokenizer.save_pretrained(path)
