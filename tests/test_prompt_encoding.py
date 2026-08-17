"""The head stage must turn chat messages into token IDS, whatever the
tokenizer hands back.

Real failure this pins down: transformers 5 made `return_dict=True` the default
for `apply_chat_template`, so it returns a BatchEncoding — and iterating that
yields its KEYS. The head passed `["input_ids", "attention_mask"]` into
`torch.tensor(...)` and every single request died with
`ValueError: too many dimensions 'str'`.

It went unnoticed because the tiny test model ships no chat template, so tests
only ever took the plain-concatenation path. One of the cases below uses a
tokenizer WITH a template.
"""

import sys
from pathlib import Path

import pytest
from make_tiny_model import ensure_tiny_model

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from loom_worker.shard.server import _as_token_ids, _encode_chat  # noqa: E402

MESSAGES = [{"role": "user", "content": "hello there"}]
TEMPLATE = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(ensure_tiny_model()))


# ------------------------------------------------------ shape normalisation
def test_accepts_a_plain_list():
    assert _as_token_ids([1, 2, 3]) == [1, 2, 3]


def test_unwraps_a_batch_encoding():
    """transformers 5 default: a dict-like whose iteration yields key names."""

    class FakeBatchEncoding(dict):
        pass

    encoded = FakeBatchEncoding(input_ids=[[5, 6, 7]], attention_mask=[[1, 1, 1]])
    assert _as_token_ids(encoded) == [5, 6, 7]
    # The bug in one line: naive iteration gives strings, not ids.
    assert list(encoded) == ["input_ids", "attention_mask"]


def test_unwraps_a_batch_of_one_and_a_tensor():
    assert _as_token_ids([[8, 9]]) == [8, 9]
    assert _as_token_ids(torch.tensor([[8, 9]])) == [8, 9]


def test_rejects_strings_instead_of_silently_passing_them_on():
    with pytest.raises((TypeError, ValueError)):
        _as_token_ids(["input_ids", "attention_mask"])
    with pytest.raises((TypeError, ValueError)):
        _as_token_ids([])


# ---------------------------------------------------------- real tokenizers
def test_templated_tokenizer_yields_ids_torch_can_use(tokenizer):
    tokenizer.chat_template = TEMPLATE
    try:
        ids = _encode_chat(tokenizer, MESSAGES)
    finally:
        tokenizer.chat_template = None
    assert ids and all(isinstance(i, int) for i in ids)
    # The call that used to explode.
    assert torch.tensor([ids], dtype=torch.long).shape == (1, len(ids))


def test_tokenizer_without_a_template_still_works(tokenizer):
    assert tokenizer.chat_template is None
    ids = _encode_chat(tokenizer, MESSAGES)
    assert ids and all(isinstance(i, int) for i in ids)


def test_falls_back_when_the_template_misbehaves(tokenizer):
    """A broken template must degrade to concatenation, not kill the request."""

    class Broken:
        chat_template = TEMPLATE

        def apply_chat_template(self, *a, **k):
            return {"attention_mask": [[1, 1]]}  # no input_ids at all

        def encode(self, text):
            return tokenizer.encode(text)

    ids = _encode_chat(Broken(), MESSAGES)
    assert ids and all(isinstance(i, int) for i in ids)


# ------------------------------------------------------- generation limits
def test_default_token_budget_fits_a_reasoning_model():
    """A 128-token default truncates Qwen3 mid-<think> and looks like a crash."""
    from loom_worker.shard import server

    assert server.DEFAULT_MAX_TOKENS >= 1024


def test_template_kwargs_reach_the_template(tokenizer):
    """`chat_template_kwargs` is how a caller turns reasoning off per request."""
    seen = {}

    class Recording:
        chat_template = TEMPLATE

        def apply_chat_template(self, messages, **kwargs):
            seen.update(kwargs)
            # Render with the template passed explicitly: the shared fixture's
            # own attribute is cleared by the test above.
            return tokenizer.apply_chat_template(
                messages, chat_template=TEMPLATE, add_generation_prompt=True, tokenize=True
            )

    ids = _encode_chat(Recording(), MESSAGES, {"enable_thinking": False})
    assert seen.get("enable_thinking") is False
    assert ids and all(isinstance(i, int) for i in ids)


def test_console_exposes_the_token_budget_and_thinking_switch():
    html = (Path(__file__).resolve().parent.parent / "src/loom/api/admin_ui.html").read_text()
    assert "c-maxtok" in html and "max_tokens" in html
    assert "chat_template_kwargs" in html and "enable_thinking" in html
    # A capped answer must say so instead of looking like a broken pipeline.
    assert "finish_reason=length" in html
