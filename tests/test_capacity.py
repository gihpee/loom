"""ShardCapacity: the explicit capacity input for Phase-1."""

from math import floor

from conftest import GIB, make_model_info_kwargs

from loom.planning import ModelInfo, ShardCapacity


def test_capacity_matches_manual_formula():
    mi = ModelInfo(**make_model_info_kwargs())
    quota = 24 * GIB
    cap = ShardCapacity.from_model_info(mi, vram_quota_bytes=quota)

    per_layer = mi.decoder_layer_io_bytes(roofline=False)
    assert cap.decoder_layer_capacity() == floor(floor(quota * 0.5) / per_layer)

    # Embedding discount is applied once when tied, twice when untied.
    with_embed = cap.decoder_layer_capacity(include_input_embed=True)
    with_both = cap.decoder_layer_capacity(include_input_embed=True, include_lm_head=True)
    assert with_embed == floor((floor(quota * 0.5) - mi.embedding_io_bytes) / per_layer)
    assert with_both == with_embed  # tie_embedding=True -> lm_head is free

    untied = ShardCapacity.from_model_info(
        ModelInfo(**{**make_model_info_kwargs(), "tie_embedding": False}),
        vram_quota_bytes=quota,
    )
    assert untied.decoder_layer_capacity(
        include_input_embed=True, include_lm_head=True
    ) == floor((floor(quota * 0.5) - 2 * mi.embedding_io_bytes) / per_layer)


def test_quota_slicing_scales_capacity():
    """Half the quota -> (roughly) half the layers: broker-controlled capacity."""
    mi = ModelInfo(**make_model_info_kwargs())
    full = ShardCapacity.from_model_info(mi, vram_quota_bytes=48 * GIB)
    half = ShardCapacity.from_model_info(mi, vram_quota_bytes=24 * GIB)
    assert 0 < half.decoder_layer_capacity() <= full.decoder_layer_capacity()
    assert half.decoder_layer_capacity() >= full.decoder_layer_capacity() // 2 - 1


def test_mlx_bit_factor_applied():
    kwargs = make_model_info_kwargs()
    kwargs["mlx_param_bytes_per_element"] = 4  # 2x factor vs param_bytes_per_element=2
    mi = ModelInfo(**kwargs)
    cuda_cap = ShardCapacity.from_model_info(mi, vram_quota_bytes=24 * GIB, device="cuda")
    mlx_cap = ShardCapacity.from_model_info(mi, vram_quota_bytes=24 * GIB, device="mlx")
    assert mlx_cap.per_layer_param_bytes == 2 * cuda_cap.per_layer_param_bytes
    assert mlx_cap.decoder_layer_capacity() <= cuda_cap.decoder_layer_capacity() // 2 + 1


def test_kv_budget():
    mi = ModelInfo(**make_model_info_kwargs())
    cap = ShardCapacity.from_model_info(mi, vram_quota_bytes=10 * GIB)
    assert cap.kv_cache_budget_bytes() == floor(10 * GIB * 0.3)
    assert cap.per_layer_kv_cache_memory(0) is None
    assert cap.per_layer_kv_cache_memory(5) == floor(cap.kv_cache_budget_bytes() / 5)
