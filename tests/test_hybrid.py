from src.retrieval.hybrid_engine import expand_query_tokens, tokenize_bandura


def test_tokenize_keeps_self_efficacy_intact() -> None:
    tokens = tokenize_bandura("低自我效能会阻止掌握经验发生")
    assert "自我效能" in tokens
    assert "掌握经验" in tokens


def test_query_expansion_maps_mastery_aliases() -> None:
    expanded = expand_query_tokens(["掌握经验"])
    assert "mastery experience" in expanded
    assert "performance accomplishments" in expanded
