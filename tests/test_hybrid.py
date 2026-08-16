from src.retrieval.hybrid import RetrievalHit, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_prefers_overlap() -> None:
    dense = [
        RetrievalHit(text="镜像阶段是一种误认", score=0.8, source="dense"),
        RetrievalHit(text="压抑使欲望改道", score=0.7, source="dense"),
    ]
    bm25 = [
        RetrievalHit(text="镜像阶段是一种误认", score=4.2, source="bm25"),
        RetrievalHit(text="移情重复童年场景", score=1.1, source="bm25"),
    ]

    fused = reciprocal_rank_fusion([dense, bm25], weight=0.6)

    assert fused[0].text == "镜像阶段是一种误认"
    assert fused[0].source == "hybrid"
    assert {hit.text for hit in fused} == {
        "镜像阶段是一种误认",
        "压抑使欲望改道",
        "移情重复童年场景",
    }
