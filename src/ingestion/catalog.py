"""Filename-level metadata for known Bandura sources in data/raw."""

from __future__ import annotations

FILE_CATALOG: dict[str, dict] = {
    "sample_self_efficacy.txt": {
        "author": "Albert Bandura",
        "school": "自我效能",
        "core_concepts": ["自我效能", "效能预期", "结果预期", "掌握经验"],
    },
    "bandura_self_efficacy_unifying.md": {
        "author": "Albert Bandura",
        "school": "自我效能",
        "core_concepts": ["自我效能", "效能预期", "结果预期", "行为改变"],
    },
    "bandura_four_sources.md": {
        "author": "Albert Bandura",
        "school": "自我效能",
        "core_concepts": ["掌握经验", "替代经验", "社会劝说", "生理状态"],
    },
    "bandura_mastery_proximal.md": {
        "author": "Albert Bandura",
        "school": "目标与自我调节",
        "core_concepts": ["掌握经验", "引导掌握", "参与示范", "近端目标"],
    },
    "bandura_vicarious_peer_models.md": {
        "author": "Albert Bandura",
        "school": "观察学习",
        "core_concepts": ["替代经验", "观察学习", "应对榜样", "同伴榜样"],
    },
    "bandura_persuasion_attribution.md": {
        "author": "Albert Bandura",
        "school": "自我效能",
        "core_concepts": ["社会劝说", "言语劝说", "归因反馈", "认知加工"],
    },
    "bandura_physiological_affect.md": {
        "author": "Albert Bandura",
        "school": "自我效能",
        "core_concepts": ["生理状态", "情绪状态", "恐惧唤醒", "应对效能"],
    },
    "bandura_human_agency.md": {
        "author": "Albert Bandura",
        "school": "社会认知理论",
        "core_concepts": ["人类能动性", "自我效能", "目标", "三元交互"],
    },
    "bandura_observational_learning.md": {
        "author": "Albert Bandura",
        "school": "观察学习",
        "core_concepts": ["观察学习", "示范", "注意", "保持", "动作复现", "动机"],
    },
    "bandura_reciprocal_determinism.md": {
        "author": "Albert Bandura",
        "school": "社会认知理论",
        "core_concepts": ["三元交互", "相互决定论", "环境", "行为", "个人因素"],
    },
    "bandura_moral_disengagement.md": {
        "author": "Albert Bandura",
        "school": "道德疏离",
        "core_concepts": ["道德疏离", "道德自我调节", "责任分散"],
    },
    "schunk_attribution_feedback.md": {
        "author": "Dale H. Schunk",
        "school": "干预与测量",
        "core_concepts": ["归因反馈", "努力归因", "自我效能", "近端目标"],
    },
    "usher_pajares_sources_review.md": {
        "author": "Ellen L. Usher",
        "school": "干预与测量",
        "core_concepts": ["掌握经验", "替代经验", "社会劝说", "生理状态"],
    },
    "artino_2012_academic_self_efficacy.pdf": {
        "author": "Anthony R. Artino Jr.",
        "school": "自我效能",
        "core_concepts": ["自我效能", "掌握经验", "替代经验", "社会劝说", "生理状态"],
    },
    "ashford_2010_change_self_efficacy.pdf": {
        "author": "Stefanie Ashford",
        "school": "干预与测量",
        "core_concepts": ["自我效能", "掌握经验", "替代经验", "社会劝说", "分级掌握"],
    },
    "egele_2025_sources_ranking.pdf": {
        "author": "Viktoria S. Egele",
        "school": "干预与测量",
        "core_concepts": ["掌握经验", "替代经验", "社会劝说", "生理状态", "自我效能"],
    },
    "panadero_2017_srl_review.pdf": {
        "author": "Ernesto Panadero",
        "school": "目标与自我调节",
        "core_concepts": ["自我调节", "自我效能", "社会认知理论", "目标"],
    },
    "zakariya_2022_math_self_efficacy_review.pdf": {
        "author": "Yusuf F. Zakariya",
        "school": "干预与测量",
        "core_concepts": ["自我效能", "掌握经验", "替代经验", "近端目标"],
    },
    "schunk_1982_goal_proximity.pdf": {
        "author": "Dale H. Schunk",
        "school": "自我效能",
        "core_concepts": ["自我效能", "近端目标", "归因反馈", "掌握经验"],
    },
    "schunk_1982_social_comparison_goals.pdf": {
        "author": "Dale H. Schunk",
        "school": "目标与自我调节",
        "core_concepts": ["近端目标", "社会比较", "自我效能", "同伴榜样"],
    },
    "schunk_1983_rewards_goals.pdf": {
        "author": "Dale H. Schunk",
        "school": "目标与自我调节",
        "core_concepts": ["近端目标", "自我效能", "掌握经验"],
    },
    "schunk_1987_self_efficacy_review.pdf": {
        "author": "Dale H. Schunk",
        "school": "自我效能",
        "core_concepts": ["自我效能", "掌握经验", "替代经验", "归因反馈"],
    },
    "schunk_1988_self_modeling.pdf": {
        "author": "Dale H. Schunk",
        "school": "观察学习",
        "core_concepts": ["观察学习", "自我示范", "掌握经验", "自我效能"],
    },
    "schunk_1989_peer_models.pdf": {
        "author": "Dale H. Schunk",
        "school": "观察学习",
        "core_concepts": ["应对榜样", "掌握榜样", "同伴榜样", "自我效能"],
    },
    "schunk_1995_social_origins_srl.pdf": {
        "author": "Dale H. Schunk",
        "school": "目标与自我调节",
        "core_concepts": ["自我调节", "观察学习", "应对榜样", "同伴榜样", "自我效能"],
    },
    "bandura_1978_reciprocal_determinism.pdf": {
        "author": "Albert Bandura",
        "school": "社会认知理论",
        "core_concepts": ["三元交互", "相互决定论", "自我系统", "自我调节"],
    },
    "bandura_1982_mechanism_agency.pdf": {
        "author": "Albert Bandura",
        "school": "社会认知理论",
        "core_concepts": ["人类能动性", "自我效能", "集体效能", "代理控制"],
    },
    "bandura_2001_agentic_perspective.pdf": {
        "author": "Albert Bandura",
        "school": "社会认知理论",
        "core_concepts": ["人类能动性", "自我效能", "集体效能", "意向性", "预见"],
    },
}


def lookup_catalog(file_name: str) -> dict:
    return dict(FILE_CATALOG.get(file_name) or {})
