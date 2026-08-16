"""Filename-level metadata for known psychoanalysis sources in data/raw."""

from __future__ import annotations

FILE_CATALOG: dict[str, dict] = {
    "18.01.059.20241201.pdf": {
        "author": "Samindar J. Vibhute; B. Suresh Kumar",
        "school": "分析心理学",
        "core_concepts": ["analytical psychology", "collective unconscious", "archetype", "individuation"],
    },
    "19-7-wujie.pdf": {
        "author": "吴杰; 郭本禹",
        "school": "个体心理学",
        "core_concepts": ["阿德勒", "个体心理学", "自卑", "社会兴趣"],
    },
    "2007-6A-18.pdf": {
        "author": "丁建略; 田浩",
        "school": "精神分析",
        "core_concepts": ["霍妮", "神经症", "基本焦虑", "人际策略"],
    },
    "2021-tfp-extended-development-and-recent-advances.pdf": {
        "author": "John F. Clarkin; Eve Caligor; Julia Sowislo",
        "school": "客体关系",
        "core_concepts": ["TFP", "移情焦点治疗", "人格组织", "边缘型人格"],
    },
    "Clin Psychology and Psychoth - 2025 - Jørgensen - Mentalization‐Based Therapy for Borderline Personality Disorder .pdf": {
        "author": "Mie Sedoc Jørgensen; Stine Steen Høgenhaug; Carla Sharp; Sune Bo",
        "school": "心智化",
        "core_concepts": ["mentalization", "MBT", "borderline personality disorder", "心智化"],
    },
    "Freud and Epicurean Philosophy  Revisiting Drive Theory.pdf": {
        "author": "Jonathan Yahalom",
        "school": "精神分析",
        "core_concepts": ["drive theory", "死本能", "Freud", "Epicurean"],
    },
    "Int J of Psychoanalysis - 2010 - Schneider - From Freud s dream‐work to Bion s work of dreaming  The changing conception of.pdf": {
        "author": "John A. Schneider",
        "school": "精神分析",
        "core_concepts": ["dream-work", "Bion", "work of dreaming", "containment"],
    },
    "Object_Relations_Theory_A_Primer_for_Rehabilitatio.pdf": {
        "author": "Kenneth R. Thomas; Kaiqi Zhou; David A. Rosenthal",
        "school": "客体关系",
        "core_concepts": ["object relations", "Melanie Klein", "Fairbairn", "Winnicott"],
    },
    "Psychoanalytic ego psychology  A European perspective.pdf": {
        "author": "Marco Conci",
        "school": "精神分析",
        "core_concepts": ["ego psychology", "Anna Freud", "Hartmann", "自我心理学"],
    },
    "Relational Psychoanalysis  A Review.pdf": {
        "author": "Frederic Perlman; Jay Frankel",
        "school": "关系精神分析",
        "core_concepts": ["relational psychoanalysis", "intersubjectivity", "two-person psychology"],
    },
    "Sullivan and the intersubjective perspective.pdf": {
        "author": "Marco Conci",
        "school": "关系精神分析",
        "core_concepts": ["Sullivan", "interpersonal", "intersubjective"],
    },
    "The Separation-Individuation Inventory_Association with Borderline Phenomena.pdf": {
        "author": "unknown",
        "school": "客体关系",
        "core_concepts": ["separation-individuation", "Mahler", "borderline"],
    },
    "World Psychiatry - 2023 - Leichsenring - The status of psychodynamic psychotherapy as an empirically supported treatment.pdf": {
        "author": "Falk Leichsenring; Allan Abbass; Nikolas Heim; John R. Keefe; Steve Kisely; Patrick Luyten; Sven Rabung; Christiane Steinert",
        "school": "精神分析",
        "core_concepts": ["psychodynamic psychotherapy", "empirically supported treatment", "umbrella review"],
    },
    "behavsci-03-00562.pdf": {
        "author": "Christian Roesler",
        "school": "分析心理学",
        "core_concepts": ["Jungian psychotherapy", "outcome research", "archetype"],
    },
    "fpsyt-14-1237005.pdf": {
        "author": "Zhengyan Xie; Yuting Yan; Kejuan Peng",
        "school": "客体关系",
        "core_concepts": ["Winnicott", "true self", "false self", "transitional object"],
    },
    "paper14ConficlIntroBrennerbook.pdf": {
        "author": "Charles Brenner",
        "school": "精神分析",
        "core_concepts": ["conflict theory", "compromise formation", "ego"],
    },
    "自体心理学_理解和防治抑郁症的新视角.pdf": {
        "author": "张璇; 王申连",
        "school": "自体心理学",
        "core_concepts": ["自体心理学", "自体客体", "自恋", "抑郁症"],
    },
    "freud_unconscious.md": {
        "author": "Sigmund Freud",
        "school": "精神分析",
        "core_concepts": ["unconscious", "repression", "dream-work"],
    },
    "lacan_mirror.md": {
        "author": "Jacques Lacan",
        "school": "拉康派",
        "core_concepts": ["mirror stage", "imaginary", "subject"],
    },
    "sample_repression.txt": {
        "author": "Sigmund Freud",
        "school": "精神分析",
        "core_concepts": ["压抑", "无意识", "移情", "梦的工作"],
    },
}


def lookup_catalog(file_name: str) -> dict:
    return dict(FILE_CATALOG.get(file_name) or {})
