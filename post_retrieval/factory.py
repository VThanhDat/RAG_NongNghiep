"""
Factory helpers for the simplified post-retrieval stage.
"""
from __future__ import annotations

from post_retrieval.pipeline import PostRetrievalPipeline


def build_pipeline(**kwargs) -> PostRetrievalPipeline:
    return PostRetrievalPipeline(**kwargs)


def build_pipeline_from_config(cfg: dict) -> PostRetrievalPipeline:
    post_cfg = cfg.get("query_pipeline", {}).get("post_retrieval", {})
    return PostRetrievalPipeline(
        top_n=post_cfg.get("top_n", 5),
        score_threshold=post_cfg.get("score_threshold", 0.0),
        apply_redundancy=post_cfg.get("apply_redundancy", True),
        redundancy_threshold=post_cfg.get("redundancy_threshold", 0.9),
        context_ordering=post_cfg.get("context_ordering", "relevance"),
        apply_toc_filter=post_cfg.get("apply_toc_filter", True),
        toc_dot_ratio=post_cfg.get("toc_dot_ratio", 0.15),
        toc_short_line_ratio=post_cfg.get("toc_short_line_ratio", 0.70),
        toc_min_chars=post_cfg.get("toc_min_chars", 200),
        toc_early_page_limit=post_cfg.get("toc_early_page_limit", 3),
    )

