"""
Factory helpers for the simplified pre-retrieval pipeline.
"""
from __future__ import annotations

from typing import Any

from pre_retrieval.core.config import PreRetrievalConfig
from pre_retrieval.core.pipeline import PreRetrievalPipeline


def build_pipeline(transformations: list[str] | None = None, **kwargs: Any) -> PreRetrievalPipeline:
    cfg_dict = dict(kwargs)
    if transformations is not None:
        cfg_dict["transformations"] = transformations
    config = PreRetrievalConfig.from_dict(cfg_dict, global_cfg={"language": cfg_dict.get("language", "both")})
    return PreRetrievalPipeline(config)


def build_pipeline_from_config(cfg: dict) -> PreRetrievalPipeline:
    query_cfg = cfg.get("query_pipeline", {})
    pre_cfg = query_cfg.get("pre_retrieval", {})
    config = PreRetrievalConfig.from_dict(pre_cfg, global_cfg=cfg.get("data", {}))
    return PreRetrievalPipeline(config)
