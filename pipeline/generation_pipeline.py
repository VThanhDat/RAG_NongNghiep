"""
Stage 2 generation pipeline for RAG Nong Nghiep.

Flow:
    query -> pre_retrieval -> hybrid retrieval -> post_retrieval
    -> citation prompt -> Gemini Flash generation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import yaml
from langchain_core.documents import Document

from generation.core.base import GenerationResult
from post_retrieval.core.types import PostRetrievalResult
from prompt.core.base import PromptResult

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except Exception:
    pass


@dataclass
class GenerationStepResult:
    step: str
    success: bool
    meta: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class GenerationPipelineResult:
    success: bool
    answer: str = ""
    query_used: str = ""
    retrieved: list[Document] = field(default_factory=list)
    final_docs: list[Document] = field(default_factory=list)
    prompt_result: PromptResult | None = None
    post_result: PostRetrievalResult | None = None
    gen_result: GenerationResult | None = None
    steps: list[GenerationStepResult] = field(default_factory=list)
    error: str | None = None

    @property
    def cited_sources(self) -> list[int]:
        return self.gen_result.cited_sources if self.gen_result else []


class GenerationPipeline:
    @classmethod
    def from_config_file(cls, config_path: str = "config.yaml") -> tuple["GenerationPipeline", dict]:
        with open(config_path, encoding="utf-8") as file:
            return cls(), yaml.safe_load(file)

    def run(
        self,
        query: str,
        vdb_result: dict,
        cfg: dict,
        history: list[dict] | None = None,
        skip_generation: bool = False,
        metadata_filter: dict | None = None,
    ) -> GenerationPipelineResult:
        result = GenerationPipelineResult(success=False)

        try:
            pre_result = self._run_pre_retrieval(query, cfg, history)
            if metadata_filter:
                current_filter = pre_result.metadata_filter or {}
                current_filter.update(metadata_filter)
                pre_result.metadata_filter = current_filter
            result.steps.append(GenerationStepResult("pre_retrieval", True, {"intent": pre_result.intent}))
            if not pre_result.is_valid:
                result.error = pre_result.rejection_reason or "Invalid query."
                result.steps[-1].success = False
                result.steps[-1].error = result.error
                return result

            query_used = pre_result.all_queries()[0] if pre_result.all_queries() else query
            result.query_used = query_used

            retrieved = self._run_retrieval(pre_result, vdb_result, cfg)
            result.retrieved = retrieved
            result.steps.append(GenerationStepResult("retrieval", True, {"n_retrieved": len(retrieved)}))

            post_result = self._run_post_retrieval(query_used, retrieved, cfg)
            result.post_result = post_result
            result.final_docs = post_result.documents
            result.steps.append(
                GenerationStepResult(
                    "post_retrieval",
                    True,
                    {"n_input": post_result.input_count, "n_output": post_result.output_count},
                )
            )

            prompt_result = self._run_prompt(query_used, post_result.documents, cfg, history)
            result.prompt_result = prompt_result
            result.steps.append(
                GenerationStepResult("prompt", True, {"template": prompt_result.template_name})
            )

            if skip_generation:
                result.success = True
                return result

            gen_result = self._run_generation(prompt_result, cfg)
            result.gen_result = gen_result
            result.answer = gen_result.answer
            result.steps.append(
                GenerationStepResult(
                    "generation",
                    True,
                    {"provider": gen_result.provider, "model": gen_result.model_name},
                )
            )
        except Exception as exc:
            result.error = str(exc)
            result.steps.append(GenerationStepResult("failed", False, error=str(exc)))
            return result

        result.success = True
        return result

    def stream(
        self,
        query: str,
        vdb_result: dict,
        cfg: dict,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        dry_run = self.run(query, vdb_result, cfg, history=history, skip_generation=True)
        if not dry_run.success or dry_run.prompt_result is None:
            raise RuntimeError(dry_run.error or "Generation pipeline failed before streaming.")

        gen_cfg = cfg.get("query_pipeline", {}).get("generation", {})
        from generation import get_generator

        generator = get_generator(
            provider=gen_cfg.get("provider", "gemini"),
            model_name=gen_cfg.get("model_name", "gemini-3.6-flash"),
            api_key=gen_cfg.get("api_key"),
            temperature=gen_cfg.get("temperature", 0.0),
            max_tokens=gen_cfg.get("max_tokens", 1024),
            streaming=True,
        )
        yield from generator.stream(dry_run.prompt_result)

    def _run_pre_retrieval(self, query: str, cfg: dict, history: list[dict] | None):
        from pre_retrieval import build_pipeline_from_config

        pipeline = build_pipeline_from_config(cfg)
        return pipeline.transform(query, chat_history=history or [])

    def _run_retrieval(self, pre_result, vdb_result: dict, cfg: dict) -> list[Document]:
        from retrieval import build_retriever_from_config

        vector_store = vdb_result.get("vector_store")
        documents = vdb_result.get("documents") or vdb_result.get("chunks")
        if vector_store is None:
            raise ValueError("Missing vector_store. Run indexing first.")
        if not documents:
            raise ValueError("Missing indexed documents for hybrid BM25 retrieval.")

        retriever = build_retriever_from_config(cfg, vector_store=vector_store, documents=documents)
        return retriever.retrieve(pre_result)

    def _run_post_retrieval(self, query: str, docs: list[Document], cfg: dict) -> PostRetrievalResult:
        from post_retrieval import build_pipeline_from_config

        pipeline = build_pipeline_from_config(cfg)
        return pipeline.process(query=query, docs=docs)

    def _run_prompt(
        self,
        query: str,
        docs: list[Document],
        cfg: dict,
        history: list[dict] | None,
    ) -> PromptResult:
        from prompt import build_prompt_builder_from_config

        builder = build_prompt_builder_from_config(cfg)
        return builder.build(query=query, docs=docs, history=history or [])

    def _run_generation(self, prompt_result: PromptResult, cfg: dict) -> GenerationResult:
        from generation import build_generator_from_config

        generator = build_generator_from_config(cfg)
        return generator.generate(prompt_result)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def _cli() -> None:
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Run the RAG Nong Nghiep generation pipeline.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--vdb-result", required=True, help="Pickle file containing vdb_result.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.vdb_result, "rb") as file:
        vdb_result = pickle.load(file)

    cfg = load_config(args.config)
    result = GenerationPipeline().run(args.query, vdb_result, cfg)
    if not result.success:
        raise SystemExit(f"Generation failed: {result.error}")
    print(result.answer)


if __name__ == "__main__":
    _cli()
