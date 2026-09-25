"""Cohere multilingual embeddings provider."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings

from embedding.core.base import BaseEmbedder


class _CohereLangChainEmbeddings(Embeddings):
    """LangChain adapter for Cohere retrieval document/query embeddings."""

    MAX_INPUTS_PER_REQUEST = 96

    def __init__(
        self,
        *,
        client,
        model_name: str,
        output_dimension: int | None,
        batch_size: int,
        document_input_type: str,
        query_input_type: str,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.output_dimension = output_dimension
        self.batch_size = min(max(1, int(batch_size)), self.MAX_INPUTS_PER_REQUEST)
        self.document_input_type = document_input_type
        self.query_input_type = query_input_type

    @staticmethod
    def _float_embeddings(response) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")

        values = getattr(embeddings, "float", None)
        if values is None and isinstance(embeddings, dict):
            values = embeddings.get("float")
        if not values:
            raise RuntimeError("Cohere embedding response did not contain float vectors.")
        return [[float(value) for value in vector] for vector in values]

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            kwargs = {
                "texts": batch,
                "model": self.model_name,
                "input_type": input_type,
                "embedding_types": ["float"],
            }
            if self.output_dimension is not None:
                kwargs["output_dimension"] = self.output_dimension
            response = self.client.embed(**kwargs)
            embeddings = self._float_embeddings(response)
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    "Cohere returned an unexpected number of embeddings: "
                    f"expected {len(batch)}, got {len(embeddings)}."
                )
            vectors.extend(embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self.document_input_type)

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query], self.query_input_type)[0]


class CohereEmbedder(BaseEmbedder):
    """Cohere Embed v4 wrapper for multilingual agricultural retrieval."""

    def __init__(
        self,
        model_name: str = "embed-v4.0",
        api_key: str | None = None,
        output_dimension: int | None = 1024,
        batch_size: int = 96,
        document_input_type: str = "search_document",
        query_input_type: str = "search_query",
        **kwargs,
    ) -> None:
        if model_name != "embed-v4.0":
            raise ValueError("Only Cohere embed-v4.0 is supported by this provider.")
        super().__init__(model_name, **kwargs)
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        self._output_dimension = output_dimension
        self._batch_size = batch_size
        self._document_input_type = document_input_type
        self._query_input_type = query_input_type

    def _build(self) -> Embeddings:
        if not self._api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not configured. Add it to .env or the deployment secret store."
            )
        import cohere

        return _CohereLangChainEmbeddings(
            client=cohere.ClientV2(api_key=self._api_key),
            model_name=self._model_name,
            output_dimension=self._output_dimension,
            batch_size=self._batch_size,
            document_input_type=self._document_input_type,
            query_input_type=self._query_input_type,
        )
