"""
Retrieval validation helpers for Vector DB search quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_ids: frozenset[str]
    filter: dict | None = None


def evaluate_retrieval(
    store,
    cases: Iterable[RetrievalCase | dict[str, Any]],
    *,
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, float | int]:
    """
    Evaluate retrieval with Recall@k, Precision@k, Hit Rate, and MRR.

    ``store`` is expected to expose the normalized ``search(query, k, filter)``
    helper attached by vector_db stores.
    """
    normalized_cases = [_normalize_case(item) for item in cases]
    if not normalized_cases:
        raise ValueError("Retrieval evaluation needs at least one query case.")

    max_k = max(k_values)
    totals: dict[str, float] = {f"recall@{k}": 0.0 for k in k_values}
    totals.update({f"precision@{k}": 0.0 for k in k_values})
    hit_count = 0
    reciprocal_rank_sum = 0.0

    for case in normalized_cases:
        results = store.search(case.query, k=max_k, filter=case.filter)
        retrieved_ids = [str(item.get("id")) for item in results if item.get("id")]
        relevant = set(case.relevant_ids)

        for k in k_values:
            top_ids = retrieved_ids[:k]
            hits = len(relevant.intersection(top_ids))
            totals[f"recall@{k}"] += hits / len(relevant) if relevant else 0.0
            totals[f"precision@{k}"] += hits / k if k else 0.0

        first_hit_rank = next(
            (
                index + 1
                for index, retrieved_id in enumerate(retrieved_ids)
                if retrieved_id in relevant
            ),
            None,
        )
        if first_hit_rank is not None:
            hit_count += 1
            reciprocal_rank_sum += 1.0 / first_hit_rank

    count = len(normalized_cases)
    metrics: dict[str, float | int] = {
        key: round(value / count, 6)
        for key, value in totals.items()
    }
    metrics["hit_rate"] = round(hit_count / count, 6)
    metrics["mrr"] = round(reciprocal_rank_sum / count, 6)
    metrics["query_count"] = count
    return metrics


def _normalize_case(item: RetrievalCase | dict[str, Any]) -> RetrievalCase:
    if isinstance(item, RetrievalCase):
        return item
    relevant = item.get("relevant_ids") or item.get("expected_ids") or []
    return RetrievalCase(
        query=str(item["query"]),
        relevant_ids=frozenset(str(value) for value in relevant),
        filter=item.get("filter"),
    )
