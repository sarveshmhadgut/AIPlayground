"""
state.py: Data models, schemas, and LangGraph workflow State definition.
"""

import sys
from pathlib import Path
from typing import TypedDict

from langchain_core.documents.base import Document
from pydantic import BaseModel, Field

ROOT: Path = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DocumentGrade(BaseModel):
    """Structured evaluation output for an individual retrieved document chunk."""

    document_index: int = Field(
        ge=1,
        description=(
            "The 1-based index corresponding to the numbered retrieved document passage (e.g. 1 for [Document 1])."
        ),
    )
    score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Calibrated evaluation score from 0.0 to 10.0 strictly assigned according to the prompt's scoring rubric: "
            "Hit / High Relevance for documents that directly address the query or provide key factual context; "
            "Ambiguous / Partial Relevance for tangentially related or incomplete context insufficient on its own; "
            "Miss / Irrelevant for completely off-topic, unrelated, or useless content."
        ),
    )


class EvaluationSchema(BaseModel):
    """Structured batch evaluation response containing scores for all retrieved documents."""

    evaluations: list[DocumentGrade] = Field(
        default_factory=list,
        description="List of evaluation grades, one for each retrieved document chunk.",
    )


class RAGQueryRewritingSchema(BaseModel):
    """Structured output schema for vector store semantic retrieval query reformulation."""

    rewritten_query: str = Field(
        description=(
            "Semantic retrieval query optimized for dense vector similarity search. "
            "Formulated by expanding core technical concepts, domain entities, and key terms while "
            "stripping conversational filler to maximize embedding cosine similarity against document chunks."
        )
    )


class WebQueryRewritingSchema(BaseModel):
    """Structured output schema for search-engine-optimized (SEO) query reformulation."""

    rewritten_query: str = Field(
        description=(
            "Search-engine-optimized (SEO) keyword query. Formatted with 3 to 6 high-intent, "
            "disambiguated search terms, stripped of conversational filler, stop words, and punctuation, "
            "anchored with domain identifiers to maximize search engine retrieval precision."
        )
    )


class RefinementSchema(BaseModel):
    """Structured output schema for query-focused context refinement."""

    refined_context: str = Field(
        description=(
            "Distilled, de-duplicated, and factually grounded context synthesized from "
            "the provided internal RAG and external web search sources. Contains only key facts, "
            "definitions, numbers, and statements directly relevant to the user query, with all "
            "redundancy, promotional noise, and irrelevant passages completely removed."
        )
    )


class State(TypedDict):
    """Schema representing the LangGraph workflow state.

    Attributes:
        query (str): The user's input query.
        rewritten_rag_query (str): Optimized semantic query formulated for vector store document retrieval.
        rewritten_query (str): Optimized search keywords formulated for web search.
        retrieved_documents (list[Document]): Ingested documents retrieved from vector store.
        hit_documents (list[Document]): Documents evaluated as highly relevant with score in metadata.
        miss_documents (list[Document]): Documents evaluated as irrelevant with score in metadata.
        ambiguous_documents (list[Document]): Documents evaluated as partially relevant with score in metadata.
        web_search_context (str): Supplemental context fetched from web search.
        rag_context (str): Retrieved context documents formatted as a string.
        refined_context (str): Distilled context synthesized from retrieved RAG and web search contexts.
        response (str): Final generated response from the LLM.
    """

    query: str
    rewritten_rag_query: str
    rewritten_query: str
    retrieved_documents: list[Document]
    hit_documents: list[Document]
    miss_documents: list[Document]
    ambiguous_documents: list[Document]
    web_search_context: str
    rag_context: str
    refined_context: str
    response: str


__all__: list[str] = [
    "DocumentGrade",
    "EvaluationSchema",
    "RAGQueryRewritingSchema",
    "RefinementSchema",
    "WebQueryRewritingSchema",
    "State",
]
