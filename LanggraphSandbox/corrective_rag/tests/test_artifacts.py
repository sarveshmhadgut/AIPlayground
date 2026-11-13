"""
test_artifacts.py: Unit tests for artifact persistence, query slugging, and retention pruning.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from langchain_core.documents import Document

ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.artifacts import (
    generate_artifacts_catalog,
    prune_old_artifacts,
    sanitize_query_slug,
    save_run_artifacts,
)


def test_sanitize_query_slug_max_length() -> None:
    """Test that query slug is sanitized and truncated strictly to max 50 characters."""
    long_query: str = (
        "How do the RAG evaluation patterns described in AI Engineering "
        "compare to current benchmarks like RAGAS and ARES?"
    )
    slug: str = sanitize_query_slug(long_query, max_length=50)

    assert len(slug) <= 50
    assert not slug.endswith("_")
    assert "rag_evaluation_patterns" in slug
    assert "?" not in slug


def test_sanitize_query_slug_special_characters_and_fallback() -> None:
    """Test sanitization handles special characters, multiple spaces, and empty queries."""
    special_query: str = "  What is Deep Learning?! @#$%^&*   "
    slug: str = sanitize_query_slug(special_query, max_length=50)
    assert slug == "what_is_deep_learning"

    empty_query: str = "   !!!???   "
    fallback_slug: str = sanitize_query_slug(empty_query, max_length=50)
    assert fallback_slug == "query"


def test_save_run_artifacts_directory_structure_and_files(tmp_path: Path) -> None:
    """Test save_run_artifacts creates expected directories, catalog, and formatted files."""
    query: str = "Explain transformer attention and Mamba state space models"

    doc_hit: Document = Document(
        page_content="Transformer attention mechanisms have quadratic complexity.",
        metadata={"source": "deep_learning.pdf", "page": 10, "score": 8.5},
    )
    doc_ambiguous: Document = Document(
        page_content="State space sequence modeling offers linear scaling.",
        metadata={"source": "ai_engineering.pdf", "page": 42, "score": 6.0},
    )
    doc_miss: Document = Document(
        page_content="Cooking recipes for dinner.",
        metadata={"source": "unrelated.pdf", "page": 1, "score": 1.0},
    )

    mock_state: dict[str, object] = {
        "rag_context": "Transformer attention mechanisms have quadratic complexity.",
        "web_search_context": "Mamba uses structured state spaces (S4) for linear inference.",
        "response": "Transformers and Mamba differ fundamentally in computational complexity.",
        "retrieved_documents": [doc_hit, doc_ambiguous, doc_miss],
        "hit_documents": [doc_hit],
        "ambiguous_documents": [doc_ambiguous],
        "miss_documents": [doc_miss],
    }

    run_dirpath: Path = save_run_artifacts(
        query=query,
        state=mock_state,
        artifacts_dirpath=tmp_path,
        max_runs=50,
    )

    assert run_dirpath.exists()
    assert run_dirpath.is_dir()
    # Directory name starts with timestamp_slug (YYYYMMDD_HHMMSS_query[50])
    assert re.match(r"^\d{8}_\d{6}_explain_transformer_attention", run_dirpath.name)

    # 1. contexts directory check
    contexts_file: Path = run_dirpath / "contexts" / "contexts.md"
    assert contexts_file.exists()
    contexts_content: str = contexts_file.read_text(encoding="utf-8")
    assert "Query:\nExplain transformer attention and Mamba state space models" in contexts_content
    assert "RAG Context:\nTransformer attention mechanisms have quadratic complexity." in contexts_content
    assert "Web Search Context:\nMamba uses structured state spaces (S4)" in contexts_content

    # 2. response directory check
    response_file: Path = run_dirpath / "response" / "response.md"
    assert response_file.exists()
    response_content: str = response_file.read_text(encoding="utf-8")
    assert "Query: Explain transformer attention and Mamba state space models" in response_content
    assert "Response:\nTransformers and Mamba differ fundamentally in computational complexity." in response_content

    # 3. metadata.json check
    metadata_file: Path = run_dirpath / "metadata.json"
    assert metadata_file.exists()
    metadata: dict[str, object] = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata.get("query") == query
    assert metadata.get("routing_decision") == "generation"
    counts: dict[str, int] = metadata.get("counts", {})  # type: ignore[assignment]
    assert counts.get("retrieved_documents") == 3
    assert counts.get("hit_documents") == 1
    assert counts.get("ambiguous_documents") == 1
    assert counts.get("miss_documents") == 1

    # 4. evaluations directory check
    evaluations_json: Path = run_dirpath / "evaluations" / "evaluations.json"
    evaluations_md: Path = run_dirpath / "evaluations" / "evaluations.md"
    assert evaluations_json.exists()
    assert evaluations_md.exists()

    evaluated_records: list[dict[str, object]] = json.loads(evaluations_json.read_text(encoding="utf-8"))
    assert len(evaluated_records) == 3
    assert evaluated_records[0].get("classification") == "hit"
    assert evaluated_records[1].get("classification") == "ambiguous"
    assert evaluated_records[2].get("classification") == "miss"

    # 5. catalog index check
    catalog_file: Path = tmp_path / "catalog.md"
    assert catalog_file.exists()
    catalog_content: str = catalog_file.read_text(encoding="utf-8")
    assert "# Corrective RAG Artifacts Catalog" in catalog_content
    assert run_dirpath.name in catalog_content


def test_generate_artifacts_catalog_orders_newest_first(tmp_path: Path) -> None:
    """Test that generate_artifacts_catalog orders runs strictly with the newest at the top."""
    # Create older run
    older_dir: Path = tmp_path / "20261110_120000_older_run"
    older_dir.mkdir(parents=True)
    with open(older_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"query": "Older Query", "timestamp": "2026-11-10T12:00:00"}, f)
    os.utime(older_dir, (1000, 1000))

    # Create newer run
    newer_dir: Path = tmp_path / "20261112_180000_newer_run"
    newer_dir.mkdir(parents=True)
    with open(newer_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"query": "Newer Query", "timestamp": "2026-11-12T18:00:00"}, f)
    os.utime(newer_dir, (2000, 2000))

    catalog_path: Path = generate_artifacts_catalog(artifacts_dirpath=tmp_path)
    assert catalog_path.exists()
    content: str = catalog_path.read_text(encoding="utf-8")

    # Newer run must appear before older run in the catalog table
    newer_position: int = content.find("20261112_180000_newer_run")
    older_position: int = content.find("20261110_120000_older_run")

    assert newer_position != -1
    assert older_position != -1
    assert newer_position < older_position


def test_prune_old_artifacts_retention_limit(tmp_path: Path) -> None:
    """Test that prune_old_artifacts retains only the latest max_runs directories."""
    # Create 55 dummy run directories with incrementing timestamps
    created_directories: list[Path] = []
    base_time: float = time.time() - 1000

    for i in range(55):
        dir_name: str = f"20261112_{i:06d}_run_{i:02d}"
        dirpath: Path = tmp_path / dir_name
        dirpath.mkdir(parents=True)
        # Stagger mtimes
        mtime: float = base_time + (i * 10)
        os.utime(dirpath, (mtime, mtime))
        created_directories.append(dirpath)

    assert len([d for d in tmp_path.iterdir() if d.is_dir()]) == 55

    deleted_names: list[str] = prune_old_artifacts(artifacts_dirpath=tmp_path, max_runs=50)

    assert len(deleted_names) == 5
    remaining_directories: list[Path] = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(remaining_directories) == 50

    # Oldest 5 (run_00 through run_04) should have been pruned
    for i in range(5):
        assert f"run_{i:02d}" in deleted_names[i]
        assert not (tmp_path / created_directories[i].name).exists()

    # Remaining 50 (run_05 through run_54) must exist
    for i in range(5, 55):
        assert (tmp_path / created_directories[i].name).exists()
