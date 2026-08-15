"""Tests for centralized index configuration."""

import unittest
from dataclasses import replace

from src.config import INDEX_CONFIG, IndexConfiguration
from src.embeddings import EMBEDDING_MODEL_NAME
from src.retriever import MAX_TOP_K, TOP_K
from src.text_splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.vector_store import COLLECTION_NAME, DISTANCE_METRIC


class IndexConfigurationTests(unittest.TestCase):
    """Verify shared settings and validation."""

    def test_existing_modules_reuse_central_configuration(self) -> None:
        self.assertEqual(EMBEDDING_MODEL_NAME, INDEX_CONFIG.embedding_model_name)
        self.assertEqual(DEFAULT_CHUNK_SIZE, INDEX_CONFIG.chunk_size)
        self.assertEqual(DEFAULT_CHUNK_OVERLAP, INDEX_CONFIG.chunk_overlap)
        self.assertEqual(COLLECTION_NAME, INDEX_CONFIG.collection_name)
        self.assertEqual(DISTANCE_METRIC, INDEX_CONFIG.distance_metric)
        self.assertEqual(TOP_K, INDEX_CONFIG.top_k)
        self.assertEqual(MAX_TOP_K, INDEX_CONFIG.max_top_k)

    def test_retrieval_only_top_k_is_not_part_of_index_manifest(self) -> None:
        changed = replace(INDEX_CONFIG, top_k=2)
        self.assertEqual(
            changed.manifest_configuration(),
            INDEX_CONFIG.manifest_configuration(),
        )

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            IndexConfiguration(chunk_size=100, chunk_overlap=100)

    def test_category_directory_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported category"):
            INDEX_CONFIG.category_directory("unknown")


if __name__ == "__main__":
    unittest.main()
