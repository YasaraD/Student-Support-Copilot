"""Tests for manifests and content-based change detection."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.config import INDEX_CONFIG
from src.index_manifest import (
    DocumentFingerprint,
    IndexManifest,
    IndexManifestError,
    compare_index_state,
    configuration_sha256,
    read_index_manifest,
    write_index_manifest,
)


def make_fingerprint(
    path: str, sha256: str = "a" * 64, size: int = 10
) -> DocumentFingerprint:
    """Create one valid test fingerprint."""
    return DocumentFingerprint(
        relative_path=path,
        category=path.split("/", 1)[0],
        sha256=sha256,
        size_bytes=size,
    )


def make_manifest(
    documents: tuple[DocumentFingerprint, ...],
    *,
    config=INDEX_CONFIG,
) -> IndexManifest:
    """Create a valid test manifest."""
    configuration = config.manifest_configuration()
    return IndexManifest(
        build_id="test-build",
        built_at_utc="2026-08-15T00:00:00+00:00",
        configuration=configuration,
        configuration_sha256=configuration_sha256(configuration),
        documents=documents,
        page_count=4,
        chunk_count=4,
        category_chunk_counts={category: 1 for category in config.supported_categories},
    )


class IndexManifestTests(unittest.TestCase):
    """Verify manifest persistence and comparisons."""

    def test_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            expected = make_manifest(
                (make_fingerprint("examinations/policy.pdf"),)
            )
            write_index_manifest(expected, path)
            self.assertEqual(read_index_manifest(path), expected)

    def test_added_modified_and_removed_documents_are_detected(self) -> None:
        previous = (
            make_fingerprint("examinations/old.pdf"),
            make_fingerprint("modules/changed.pdf"),
        )
        current = (
            make_fingerprint("modules/changed.pdf", "b" * 64),
            make_fingerprint("student_services/new.pdf"),
        )
        report = compare_index_state(current, make_manifest(previous))
        self.assertEqual(report.added, ("student_services/new.pdf",))
        self.assertEqual(report.modified, ("modules/changed.pdf",))
        self.assertEqual(report.removed, ("examinations/old.pdf",))
        self.assertTrue(report.rebuild_required)

    def test_index_configuration_change_is_detected(self) -> None:
        documents = (make_fingerprint("examinations/policy.pdf"),)
        manifest = make_manifest(documents)
        changed_config = replace(INDEX_CONFIG, chunk_size=900)
        report = compare_index_state(documents, manifest, changed_config)
        self.assertTrue(report.configuration_changed)
        self.assertTrue(report.rebuild_required)

    def test_retrieval_only_change_does_not_require_rebuild(self) -> None:
        documents = (make_fingerprint("examinations/policy.pdf"),)
        manifest = make_manifest(documents)
        changed_config = replace(INDEX_CONFIG, top_k=2)
        report = compare_index_state(documents, manifest, changed_config)
        self.assertFalse(report.configuration_changed)
        self.assertFalse(report.rebuild_required)

    def test_missing_manifest_requires_rebuild(self) -> None:
        documents = (make_fingerprint("examinations/policy.pdf"),)
        report = compare_index_state(documents, None)
        self.assertTrue(report.manifest_missing)
        self.assertTrue(report.rebuild_required)

    def test_corrupt_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            with self.assertRaises(IndexManifestError):
                read_index_manifest(path)


if __name__ == "__main__":
    unittest.main()
