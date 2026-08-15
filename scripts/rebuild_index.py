"""Command-line interface for safe local knowledge-index management."""

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import INDEX_CONFIG  # noqa: E402
from src.index_manager import (  # noqa: E402
    IndexManagementError,
    IndexStatus,
    inspect_index_status,
    rollback_index,
    safe_rebuild_index,
)


def _print_changes(status: IndexStatus) -> None:
    """Print a readable source/configuration comparison."""
    changes = status.changes
    print(f"Source PDFs: {len(status.source_documents)}")
    print(
        "Active collection: "
        + (
            f"{status.active_chunk_count} chunks"
            if status.active_collection_exists
            else "missing"
        )
    )
    print(
        "Rollback collection: "
        + (
            f"{status.backup_chunk_count} chunks"
            if status.backup_collection_exists
            else "not available"
        )
    )
    print(
        "Manifest: "
        + (
            f"build {status.manifest.build_id}"
            if status.manifest is not None
            else "missing"
        )
    )
    print(f"Configuration changed: {'yes' if changes.configuration_changed else 'no'}")
    print(f"Added documents: {len(changes.added)}")
    for path in changes.added:
        print(f"  + {path}")
    print(f"Modified documents: {len(changes.modified)}")
    for path in changes.modified:
        print(f"  ~ {path}")
    print(f"Removed documents: {len(changes.removed)}")
    for path in changes.removed:
        print(f"  - {path}")
    print(f"Rebuild required: {'yes' if changes.rebuild_required else 'no'}")


def _confirm(prompt: str, assume_yes: bool) -> bool:
    """Require explicit confirmation unless --yes was supplied."""
    if assume_yes:
        return True
    response = input(f"{prompt} Type 'yes' to continue: ").strip().lower()
    return response == "yes"


def _create_parser() -> argparse.ArgumentParser:
    """Create the three-operation command parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect, rebuild, or roll back the local Student Support Copilot "
            "Chroma knowledge index. Stop Streamlit before rebuild or rollback."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Detect document/configuration changes.")

    rebuild_parser = subparsers.add_parser(
        "rebuild",
        help="Build staging, validate it, then promote it safely.",
    )
    rebuild_parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the manifest reports no changes.",
    )
    rebuild_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )

    rollback_parser = subparsers.add_parser(
        "rollback",
        help="Swap the active collection with the retained previous generation.",
    )
    rollback_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one explicit index-management operation."""
    arguments = _create_parser().parse_args(argv)
    try:
        if arguments.command == "status":
            _print_changes(inspect_index_status())
            return 0

        if arguments.command == "rebuild":
            status = inspect_index_status()
            _print_changes(status)
            if not status.changes.rebuild_required and not arguments.force:
                print("No rebuild was performed because no changes were detected.")
                return 0
            if not _confirm(
                "A validated staging collection will replace the active index.",
                arguments.yes,
            ):
                print("Rebuild cancelled; the active index was not changed.")
                return 0
            print("Building and validating the staging index...")
            result = safe_rebuild_index(force=arguments.force)
            if result.manifest is None:
                raise IndexManagementError("The rebuild produced no manifest.")
            print("Index rebuild completed successfully.")
            print(f"Build ID: {result.manifest.build_id}")
            print(f"Documents: {len(result.manifest.documents)}")
            print(f"Pages: {result.manifest.page_count}")
            print(f"Chunks: {result.manifest.chunk_count}")
            print(
                "Rollback collection: "
                + ("created" if result.backup_created else "not available")
            )
            return 0

        if not _confirm(
            "The active and previous Chroma generations will be swapped.",
            arguments.yes,
        ):
            print("Rollback cancelled; the active index was not changed.")
            return 0
        result = rollback_index()
        print("Index rollback completed successfully.")
        print(f"Active chunks: {result.active_chunk_count}")
        print(f"Rollback chunks: {result.backup_chunk_count}")
        print(
            "Active manifest: "
            + (
                result.active_manifest.build_id
                if result.active_manifest is not None
                else "legacy index without a manifest"
            )
        )
        return 0
    except IndexManagementError as error:
        print(f"Index operation failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nIndex operation cancelled; no further action was requested.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
