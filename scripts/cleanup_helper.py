"""
P3.2 - File cleanup script.

This script helps identify and clean up old files after refactoring.

Run with: python scripts/cleanup_helper.py --check
"""

import sys
from pathlib import Path

# Files to check/delete after refactoring
FILES_TO_DELETE = [
    # Old pipeline modules that have been moved/consolidated
    "pipeline/signals_multi.py",  # Logic moved to strategies/
    "pipeline/detection_multi.py",  # To be consolidated
    "pipeline/asset_tracker.py",  # To be moved to execution/

    # Old output files
    "outputs/signal_persistence_state.json",  # Replaced by data/cache/
]

# Files to update (change import paths)
FILES_TO_UPDATE = [
    "scheduler_service.py",  # Already updated ✓
    "ui/app.py",  # Needs to use new schema
    "dashboard/",  # Needs to use new schema
    "backend/",  # Needs to use new schema
]

# Directories created during refactoring
NEW_DIRECTORIES = [
    "strategies/",
    "features/",
    "data/",
    "execution/",
    "research/",
    "scoring/",
    "configs/strategies/",
    "configs/instruments/",
    "scripts/",
]


def check_file_status():
    """Check status of files that need cleanup."""
    print("=" * 60)
    print("P3.2 FILE CLEANUP CHECK")
    print("=" * 60)

    print("\n[1] Files to DELETE:")
    for f in FILES_TO_DELETE:
        path = Path(f)
        if path.exists():
            print(f"  ✓ {f} (exists, {path.stat().st_size} bytes)")
        else:
            print(f"  - {f} (not found)")

    print("\n[2] Files to UPDATE:")
    for f in FILES_TO_UPDATE:
        path = Path(f)
        if path.exists():
            print(f"  • {f}")
        elif path.is_dir():
            print(f"  • {f}/ (directory)")
        else:
            print(f"  - {f} (not found)")

    print("\n[3] NEW DIRECTORIES:")
    for d in NEW_DIRECTORIES:
        path = Path(d)
        if path.exists() and path.is_dir():
            print(f"  ✓ {d}/")
        else:
            print(f"  - {d}/ (not found)")

    print("\n" + "=" * 60)
    print("Run with --delete to actually delete files")
    print("=" * 60)


def delete_old_files():
    """Delete old files that are no longer needed."""
    print("=" * 60)
    print("DELETING OLD FILES")
    print("=" * 60)

    deleted = []
    skipped = []

    for f in FILES_TO_DELETE:
        path = Path(f)
        if path.exists():
            try:
                path.unlink()
                print(f"  Deleted: {f}")
                deleted.append(f)
            except Exception as e:
                print(f"  ERROR deleting {f}: {e}")
                skipped.append(f)
        else:
            print(f"  Skipped (not found): {f}")

    print(f"\nDeleted: {len(deleted)} files")
    print(f"Skipped: {len(skipped)} files")


def update_imports():
    """Show import paths that need updating."""
    print("=" * 60)
    print("IMPORT PATH UPDATES NEEDED")
    print("=" * 60)

    updates = {
        "ui/app.py": [
            "from pipeline.* → from strategies.*",
            "Update to use ResearchSignal/TradeCandidate schema",
        ],
        "dashboard/": [
            "Update to use new signal schema",
            "Use strategies/ for signal generation",
        ],
    }

    for file_path, notes in updates.items():
        print(f"\n{file_path}:")
        for note in notes:
            print(f"  • {note}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cleanup helper for P3.2")
    parser.add_argument("--check", action="store_true", help="Check file status")
    parser.add_argument("--delete", action="store_true", help="Delete old files")
    parser.add_argument("--imports", action="store_true", help="Show import updates needed")
    args = parser.parse_args()

    if args.check:
        check_file_status()
    elif args.delete:
        delete_old_files()
    elif args.imports:
        update_imports()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
