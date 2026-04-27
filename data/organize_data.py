from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source: full HOMUS dataset (absolute, relative to this script's location)
SOURCE_DIR = PROJECT_ROOT / "HOMUS"

# Target: project data/raw subdirectories
TARGET_BASE = PROJECT_ROOT / "data" / "raw"

# Maps lowercased HOMUS class label (first line of each file) -> target subfolder.
# Substring matching is used, so keys must be specific enough to avoid false hits.
CLASS_MAP = {
    # Notes (original 3)
    "whole-note":   TARGET_BASE / "whole-note",
    "half-note":    TARGET_BASE / "half-note",
    "quarter-note": TARGET_BASE / "quarter-note",
    # Clefs (new)
    "g-clef":       TARGET_BASE / "g-clef",
    "f-clef":       TARGET_BASE / "f-clef",
    # Rests (new)
    "quarter-rest": TARGET_BASE / "quarter-rest",
    "eighth-rest":  TARGET_BASE / "eighth-rest",
    # Time signatures (new)
    "common-time":  TARGET_BASE / "common-time",
}


def verify_source(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(
            f"Source directory not found: {source}\n"
            "Check that the HOMUS dataset is unzipped at that path."
        )
    all_files = list(source.rglob("*.txt"))
    if not all_files:
        raise FileNotFoundError(f"No .txt files found under {source}")
    print(f"Source confirmed: {source}")
    print(f"First 5 files found:")
    for f in all_files[:5]:
        print(f"  {f.relative_to(source)}")
    print(f"Total .txt files: {len(all_files)}\n")
    return all_files


def organize_homus():
    all_files = verify_source(SOURCE_DIR)

    # Create target directories
    for folder in CLASS_MAP.values():
        folder.mkdir(parents=True, exist_ok=True)

    counts = {key: 0 for key in CLASS_MAP}
    skipped = 0

    for i, src_file in enumerate(all_files, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(all_files)} files...")

        try:
            # Class label is the first line of each HOMUS file
            first_line = src_file.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip().lower()
        except (OSError, IndexError):
            skipped += 1
            continue

        matched = False
        for key, dest_dir in CLASS_MAP.items():
            if key in first_line:
                shutil.copy2(src_file, dest_dir / src_file.name)
                counts[key] += 1
                matched = True
                break

        if not matched:
            skipped += 1

    print("\n--- Results ---")
    for key, n in counts.items():
        print(f"  {key:10s} -> data/raw/{key}/  ({n} files)")
    print(f"  {'skipped':10s}              ({skipped} files, other classes)")
    print(f"  Total processed: {sum(counts.values()) + skipped}/{len(all_files)}")


if __name__ == "__main__":
    organize_homus()
