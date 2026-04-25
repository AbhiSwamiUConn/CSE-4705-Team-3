from pathlib import Path
import shutil

# Source: full 15,200-file HOMUS dataset (Windows Downloads)
SOURCE_DIR = Path("/mnt/c/Users/Krish/Downloads/HOMUS/HOMUS")

# Target: project data/raw subdirectories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_BASE = PROJECT_ROOT / "data" / "raw"

# Class label (first line of file) -> target subfolder
# The HOMUS files use "Whole-Note", "Half-Note", "Quarter-Note" (capital N)
CLASS_MAP = {
    "whole": TARGET_BASE / "whole",
    "half":  TARGET_BASE / "half",
    "quarter": TARGET_BASE / "quarter",
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
