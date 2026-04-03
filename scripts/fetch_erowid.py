"""
fetch_erowid.py
===============
Download the three Erowid-LSA source files needed to run the scraper,
then invoke the scraper to populate substance experience directories.

Why not 'git clone'?
    The erowid-lsa repo contains experience/ subdirectory names with characters
    that are illegal on Windows filesystems ('?', ':', '&', '*').  Git's
    core.protectNTFS setting causes the checkout to fail on Windows even though
    the pack download succeeds.  Downloading only the Python and stopword files
    directly from GitHub's raw-content CDN avoids this entirely.

Source files downloaded:
    https://raw.githubusercontent.com/Monkeyanator/erowid-lsa/master/erowid-scrape.py
    https://raw.githubusercontent.com/Monkeyanator/erowid-lsa/master/analysis.py
    https://raw.githubusercontent.com/Monkeyanator/erowid-lsa/master/stopwords_en.txt

Usage:
    python scripts/fetch_erowid.py

Outputs:
    data/raw/erowid-lsa-repo/erowid-scrape.py   ← scraper (run by this script)
    data/raw/erowid-lsa-repo/analysis.py         ← LSA reference
    data/raw/erowid-lsa-repo/stopwords_en.txt    ← stopword list
    data/raw/erowid-lsa-repo/experiences/        ← populated by the scraper
        <substance>/
            1.txt, 2.txt, …

Notes:
    - The scrape is network-bound; expect 10–30 minutes for full coverage.
    - Partial runs are fine — process_erowid_lsa.py uses any substance
      with 20+ non-empty report files.
    - Re-running this script is idempotent: skips download if files already
      exist, skips scraping if experiences/ already has substance dirs.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT      = Path(__file__).parent.parent
CLONE_DIR = ROOT / "data" / "raw" / "erowid-lsa-repo"

# Only the three files we actually need from the repo
_RAW_BASE = "https://raw.githubusercontent.com/Monkeyanator/erowid-lsa/master"
_FILES_TO_DOWNLOAD = [
    "erowid-scrape.py",
    "analysis.py",
    "stopwords_en.txt",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_files(dest_dir: Path) -> None:
    """Download the three source files from GitHub raw content."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filename in _FILES_TO_DOWNLOAD:
        url      = f"{_RAW_BASE}/{filename}"
        out_path = dest_dir / filename
        print(f"  Downloading {filename} …")
        try:
            urllib.request.urlretrieve(url, str(out_path))
            print(f"    OK  {out_path.stat().st_size:,} bytes")
        except Exception as exc:
            print(f"    ERROR downloading {url}: {exc}")
            sys.exit(1)


def _source_files_present(dest_dir: Path) -> bool:
    """Return True if all three source files already exist."""
    return all((dest_dir / f).exists() for f in _FILES_TO_DOWNLOAD)


def _force_remove(dest_dir: Path) -> None:
    """
    Remove dest_dir, forcing read-only files writable first.
    Git pack files on Windows are read-only; shutil.rmtree alone fails on them.
    """
    import os
    import stat

    def _on_error(func, path, _exc):
        # Make the file writable, then retry
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(dest_dir, onerror=_on_error)


def _remove_broken_git_clone(dest_dir: Path) -> None:
    """
    Remove a broken git clone (has .git/ but no Python files).
    Called automatically when a previous git-clone attempt left a bad state.
    Git pack files are read-only on Windows, so plain rmtree would fail;
    _force_remove() clears the read-only bit before each delete.
    """
    git_dir = dest_dir / ".git"
    if dest_dir.exists() and git_dir.exists() and not _source_files_present(dest_dir):
        print(f"  Removing broken git clone at {dest_dir} …")
        _force_remove(dest_dir)
        print("  Removed.")


def _run_scraper(clone_dir: Path, experiences_dir: Path) -> None:
    """Run erowid-scrape.py inside clone_dir to populate experiences/."""
    scrape_script = clone_dir / "erowid-scrape.py"
    if not scrape_script.exists():
        print(f"WARNING — scraper not found at {scrape_script}; skipping.")
        return

    print("\nRunning Erowid scraper to collect experience reports …")
    print("(This may take 10–30 minutes depending on network speed.)")

    scrape = subprocess.run(
        [sys.executable, "erowid-scrape.py"],
        cwd=clone_dir,
        capture_output=True,
        text=True,
    )

    if scrape.returncode != 0:
        print("WARNING — scraper exited with errors (partial data may still be usable):")
        print(scrape.stderr[-2000:])

    n_dirs = sum(1 for _ in experiences_dir.iterdir()) if experiences_dir.exists() else 0
    print(f"Scrape complete. Substance dirs found: {n_dirs:,}")
    print(f"Location: {experiences_dir}")


# ── Main entry point ──────────────────────────────────────────────────────────

def fetch_erowid(clone_dir: Path = CLONE_DIR) -> None:
    experiences_dir = clone_dir / "experiences"

    # Clean up any broken git-clone remnant before proceeding
    _remove_broken_git_clone(clone_dir)

    # Download source files if not already present
    if _source_files_present(clone_dir):
        print(f"Source files already present at {clone_dir}")
    else:
        print(f"Downloading Erowid-LSA source files to {clone_dir} …")
        _download_files(clone_dir)
        print("Download complete.\n")

    # Run scraper if experiences/ is empty
    n_dirs = sum(1 for _ in experiences_dir.iterdir()) if experiences_dir.exists() else 0
    print(f"Experience dirs (scraped): {n_dirs:,}")
    if n_dirs == 0:
        print("No scraped data found yet. Running scraper …")
        _run_scraper(clone_dir, experiences_dir)
    else:
        print("Scrape already complete — skipping.")
        print(f"Location: {experiences_dir}")


if __name__ == "__main__":
    fetch_erowid()
