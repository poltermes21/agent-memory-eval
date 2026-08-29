"""Download data/raw/locomo10.json from the upstream repo.

Not vendored in git: the dataset is CC BY-NC (see README "Dataset").
Idempotent — skips the download if the file already exists, unless --force.
"""
import argparse
import urllib.request

from src.config import DATA_RAW_DIR, LOCOMO_PATH, LOCOMO_URL


def fetch(force: bool = False) -> None:
    if LOCOMO_PATH.exists() and not force:
        print(f"already present: {LOCOMO_PATH} (use --force to re-download)")
        return
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {LOCOMO_URL}")
    urllib.request.urlretrieve(LOCOMO_URL, LOCOMO_PATH)
    size_mb = LOCOMO_PATH.stat().st_size / 1_000_000
    print(f"saved {LOCOMO_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    fetch(force=args.force)
