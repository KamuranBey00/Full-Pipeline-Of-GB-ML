"""
data.py
=======
Responsible for ONE thing: getting a clean, well-typed DataFrame into memory.

WHY isolate data loading in its own module?
- Every other part of the project (EDA, training, API) can call `load_data()`
  and trust it returns the same, validated dataset. Single source of truth.
- If the data source ever changes, we fix it here and nowhere else.
"""

import urllib.request
import pandas as pd

from config import RAW_CSV, DATA_DIR, TARGET, NUMERIC


# The canonical source is the UCI ML Repository. We also keep a GitHub mirror
# as a fallback because the UCI endpoint occasionally rejects automated
# downloads. Either way, the file is identical (10,000 rows, 14 columns).
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
MIRROR_URL = ("https://raw.githubusercontent.com/SamyamoyRakshit/"
              "AI4I-2020-Predictive-Maintenance-Dataset__Linear-Regression/"
              "main/ai4i2020.csv")


def download_data(force: bool = False) -> None:
    """Download the raw CSV into data/ if it is not already there.

    `force=True` re-downloads even if the file exists (useful to refresh data).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and not force:
        print(f"[data] Already present: {RAW_CSV}")
        return

    for url in (UCI_URL, MIRROR_URL):
        try:
            print(f"[data] Downloading from {url} ...")
            urllib.request.urlretrieve(url, RAW_CSV)
            print(f"[data] Saved to {RAW_CSV}")
            return
        except Exception as e:                       # try the next source
            print(f"[data]   failed: {e}")
    raise RuntimeError("Could not download the dataset from any source.")


def load_data() -> pd.DataFrame:
    """Load the dataset and run light validation so downstream code is safe.

    We validate here (not everywhere) so that a corrupt/short file fails fast
    with a clear message instead of causing confusing errors later.
    """
    if not RAW_CSV.exists():
        download_data()

    # encoding='utf-8-sig' strips the byte-order-mark (BOM) that Excel/UCI adds,
    # otherwise the first column name would be '\ufeffUDI' and break lookups.
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")

    # --- basic sanity checks (contract the data must satisfy) ---
    assert len(df) == 10_000, f"Expected 10,000 rows, got {len(df)}"
    assert TARGET in df.columns, f"Target column '{TARGET}' missing"
    for col in NUMERIC:
        assert col in df.columns, f"Numeric column '{col}' missing"

    return df


if __name__ == "__main__":
    # Running this file directly downloads + loads and prints a quick summary.
    df = load_data()
    print(df.shape)
    print(df.head())
