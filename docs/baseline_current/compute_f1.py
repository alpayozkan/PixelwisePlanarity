"""Compute F1 = 2*P*R/(P+R) for all Precision/Recall threshold pairs in baseline CSVs.

Usage:
    python compute_f1.py                          # process all CSVs in this directory
    python compute_f1.py scannetpp_*.csv           # process specific files
    python compute_f1.py --output-dir ./with_f1/   # write to a different directory
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def add_f1_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Insert F1@X column after each (P@X, R@X) pair."""
    # Find all threshold tags: e.g. "0.1cm", "0.5cm", "2.0cm"
    prec_cols = [c for c in df.columns if c.startswith("P@")]
    thresholds = [c[2:] for c in prec_cols]  # strip "P@"

    # Build new column order with F1 inserted after each R@X
    new_cols = []
    f1_data = {}
    for col in df.columns:
        new_cols.append(col)
        # Check if this is a R@X column
        match = re.match(r"^R@(.+)$", col)
        if match:
            tag = match.group(1)
            p_col = f"P@{tag}"
            r_col = f"R@{tag}"
            f1_col = f"F1@{tag}"
            if p_col in df.columns and r_col in df.columns:
                p = df[p_col].astype(float)
                r = df[r_col].astype(float)
                denom = p + r
                f1 = (2 * p * r / denom).where(denom > 0, 0.0)
                f1_data[f1_col] = f1.round(3)
                new_cols.append(f1_col)

    # Add F1 columns to dataframe and reorder
    for col_name, col_data in f1_data.items():
        df[col_name] = col_data

    return df[new_cols]


def main():
    parser = argparse.ArgumentParser(description="Add F1 columns to baseline CSVs")
    parser.add_argument("files", nargs="*", help="CSV files to process (default: all in this dir)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same dir, with _f1 suffix)")
    parser.add_argument("--suffix", type=str, default="_f1",
                        help="Suffix to add to output filenames (default: _f1)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    if args.files:
        csv_files = [Path(f) for f in args.files]
    else:
        csv_files = sorted(base_dir.glob("*.csv"))
        # Exclude already-computed F1 files
        csv_files = [f for f in csv_files if "_f1" not in f.stem]

    if not csv_files:
        print("No CSV files found.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        # Check if there are P@/R@ columns
        prec_cols = [c for c in df.columns if c.startswith("P@")]
        if not prec_cols:
            print(f"[SKIP] {csv_path.name} — no P@/R@ columns found")
            continue

        df_f1 = add_f1_columns(df)

        out_name = csv_path.stem + args.suffix + csv_path.suffix
        out_path = output_dir / out_name
        df_f1.to_csv(out_path, index=False)

        # Print summary
        thresholds = [c[2:] for c in prec_cols]
        print(f"[OK] {csv_path.name} → {out_name}")
        print(f"     Thresholds: {', '.join(thresholds)}")
        print(f"     Methods: {', '.join(df['Method'].tolist())}")

        # Print compact table
        f1_cols = [c for c in df_f1.columns if c.startswith("F1@")]
        print(f"     {'Method':<25s} " + " ".join(f"{c:>9s}" for c in f1_cols))
        for _, row in df_f1.iterrows():
            vals = " ".join(f"{row[c]:>9.3f}" for c in f1_cols)
            print(f"     {row['Method']:<25s} {vals}")
        print()


if __name__ == "__main__":
    main()
