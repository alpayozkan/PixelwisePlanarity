#!/usr/bin/env python3
"""Add F1 columns after each P@/R@ pair in a baseline table CSV.

Usage:
    python add_f1_to_table.py table_combined_baselines_v1.csv
    python add_f1_to_table.py table1.csv table2.csv table3.csv
"""

import sys
import re
import pandas as pd
import numpy as np


def add_f1_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Insert F1@Xcm column after each (P@Xcm, R@Xcm) pair."""
    new_cols = []
    seen_thresholds = set()

    for col in df.columns:
        new_cols.append(col)
        m = re.match(r"R@(.+)", col)
        if m:
            thresh = m.group(1)
            p_col = f"P@{thresh}"
            f1_col = f"F1@{thresh}"
            if p_col in df.columns and thresh not in seen_thresholds:
                p = df[p_col].values.astype(float)
                r = df[col].values.astype(float)
                denom = p + r
                f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
                df[f1_col] = f1
                new_cols.append(f1_col)
                seen_thresholds.add(thresh)

    return df[new_cols]


def main():
    if len(sys.argv) < 2:
        print("Usage: python add_f1_to_table.py <csv_file> [csv_file ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        df = pd.read_csv(path)
        df_out = add_f1_columns(df)
        out_path = path.replace(".csv", "_f1.csv")
        df_out.to_csv(out_path, index=False)
        print(f"{path} -> {out_path}  ({len(df_out)} rows, added F1 columns)")


if __name__ == "__main__":
    main()
