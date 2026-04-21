"""
Convert a wide-format codebook (one column per bit) to the fishsim format
used by generate_scene.py and the simulation pipeline.

Input format (e.g. BRBB):
    name,id,bit1,bit2,...,bitN
    Igf2,Igf2,0,0,0,...
    blank0001,blank0001,1,1,0,...

Output format (fishsim / MERlin):
    version,1
    codebook_name,<name>
    bit_names,bit1,bit2,...,bitN
    name,numeric_id,id,barcode
    Igf2,0,Igf2,0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  1  1  0  0  0  0  0  0  0  1  0
    blank0001,-1,blank0001,1  1  1  0  0  0  ...

  - numeric_id is -1 for blank genes, sequential 0, 1, 2, ... for real genes.
  - Barcodes are space-separated bit values (two spaces between each digit,
    matching the existing fishsim codebook convention).

Usage:
    python -m fishsim.scripts.convert_codebook \\
        Z:/MERFISHp/codebook_BRBB_500Markergn_NewAdaptors.csv \\
        fishsim/resources/codebooks/BRBB_500Markergn.csv \\
        --codebook-name BRBB_500Markergn
"""

import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert wide-format codebook to fishsim format"
    )
    p.add_argument("input", help="Path to input codebook CSV (wide format, one column per bit)")
    p.add_argument("output", help="Path to write fishsim-format codebook CSV")
    p.add_argument(
        "--codebook-name",
        default=None,
        help="Name written to the codebook_name header row (default: input filename stem)",
    )
    p.add_argument(
        "--blank-pattern",
        default="blank",
        help="Case-insensitive substring that identifies blank/control genes (default: 'blank')",
    )
    p.add_argument(
        "--bit-prefix",
        default="bit",
        help="Column name prefix used to identify bit columns (default: 'bit')",
    )
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    # Identify bit columns by prefix, preserving their original order
    bit_cols = [c for c in df.columns if c.lower().startswith(args.bit_prefix.lower())]
    if not bit_cols:
        raise ValueError(
            f"No columns starting with '{args.bit_prefix}' found in {args.input}. "
            "Check --bit-prefix."
        )
    n_bits = len(bit_cols)
    print(f"Found {n_bits} bit columns: {bit_cols[0]} … {bit_cols[-1]}")

    # Build barcode strings using two-space separator to match existing fishsim format
    df["barcode"] = (
        df[bit_cols]
        .astype(int)
        .apply(lambda row: "  ".join(row.astype(str)), axis=1)
    )

    # Assign numeric_id: -1 for blanks, sequential for real genes
    is_blank = df["name"].str.lower().str.contains(args.blank_pattern.lower(), regex=False)
    numeric_ids = []
    real_gene_counter = 0
    for blank in is_blank:
        if blank:
            numeric_ids.append(-1)
        else:
            numeric_ids.append(real_gene_counter)
            real_gene_counter += 1
    df["numeric_id"] = numeric_ids

    n_genes = real_gene_counter
    n_blanks = is_blank.sum()
    print(f"{n_genes} real genes, {n_blanks} blank/control genes")

    # Use the 'id' column if present, otherwise fall back to gene name
    id_col = df["id"] if "id" in df.columns else df["name"]

    codebook_name = args.codebook_name or Path(args.input).stem
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="\n") as f:
        f.write(f"version,1\n")
        f.write(f"codebook_name,{codebook_name}\n")
        f.write(f"bit_names,{','.join(bit_cols)}\n")
        f.write("name,numeric_id,id,barcode\n")
        for i, row in df.iterrows():
            f.write(f"{row['name']},{row['numeric_id']},{id_col.iloc[i]},{row['barcode']}\n")

    print(f"Wrote {len(df)} entries to {output_path}")


if __name__ == "__main__":
    main()
