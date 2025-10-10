from src.psycomark.data.data_pipeline import main as _main

if __name__ == "__main__":
    import argparse, pathlib

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--output-root", default="data/derived")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--lsh-bands", type=int, default=8)
    ap.add_argument("--lsh-ham", type=int, default=4)
    args = ap.parse_args()
    # coerce to Paths for your pipeline’s arg handling
    args.data_dir = pathlib.Path(args.data_dir)
    args.output_root = pathlib.Path(args.output_root)
    _main(args)
