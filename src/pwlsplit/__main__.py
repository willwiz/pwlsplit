import argparse
from pathlib import Path

parser = argparse.ArgumentParser(prog="pwlsplit")
parser.add_argument("file", type=str, nargs="+", help="Path to the input file(s).")
parser.add_argument("--plot", action="store_true", help="Generate plots for the segmented data.")


def main() -> None:
    args = parser.parse_args()
    files = [Path(v) for f in args.file for v in Path().glob(f)]
    for file in files:
        print(f"Processing file: {file}")
        msg = "This is a placeholder for the main function."
        raise NotImplementedError(msg)


if __name__ == "__main__":
    main()
