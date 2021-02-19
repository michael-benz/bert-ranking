#! /usr/bin/env python3


import argparse
from pathlib import Path

from ranking_utils.util import read_output_files, write_trec_eval_file


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('OUTPUT_FILES', nargs='+', help='Output files')
    ap.add_argument('--out_file', default='out.tsv', help='Output file to use with TREC-eval')
    args = ap.parse_args()

    predictions, _ = read_output_files(map(Path, args.OUTPUT_FILES))
    write_trec_eval_file(Path(args.out_file), predictions, 'bert')


if __name__ == '__main__':
    main()
