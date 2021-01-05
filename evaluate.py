#! /usr/bin/env python3


import csv
import argparse
from collections import defaultdict

import torch


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('OUTPUT_FILES', nargs='+', help='Output files')
    ap.add_argument('--out_file', default='out.tsv', help='Output file to use with TREC-eval')
    args = ap.parse_args()

    results = defaultdict(dict)
    for f in args.OUTPUT_FILES:
        print(f'reading {f}...')
        for d in torch.load(f):
            results[d['q_id']][d['doc_id']] = d['prediction'][0]

    print(f'writing {args.out_file}...')
    with open(args.out_file, 'w', encoding='utf-8', newline='\n') as fp:
        writer = csv.writer(fp, delimiter='\t')
        for q_id in results:
            ranking = sorted(results[q_id].keys(), key=results[q_id].get, reverse=True)
            for rank, doc_id in enumerate(ranking, 1):
                score = results[q_id][doc_id]
                writer.writerow([q_id, 'Q0', doc_id, rank, score, 'bert'])


if __name__ == '__main__':
    main()
