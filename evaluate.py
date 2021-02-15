#! /usr/bin/env python3


import argparse
from pathlib import Path

import torch
import numpy as np

from qa_utils.util import read_output_files, write_trec_eval_file
from qa_utils.metrics import average_precision, reciprocal_rank


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('OUTPUT_FILES', nargs='+', help='Output files')
    ap.add_argument('--out_file', default='out.tsv', help='Output file to use with TREC-eval')
    ap.add_argument('--rr_k', type=int, default=10, help='Compute MRR@k')
    args = ap.parse_args()

    predictions, labels = read_output_files(map(Path, args.OUTPUT_FILES))
    write_trec_eval_file(Path(args.out_file), predictions, 'bert')

    # compute MAP and MRR@k, in case the dataset is incompatible to trec-eval
    aps, rrs = [], []
    for q_id in predictions:
        q_id_predictions, q_id_labels = [], []
        for doc_id in predictions[q_id]:
            q_id_predictions.append(predictions[q_id][doc_id])
            q_id_labels.append(labels[q_id][doc_id])
        pred_tensor = torch.FloatTensor(q_id_predictions)
        label_tensor = torch.IntTensor(q_id_labels)
        aps.append(average_precision(pred_tensor, label_tensor))
        rrs.append(reciprocal_rank(pred_tensor, label_tensor, args.rr_k))
    print(f'MAP: {np.mean(aps)}')
    print(f'MRR@{args.rr_k}: {np.mean(rrs)}')


if __name__ == '__main__':
    main()
