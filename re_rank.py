#! /usr/bin/env python3


import os
import tempfile
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Tuple

import h5py
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from ranking_utils.util import write_trec_eval_file

from model.bert import BertRanker
from model.datasets import ValTestDataset


def create_temp_testset(data_file: Path, runfile: Path) -> Tuple[int, str]:
    """Create a re-ranking testset in a temporary file.

    Args:
        data_file (Path): Pre-processed data file containing queries and documents
        runfile (Path): Runfile to re-rank (TREC format)

    Returns:
        Tuple[int, str]: Descriptor and path of the temporary file
    """
    qd_pairs = []
    with open(runfile) as fp:
        for line in fp:
            q_id, _, doc_id, _, _, _ = line.split()
            qd_pairs.append((q_id, doc_id))

    # recover the internal integer query and doc IDs
    int_q_ids = {}
    int_doc_ids = {}
    with h5py.File(data_file, 'r') as fp:
        for int_id, orig_id in enumerate(tqdm(fp['orig_q_ids'])):
            int_q_ids[orig_id] = int_id
        for int_id, orig_id in enumerate(tqdm(fp['orig_doc_ids'])):
            int_doc_ids[orig_id] = int_id

    fd, f = tempfile.mkstemp()
    with h5py.File(f, 'w') as fp:
        num_items = len(qd_pairs)
        ds = {
            'q_ids': fp.create_dataset('q_ids', (num_items,), dtype='int32'),
            'doc_ids': fp.create_dataset('doc_ids', (num_items,), dtype='int32'),
            'labels': fp.create_dataset('labels', (num_items,), dtype='int32'),
            # only used for validation
            'offsets': fp.create_dataset('offsets', (1,), dtype='int32')
        }
        ds['offsets'][0] = 0
        for i, (q_id, doc_id) in enumerate(tqdm(qd_pairs, desc='Saving testset')):
            ds['q_ids'][i] = int_q_ids[q_id]
            ds['doc_ids'][i] = int_doc_ids[doc_id]
            # only used for validation
            ds['labels'][i] = 0
    return fd, f


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('DATA_FILE', help='Preprocessed file containing queries and documents')
    ap.add_argument('CHECKPOINT', help='Model checkpoint')
    ap.add_argument('RUNFILE', help='Runfile to re-rank (TREC format)')
    ap.add_argument('--out_file', default='result.tsv', help='Output TREC runfile')
    ap.add_argument('--batch_size', type=int, default=128, help='Batch size')
    ap.add_argument('--num_workers', type=int, default=16, help='DataLoader workers')
    args = ap.parse_args()

    print(f'loading {args.CHECKPOINT}...')
    kwargs = {'data_file': None, 'train_file': None, 'val_file': None, 'test_file': None,
              'training_mode': None, 'rr_k': None, 'num_workers': None, 'freeze_bert': True}
    model = BertRanker.load_from_checkpoint(args.CHECKPOINT, **kwargs)
    model = DataParallel(model)
    model.to('cuda:0')
    model.eval()

    print('creating temporary testset...')
    fd, f = create_temp_testset(args.DATA_FILE, args.RUNFILE)
    ds = ValTestDataset(args.DATA_FILE, f, model.module.hparams['bert_type'])
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=ds.collate_fn)

    print('ranking...')
    results = defaultdict(dict)
    for q_ids, doc_ids, inputs, _ in tqdm(dl):
        with torch.no_grad():
            inputs = [i.to('cuda:0') for i in inputs]
            outputs = model(inputs)
        for q_id, doc_id, prediction in zip(q_ids, doc_ids, outputs):
            orig_q_id = ds.get_original_query_id(q_id.cpu())
            orig_doc_id = ds.get_original_document_id(doc_id.cpu())
            prediction = prediction.detach().cpu().numpy()[0]
            results[orig_q_id][orig_doc_id] = prediction

    print(f'writing {args.out_file}...')
    write_trec_eval_file(Path(args.out_file), results, 'bert')

    print(f'removing {f}...')
    os.close(fd)
    os.remove(f)


if __name__ == '__main__':
    main()
