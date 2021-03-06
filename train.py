#! /usr/bin/env python3


import sys
import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from model.bert import BertRanker


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('DATA_DIR', help='Folder with all preprocessed files')
    ap.add_argument('FOLD_NAME', help='Name of the fold (within DATA_DIR)')

    # trainer args
    # Trainer.add_argparse_args would make the help too cluttered
    ap.add_argument('--accumulate_grad_batches', type=int, default=1, help='Update weights after this many batches')
    ap.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
    ap.add_argument('--gpus', type=int, nargs='+', help='GPU IDs to train on')
    ap.add_argument('--val_check_interval', type=float, default=1.0, help='Validation check interval')
    ap.add_argument('--save_top_k', type=int, default=1, help='Save top-k checkpoints')
    ap.add_argument('--limit_val_batches', type=int, default=sys.maxsize, help='Use a subset of validation data')
    ap.add_argument('--limit_train_batches', type=int, default=sys.maxsize, help='Use a subset of training data')
    ap.add_argument('--limit_test_batches', type=int, default=sys.maxsize, help='Use a subset of test data')
    ap.add_argument('--precision', type=int, choices=[16, 32], default=32, help='Floating point precision')
    ap.add_argument('--accelerator', default='ddp', help='Distributed backend (accelerator)')

    # model args
    BertRanker.add_model_specific_args(ap)

    # remaining args
    ap.add_argument('--val_patience', type=int, default=3, help='Validation patience')
    ap.add_argument('--save_dir', default='out', help='Directory for logs, checkpoints and predictions')
    ap.add_argument('--random_seed', type=int, default=123, help='Random seed')
    ap.add_argument('--load_weights', help='Load pre-trained weights before training')
    ap.add_argument('--test', action='store_true', help='Test the model after training')
    args = ap.parse_args()

    # in DDP mode we always need a random seed
    seed_everything(args.random_seed)

    data_dir = Path(args.DATA_DIR)
    args.data_file = data_dir / 'data.h5'
    args.train_file_pointwise = data_dir / args.FOLD_NAME / 'train_pointwise.h5'
    args.train_file_pairwise = data_dir / args.FOLD_NAME / 'train_pairwise.h5'
    args.val_file = data_dir / args.FOLD_NAME / 'val.h5'
    args.test_file = data_dir / args.FOLD_NAME / 'test.h5'
    model = BertRanker(vars(args))

    if args.load_weights:
        weights = torch.load(args.load_weights)
        model.load_state_dict(weights['state_dict'])

    early_stopping = EarlyStopping(monitor='val_map', mode='max', patience=args.val_patience, verbose=True)
    model_checkpoint = ModelCheckpoint(monitor='val_map', mode='max', save_top_k=args.save_top_k, verbose=True)
    trainer = Trainer.from_argparse_args(args, deterministic=True,
                                         replace_sampler_ddp=False,
                                         default_root_dir=args.save_dir,
                                         callbacks=[LearningRateMonitor(), early_stopping, model_checkpoint])
    trainer.fit(model)
    if args.test:
        trainer.test()


if __name__ == '__main__':
    main()
