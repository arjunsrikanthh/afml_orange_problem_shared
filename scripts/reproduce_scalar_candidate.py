#!/usr/bin/env python3
"""Reproduce one scalar candidate seed while retaining the frozen split seed."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
from scripts.train_scalar_mlp import train, split_labels, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-epochs', type=int, default=2, choices=(1, 2))
    parser.add_argument('--source-commit', default='b55bd1b')
    parser.add_argument('--data-dir', type=Path, default=Path('data/23_Team_Toxic'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = {}
    for filename in ('scripts/train_scalar_mlp.py', 'scripts/weight_baselines.py'):
        content = Path(filename).read_bytes()
        committed = subprocess.check_output(['git', 'show', args.source_commit + ':' + filename])
        if content != committed:
            raise ValueError('runtime differs from committed source: ' + filename)
        source[filename] = hashlib.sha256(content).hexdigest()
    revision = subprocess.check_output(['git', 'rev-parse', args.source_commit], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    noisy_path = args.data_dir / 'train_weights_noisy.npy'
    clean_path = args.data_dir / 'train_weights_clean.npy'
    noisy = np.load(noisy_path, mmap_mode='r', allow_pickle=False)
    clean = np.load(clean_path, mmap_mode='r', allow_pickle=False)
    labels, _ = split_labels(noisy.size, seed=2301)
    started = time.monotonic()
    result = train(noisy, clean, labels, config={'seed': args.seed},
                   checkpoint=args.output / 'checkpoint.pt',
                   input_hashes={'clean': sha256_file(clean_path), 'noisy': sha256_file(noisy_path)},
                   device_name='cpu', max_epochs_this_run=args.max_epochs)
    result.update(training_seed=args.seed, split_seed=2301, runtime_source_commit=revision,
                  runtime_source_sha256=source, elapsed_seconds=time.monotonic() - started,
                  maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  checkpoint_sha256=sha256_file(args.output / 'checkpoint.pt'))
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
