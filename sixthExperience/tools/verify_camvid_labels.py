import os
import argparse
import numpy as np
from PIL import Image
from collections import Counter


def parse_args():
    ap = argparse.ArgumentParser(description='Scan CamVid labels to summarize class coverage and format')
    ap.add_argument('--labels', required=True, help='Path to label directory (e.g., trainannot)')
    ap.add_argument('--max-files', type=int, default=0, help='Limit files to scan (0 means all)')
    return ap.parse_args()


def main():
    args = parse_args()
    files = [os.path.join(args.labels, f) for f in os.listdir(args.labels) if f.lower().endswith('.png')]
    files.sort()
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit('No PNG files found in ' + args.labels)

    uniq = set()
    counter = Counter()
    modes = Counter()
    for p in files:
        img = Image.open(p)
        modes[img.mode] += 1
        arr = np.array(img)
        vals, counts = np.unique(arr, return_counts=True)
        uniq.update(vals.tolist())
        for v, c in zip(vals, counts):
            counter[int(v)] += int(c)
    uniq_sorted = sorted(int(v) for v in uniq)
    print('Scanned files:', len(files))
    print('Label modes count:', dict(modes))
    print('Unique values:', uniq_sorted)
    print('Min:', uniq_sorted[0], 'Max:', uniq_sorted[-1])
    print('Top-10 class pixel counts:', counter.most_common(10))


if __name__ == '__main__':
    main()
