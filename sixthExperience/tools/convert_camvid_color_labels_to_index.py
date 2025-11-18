import os
import argparse
from PIL import Image
import numpy as np
import json


def parse_args():
    ap = argparse.ArgumentParser(description='Convert CamVid RGB color labels to index PNG using a JSON mapping')
    ap.add_argument('--src', required=True, help='Source directory with RGB label PNGs')
    ap.add_argument('--dst', required=True, help='Destination directory for index PNGs')
    ap.add_argument('--mapping', required=True, help='JSON: either {"r,g,b": id} dict or [{"color":[r,g,b], "id":k}] list')
    ap.add_argument('--background', type=int, default=255, help='Index to use for unknown colors (default 255)')
    return ap.parse_args()


def load_mapping(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    table = {}
    if isinstance(data, dict):
        for k, v in data.items():
            parts = [int(x) for x in k.split(',')]
            assert len(parts) == 3, 'Bad key format, expected "r,g,b"'
            table[tuple(parts)] = int(v)
    elif isinstance(data, list):
        for item in data:
            assert 'color' in item and 'id' in item, 'List items must include color and id'
            col = item['color']
            assert isinstance(col, list) and len(col) == 3
            table[tuple(map(int, col))] = int(item['id'])
    else:
        raise ValueError('Unsupported JSON top-level type')
    return table


def convert_one(src_path, dst_path, table, unknown=255):
    img = Image.open(src_path).convert('RGB')
    arr = np.array(img)
    h, w, _ = arr.shape
    out = np.full((h, w), unknown, dtype=np.uint8)
    # vectorized map by building a lookup for each unique color
    flat = arr.reshape(-1, 3)
    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
    for i, color in enumerate(uniq):
        color_t = tuple(int(x) for x in color.tolist())
        idx = table.get(color_t, unknown)
        out.reshape(-1)[inverse == i] = idx
    Image.fromarray(out, mode='L').save(dst_path)


def main():
    args = parse_args()
    os.makedirs(args.dst, exist_ok=True)
    table = load_mapping(args.mapping)
    files = [f for f in os.listdir(args.src) if f.lower().endswith('.png')]
    files.sort()
    for f in files:
        sp = os.path.join(args.src, f)
        dp = os.path.join(args.dst, f)
        convert_one(sp, dp, table, args.background)
        print('Converted', f)


if __name__ == '__main__':
    main()
