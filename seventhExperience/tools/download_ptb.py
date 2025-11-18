#!/usr/bin/env python3
# 下载并准备 PTB (Penn Treebank) 数据集，适配 PyTorch examples 的数据读取格式
# 数据来源：http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

import os
import tarfile
import urllib.request

PTB_URL = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'ptb')
TMP_TGZ = os.path.join(ROOT, 'data', 'simple-examples.tgz')


def main():
    os.makedirs(os.path.dirname(TMP_TGZ), exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(TMP_TGZ):
        print('Downloading PTB from', PTB_URL)
        urllib.request.urlretrieve(PTB_URL, TMP_TGZ)
        print('Downloaded to', TMP_TGZ)
    else:
        print('Found existing', TMP_TGZ)

    with tarfile.open(TMP_TGZ, 'r:gz') as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(('.train.txt', '.valid.txt', '.test.txt'))]
        # typical paths: simple-examples/data/ptb.train.txt etc
        tf.extractall(path=os.path.join(ROOT, 'data'), members=members)
        print('Extracted PTB text files')

    # Move/rename to train.txt/valid.txt/test.txt expected by examples/Corpus
    # The extracted files are under data/simple-examples/data/
    extracted_dir = os.path.join(ROOT, 'data', 'simple-examples', 'data')
    mapping = {
        'ptb.train.txt': 'train.txt',
        'ptb.valid.txt': 'valid.txt',
        'ptb.test.txt': 'test.txt',
    }
    for src, dst in mapping.items():
        src_path = os.path.join(extracted_dir, src)
        dst_path = os.path.join(DATA_DIR, dst)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f'Missing {src_path}. Archive layout may have changed.')
        with open(src_path, 'r', encoding='utf-8') as fsrc, open(dst_path, 'w', encoding='utf-8') as fdst:
            # Keep content as-is; Corpus will add <eos> tokens per line.
            fdst.write(fsrc.read())
        print('Wrote', dst_path)

    print('PTB prepared under', DATA_DIR)


if __name__ == '__main__':
    main()
