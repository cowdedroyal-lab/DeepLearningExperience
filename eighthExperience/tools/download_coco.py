"""
MSCOCO 2014 数据集下载脚本（图像与注释）。

说明：
- 下载来源与目录结构基于 COCO 官方提供的 2014 版链接：
  images: http://images.cocodataset.org/zips/train2014.zip / val2014.zip
  annotations: http://images.cocodataset.org/annotations/annotations_trainval2014.zip
- 文件较大（数 GB），请确保磁盘空间与网络环境。
"""

from __future__ import annotations

import argparse
import os
import zipfile
import requests
from tqdm import tqdm

from src.utils.common import ensure_dir


COCO_URLS = {
	"train_images": "http://images.cocodataset.org/zips/train2014.zip",
	"val_images": "http://images.cocodataset.org/zips/val2014.zip",
	"annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}


def download_file(url: str, dst_path: str, chunk_size: int = 8192):
	ensure_dir(os.path.dirname(dst_path) or ".")
	with requests.get(url, stream=True) as r:
		r.raise_for_status()
		total = int(r.headers.get("Content-Length", 0))
		with open(dst_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dst_path)) as pbar:
			for chunk in r.iter_content(chunk_size=chunk_size):
				if chunk:
					f.write(chunk)
					pbar.update(len(chunk))


def extract_zip(zip_path: str, dst_dir: str):
	ensure_dir(dst_dir)
	with zipfile.ZipFile(zip_path, "r") as zf:
		zf.extractall(dst_dir)


def main():
	parser = argparse.ArgumentParser(description="Download MSCOCO 2014 images and annotations")
	parser.add_argument("--out_dir", type=str, default="data/coco", help="下载并解压的目标根目录")
	parser.add_argument("--skip_images", action="store_true", help="仅下载注释，不下载图片")
	parser.add_argument("--skip_annotations", action="store_true", help="仅下载图片，不下载注释")
	args = parser.parse_args()

	out_root = args.out_dir
	ensure_dir(out_root)

	# 下载与解压注释
	if not args.skip_annotations:
		ann_zip = os.path.join(out_root, "annotations_trainval2014.zip")
		if not os.path.exists(ann_zip):
			download_file(COCO_URLS["annotations"], ann_zip)
		extract_zip(ann_zip, out_root)

	# 下载与解压图片
	if not args.skip_images:
		for key, folder in [("train_images", "train2014"), ("val_images", "val2014")]:
			zip_name = os.path.basename(COCO_URLS[key])
			zip_path = os.path.join(out_root, zip_name)
			if not os.path.exists(zip_path):
				download_file(COCO_URLS[key], zip_path)
			extract_zip(zip_path, out_root)

	print("Done. Directory layout:")
	print(f"{out_root}/")
	print("  annotations/")
	print("  train2014/")
	print("  val2014/")


if __name__ == "__main__":
	main()
