import os
import argparse
import shutil
from typing import Tuple, List


def parse_args():
    ap = argparse.ArgumentParser(
        description="Prepare CamVid datasets (32-class and/or 11-class) into train/val/test folder structure"
    )
    ap.add_argument("--camvid-root", default="CamVid", help="Path to CamVid root folder")
    ap.add_argument(
        "--out-root",
        default=os.path.join("CamVid", "converted"),
        help="Output root to place converted datasets",
    )
    ap.add_argument(
        "--sets",
        default="both",
        choices=["11", "32", "both"],
        help="Which dataset(s) to prepare",
    )
    ap.add_argument(
        "--link",
        dest="use_symlink",
        action="store_true",
        help="Use symlink instead of copying files (saves disk)",
    )
    ap.add_argument(
        "--copy",
        dest="use_symlink",
        action="store_false",
        help="Copy files instead of symlink",
    )
    ap.set_defaults(use_symlink=True)
    return ap.parse_args()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_link_or_copy(src: str, dst: str, use_symlink: bool):
    if os.path.exists(dst):
        return
    dstdir = os.path.dirname(dst)
    os.makedirs(dstdir, exist_ok=True)
    if use_symlink:
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except Exception:
            # fallback to copy
            pass
    shutil.copy2(src, dst)


def read_split_list(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_path = parts[0]
            lbl_path = parts[1]
            # Only keep basenames; our local folders are flat
            img_base = os.path.basename(img_path)
            lbl_base = os.path.basename(lbl_path)
            pairs.append((img_base, lbl_base))
    return pairs


def build_pairs(camvid_root: str, split_file: str) -> List[Tuple[str, str]]:
    pairs_in_list = read_split_list(split_file)
    # local sources
    img_src_dir = os.path.join(camvid_root, "CamVid_RGB")
    # 32-class color labels live in CamVid_Label, 11-class labels in CamVidColor11 (RGB) or CamVidGray (indexed)
    lbl32_src_dir = os.path.join(camvid_root, "CamVid_Label")
    lbl11_gray_dir = os.path.join(camvid_root, "CamVidGray")
    lbl11_color_dir = os.path.join(camvid_root, "CamVidColor11")

    # We only return basenames; actual resolution for 11/32 happens in prepare functions
    return pairs_in_list


def prepare_one_set(
    camvid_root: str,
    out_root: str,
    set_name: str,
    split: str,
    use_symlink: bool,
    labels_source: str,
):
    """
    labels_source: one of {"32-color", "11-gray", "11-color"}
    - "32-color": labels from CamVid_Label (RGB 32 classes)
    - "11-gray" : labels from CamVidGray (indexed 11 classes)
    - "11-color": labels from CamVidColor11 (RGB 11 classes)
    """
    assert labels_source in {"32-color", "11-gray", "11-color"}

    split_file = os.path.join(camvid_root, "SegNetanno", f"{split}.txt")
    pairs = build_pairs(camvid_root, split_file)

    img_src_dir = os.path.join(camvid_root, "CamVid_RGB")
    if labels_source == "32-color":
        lbl_src_dir = os.path.join(camvid_root, "CamVid_Label")
        # list entries use file.png, but our files are file_L.png
        def map_label_name(lbl_base: str) -> str:
            base, ext = os.path.splitext(lbl_base)
            return f"{base}_L{ext}"
    elif labels_source == "11-gray":
        lbl_src_dir = os.path.join(camvid_root, "CamVidGray")
        def map_label_name(lbl_base: str) -> str:
            base, ext = os.path.splitext(lbl_base)
            return f"{base}_L{ext}"
    else:  # "11-color"
        lbl_src_dir = os.path.join(camvid_root, "CamVidColor11")
        def map_label_name(lbl_base: str) -> str:
            base, ext = os.path.splitext(lbl_base)
            return f"{base}_L{ext}"

    images_out = os.path.join(out_root, set_name, split, "images")
    labels_out = os.path.join(out_root, set_name, split, "labels")
    ensure_dir(images_out)
    ensure_dir(labels_out)

    n_ok, n_miss = 0, 0
    missing = []
    for img_base, lbl_base in pairs:
        img_src = os.path.join(img_src_dir, img_base)
        lbl_src = os.path.join(lbl_src_dir, map_label_name(lbl_base))
        if not os.path.isfile(img_src) or not os.path.isfile(lbl_src):
            n_miss += 1
            if not os.path.isfile(img_src):
                missing.append(f"IMG:{img_base}")
            if not os.path.isfile(lbl_src):
                missing.append(f"LBL:{map_label_name(lbl_base)}")
            continue
        img_dst = os.path.join(images_out, img_base)
        lbl_dst = os.path.join(labels_out, os.path.basename(lbl_src))
        safe_link_or_copy(img_src, img_dst, use_symlink)
        safe_link_or_copy(lbl_src, lbl_dst, use_symlink)
        n_ok += 1

    print(f"[{set_name}|{split}] prepared {n_ok} pairs, missing {n_miss}")
    if n_miss:
        print("  First few missing:", ", ".join(missing[:10]))


def main():
    args = parse_args()
    camvid_root = args.camvid_root
    out_root = args.out_root
    use_symlink = args.use_symlink

    # Prepare 32-class dataset
    if args.sets in ("both", "32"):
        for split in ("train", "val", "test"):
            prepare_one_set(camvid_root, out_root, "camvid32", split, use_symlink, labels_source="32-color")

    # Prepare 11-class dataset (prefer gray indexed labels)
    if args.sets in ("both", "11"):
        for split in ("train", "val", "test"):
            prepare_one_set(camvid_root, out_root, "camvid11", split, use_symlink, labels_source="11-gray")

    print("Done. Example paths to use in configs:")
    print(f"  32-class train images: {os.path.join(out_root, 'camvid32', 'train', 'images')}")
    print(f"  32-class train labels: {os.path.join(out_root, 'camvid32', 'train', 'labels')}")
    print(f"  11-class train images: {os.path.join(out_root, 'camvid11', 'train', 'images')}")
    print(f"  11-class train labels: {os.path.join(out_root, 'camvid11', 'train', 'labels')}")


if __name__ == "__main__":
    main()
