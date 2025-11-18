"""
从 COCO 验证集随机选择若干图片，执行推理生成字幕，并以评估指标选出得分最高的 Top-K 张图片。

输出：
- 将 Top-K 图片拷贝到 outputs_predvis/
- 对应的字幕与指标写入同名 .txt 文件

用法示例：
    python tools/eval_infer_topk.py \
        --checkpoint checkpoints/coco_caption_best.pt \
        --config configs/coco_baseline.yaml \
        --num_samples 100 --top_k 3 --metric CIDEr \
        --output_dir outputs_predvis

说明：
- 优先从 checkpoint 恢复 tokenizer（与训练一致）。
- 评估指标默认使用 CIDEr（本仓库提供的简化实现，无需 Java）。
- 随机抽样对象为图像（非注释），避免同一图像多条 caption 干扰排序。
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import shutil
from typing import Dict, Any, List, Tuple

import torch
from PIL import Image

# 确保可以以模块形式导入 src/*
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.common import load_config, set_seed
from src.datasets.vocab import BertTokenizerWrapper
from src.datasets.word_vocab import WordVocab
from src.datasets.coco_caption import resolve_coco_paths, build_default_transforms
from src.models.transformer_decoder import CaptionTransformer
from src.utils.metrics import compute_corpus_metrics


def build_tokenizer_from_checkpoint_or_cfg(cfg: Dict[str, Any], checkpoint: str, device: torch.device):
    state = torch.load(checkpoint, map_location=device)
    saved_tok_cfg = state.get("tokenizer", {}) if isinstance(state, dict) else {}
    tok_type = (saved_tok_cfg.get("type") or cfg.get("tokenizer", {}).get("type") or "bert").lower()
    if tok_type == "word" or "word_map" in saved_tok_cfg:
        word_map = saved_tok_cfg.get("word_map")
        if not word_map:
            raise RuntimeError(
                "当前 checkpoint 未保存 word_map，无法恢复词表分词。请使用保存了 tokenizer.word_map 的模型或切换到 BERT tokenizer。"
            )
        inv = (
            {int(i): w for w, i in word_map.items()}
            if all(isinstance(v, int) for v in word_map.values())
            else {int(v): k for k, v in word_map.items()}
        )
        tokenizer = WordVocab(word_map=word_map, inv_map=inv)
        bos_id = tokenizer.start_token_id
        eos_id = tokenizer.end_token_id
    else:
        tokenizer = BertTokenizerWrapper(cfg.get("tokenizer", {}))
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
    return tokenizer, bos_id, eos_id, state


def build_model(tokenizer, bos_id: int, eos_id: int, device: torch.device, state: Dict[str, Any]):
    model = CaptionTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        pad_idx=tokenizer.pad_token_id,
        bos_id=bos_id,
        eos_id=eos_id,
        fine_tune_cnn=True,
    ).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def generate_caption(model, tokenizer, img: Image.Image, transform, device: torch.device, max_len: int = 30) -> str:
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out_ids = model.generate(img_t, max_len=max_len)
        ids = out_ids[0].tolist()
        # 去掉 BOS / 截断到 EOS
        if len(ids) > 0 and ids[0] == tokenizer.bos_token_id:
            ids = ids[1:]
        if tokenizer.eos_token_id in ids:
            ids = ids[: ids.index(tokenizer.eos_token_id)]
        caption = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return caption


def rank_images_by_metric(results: List[Dict[str, Any]], metric_key: str, top_k: int) -> List[Dict[str, Any]]:
    # 若某些分数缺失，视为 0
    def key_fn(x):
        return float(x.get("metrics", {}).get(metric_key, 0.0))

    return sorted(results, key=key_fn, reverse=True)[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/coco_baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=100, help="从 val2014 随机抽取的图像数量")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--metric",
        type=str,
        default="CIDEr",
        choices=["CIDEr", "BLEU-4", "ROUGE-L", "METEOR", "BLEU-1", "BLEU-2", "BLEU-3"],
        help="用于排序选择 Top-K 的指标",
    )
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs_predvis")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device(args.device)

    # Tokenizer & Model
    tokenizer, bos_id, eos_id, state = build_tokenizer_from_checkpoint_or_cfg(cfg, args.checkpoint, device)
    model = build_model(tokenizer, bos_id, eos_id, device, state)

    # 数据路径与 transform
    paths = resolve_coco_paths(cfg)
    resize = tuple(cfg.get("transforms", {}).get("resize", [224, 224]))
    imagenet_norm = bool(cfg.get("transforms", {}).get("imagenet_normalize", True))
    transform = build_default_transforms(resize=resize, imagenet_normalize=imagenet_norm)

    # COCO 验证集句柄
    from pycocotools.coco import COCO  # 延迟导入以加快脚本启动
    coco = COCO(paths.val_ann)

    # 从所有 val 图像中随机抽样（按 image 级别）
    all_img_ids: List[int] = coco.getImgIds()
    if len(all_img_ids) == 0:
        raise RuntimeError("未在 COCO 验证集注释中找到图像。请检查配置与数据路径。")

    random.Random(args.seed).shuffle(all_img_ids)
    sample_img_ids = all_img_ids[: max(1, min(args.num_samples, len(all_img_ids)))]

    # 逐张生成字幕与计算单图指标
    results: List[Dict[str, Any]] = []
    for img_id in sample_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info["file_name"]
        image_path = os.path.join(paths.val_dir, filename)
        if not os.path.exists(image_path):
            # 若找不到原图像，跳过
            continue

        img = Image.open(image_path).convert("RGB")
        caption = generate_caption(model, tokenizer, img, transform, device, max_len=args.max_len)

        # GT captions
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        refs = [a["caption"] for a in anns] if anns else []

        # 以单图形式计算指标（简化实现，返回各项平均）
        res_map = {str(img_id): [caption]}
        gts_map = {str(img_id): refs if refs else [""]}
        metrics = compute_corpus_metrics(res_map, gts_map)

        results.append(
            {
                "image_id": img_id,
                "image_path": image_path,
                "file_name": filename,
                "caption": caption,
                "metrics": metrics,
            }
        )

    if not results:
        raise RuntimeError("没有成功处理任何图像，请检查数据路径与权限。")

    # 选出 Top-K
    topk = rank_images_by_metric(results, args.metric, args.top_k)

    # 输出到目录
    os.makedirs(args.output_dir, exist_ok=True)
    for item in topk:
        src_img_path = item["image_path"]
        stem, ext = os.path.splitext(item["file_name"])
        dst_img_name = f"{stem}_{args.metric}.jpg" if ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp"] else item["file_name"]
        dst_img_path = os.path.join(args.output_dir, dst_img_name)

        # 拷贝图片（保持原始文件名；若非常见图片后缀，则强制 .jpg 命名）
        try:
            shutil.copyfile(src_img_path, dst_img_path)
        except Exception:
            # 若拷贝失败，尝试以 PIL 重新保存为 jpg
            try:
                Image.open(src_img_path).convert("RGB").save(dst_img_path if dst_img_name.endswith(".jpg") else os.path.join(args.output_dir, f"{stem}.jpg"))
            except Exception:
                pass

        # 写出同名 .txt（字幕+指标）
        txt_path = os.path.join(args.output_dir, f"{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"image_id: {item['image_id']}\n")
            f.write(f"file_name: {item['file_name']}\n")
            f.write(f"caption: {item['caption']}\n")
            f.write("metrics:\n")
            for k in sorted(item["metrics"].keys()):
                f.write(f"  {k}: {item['metrics'][k]:.6f}\n")

    # 也打印一份简要结果
    print(f"Saved Top-{len(topk)} results to: {os.path.abspath(args.output_dir)}")
    for i, it in enumerate(topk, 1):
        print(f"[{i}] {it['file_name']} | {args.metric}={it['metrics'].get(args.metric, 0.0):.4f} | {it['caption']}")


if __name__ == "__main__":
    main()
