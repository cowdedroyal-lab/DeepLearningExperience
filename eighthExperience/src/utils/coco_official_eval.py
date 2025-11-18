"""
COCO 官方评价指标封装（使用 pycocoevalcap 库）。

Copy/Attribution:
- Metrics Implementation: https://github.com/salaniz/pycocoevalcap (derived from https://github.com/tylin/coco-caption)
- Licenses: project-wide BSD-like license (license.txt) and submodule-specific (e.g., BLEU MIT). We only call the library APIs; no code is embedded here. See the linked repositories for full licenses.

说明：
- 本文件仅做薄封装，将我们项目中的预测字典 {image_id: caption} 转换为 pycocotools/pycocoevalcap 期望的输入，
  然后返回官方计算的 BLEU_1..4、METEOR、ROUGE_L、CIDEr、SPICE（若可用）。
- METEOR 与 SPICE 需要 Java 运行环境；若不可用，将自动跳过并继续返回其余指标。
"""
from __future__ import annotations

from typing import Dict, Any, List

try:
    from pycocoevalcap.eval import COCOEvalCap  # type: ignore
except Exception as e:  # pragma: no cover
    COCOEvalCap = None  # 延迟到运行时检查


def coco_official_eval(coco, predictions: Dict[str, str], include_spice: bool | None = None) -> Dict[str, float]:
    """
    使用 pycocoevalcap 的官方实现计算指标。

    参数：
    - coco: pycocotools.coco.COCO 的 ground-truth 句柄（验证/测试集注释）
    - predictions: {str(image_id): caption}
    - include_spice: 是否强制包含 SPICE；默认 None 表示让库自行决定（需 Java 与首次下载模型）。

    返回：
    - 指标字典，包含 Bleu_1..4, METEOR, ROUGE_L, CIDEr，以及可能的 SPICE。
    """
    if COCOEvalCap is None:
        raise ImportError(
            "未安装 pycocoevalcap。请先安装: pip install pycocoevalcap (并确保已安装 pycocotools 与 Java 以支持 METEOR/SPICE)"
        )

    # 将预测转换为 COCO.loadRes 可接受的列表格式
    res_list: List[Dict[str, Any]] = []
    for k, v in predictions.items():
        try:
            img_id = int(k)
        except Exception:
            # 若本身就是 int 字符串失败，直接跳过
            continue
        res_list.append({"image_id": img_id, "caption": v})

    # 载入预测
    coco_res = coco.loadRes(res_list)

    # 构建评估器并指定评估的图像集合（所有提供了结果的 image_id）
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params["image_id"] = list(coco_res.getImgIds())

    # 执行评估
    # SPICE/METEOR 可能因 Java 环境失败，因此逐项捕获异常并继续返回其他指标。
    # 这里直接调用库的一次性 evaluate()，若失败则退回逐个指标（保底）。
    try:
        coco_eval.evaluate()
        metrics = dict(coco_eval.eval)
    except Exception:
        # 逐项计算（复制其内部流程的思想）
        metrics = {}
        try:
            from pycocoevalcap.bleu.bleu import Bleu  # type: ignore
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore
            tokenizer = PTBTokenizer()
            gts = {img_id: coco.imgToAnns[img_id] for img_id in coco_res.getImgIds()}
            res = {img_id: coco_res.imgToAnns[img_id] for img_id in coco_res.getImgIds()}
            gts = tokenizer.tokenize(gts)
            res = tokenizer.tokenize(res)
            bleu = Bleu(4)
            scores, _ = bleu.compute_score(gts, res)
            for i, sc in enumerate(scores, start=1):
                metrics[f"Bleu_{i}"] = sc
        except Exception:
            pass
        # 其他指标尝试
        for mod_name, key in (
            ("pycocoevalcap.cider.cider.Cider", "CIDEr"),
            ("pycocoevalcap.rouge.rouge.Rouge", "ROUGE_L"),
            ("pycocoevalcap.meteor.meteor.Meteor", "METEOR"),
            ("pycocoevalcap.spice.spice.Spice", "SPICE"),
        ):
            try:
                module_path, cls_name = mod_name.rsplit(".", 1)
                mod = __import__(module_path, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                scorer = cls()
                from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore
                tokenizer = PTBTokenizer()
                gts = {img_id: coco.imgToAnns[img_id] for img_id in coco_res.getImgIds()}
                res = {img_id: coco_res.imgToAnns[img_id] for img_id in coco_res.getImgIds()}
                gts = tokenizer.tokenize(gts)
                res = tokenizer.tokenize(res)
                score, _ = scorer.compute_score(gts, res)
                metrics[key] = score
            except Exception:
                continue

    # 兼容旧键名（例如 BLEU-4 / ROUGE-L 大写连字符）
    compat = {}
    if "Bleu_4" in metrics:
        compat["BLEU-4"] = float(metrics["Bleu_4"])  # type: ignore
        compat["BLEU-3"] = float(metrics.get("Bleu_3", 0.0))
        compat["BLEU-2"] = float(metrics.get("Bleu_2", 0.0))
        compat["BLEU-1"] = float(metrics.get("Bleu_1", 0.0))
    if "CIDEr" in metrics:
        compat["CIDEr"] = float(metrics["CIDEr"])  # type: ignore
    if "ROUGE_L" in metrics:
        compat["ROUGE-L"] = float(metrics["ROUGE_L"])  # type: ignore
    if "METEOR" in metrics:
        compat["METEOR"] = float(metrics["METEOR"])  # type: ignore
    if "SPICE" in metrics:
        compat["SPICE"] = float(metrics["SPICE"])  # type: ignore

    # 合并两套键，避免信息丢失
    return {**{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}, **compat}


__all__ = ["coco_official_eval"]
