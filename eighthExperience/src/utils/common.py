"""
通用工具：随机种子、配置解析（YAML）、checkpoint 保存/加载、目录工具等。
"""

from __future__ import annotations

import os
import json
import random
from typing import Any, Dict

import numpy as np
import torch

try:
	import yaml
except Exception:
	yaml = None


def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def load_config(cfg_path: str) -> Dict[str, Any]:
	if yaml is None:
		raise ImportError("需要 PyYAML 支持，请先安装：pip install pyyaml")
	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	return cfg


def save_json(obj: Dict[str, Any], path: str):
	ensure_dir(os.path.dirname(path) or ".")
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(state: Dict[str, Any], path: str):
	ensure_dir(os.path.dirname(path) or ".")
	torch.save(state, path)


def load_checkpoint(path: str, map_location: str | None = None) -> Dict[str, Any]:
	return torch.load(path, map_location=map_location)

