"""
文本清洗与辅助函数。
说明：由于本项目使用 transformers 的 BertTokenizer，复杂的分词由其内部完成，这里仅提供轻量级清洗。
"""

import re
from typing import List


def clean_caption(s: str, lower: bool = True) -> str:
	"""基础清洗：可选小写化，移除多余空白。"""
	s = s.strip()
	if lower:
		s = s.lower()
	# 保留常见标点，压缩多空格
	s = re.sub(r"\s+", " ", s)
	return s


def batch_clean(caps: List[str], lower: bool = True) -> List[str]:
	return [clean_caption(c, lower=lower) for c in caps]


__all__ = ["clean_caption", "batch_clean"]
