"""
图像描述评价指标: BLEU/METEOR/ROUGE-L/CIDEr（简化实现，教学用途）。

实现说明：
- BLEU-1/4: 使用 nltk.translate.bleu_score 的 corpus_bleu + smoothing。
- METEOR: 使用 nltk.meteor_score（若缺少 wordnet 资源，自动降级为基于重叠的简化分数）。
- ROUGE-L: 句子级 LCS F1 的平均。
- CIDEr: 简化实现，参照 COCO-Caption 思想（TF-IDF + 余弦相似），但仅用于教学评估，非官方实现。

输入格式：
- gts: Dict[str, List[str]]  每个 key 对应多个参考字幕
- res: Dict[str, List[str]]  每个 key 对应单个模型生成字幕（列表长度为 1）
两者的 keys 必须一致。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def _normalize(s: str) -> List[str]:
	import re
	s = s.strip().lower()
	s = re.sub(r"[^a-z0-9\s]", "", s)
	toks = s.split()
	return toks


# ---------------------- BLEU ----------------------
def compute_bleu(res: Dict[str, List[str]], gts: Dict[str, List[str]]) -> Tuple[float, float, float, float]:
	try:
		from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
	except Exception:
		# 若未安装 nltk，返回 0
		return 0.0, 0.0, 0.0, 0.0

	assert res.keys() == gts.keys()
	keys = list(res.keys())
	references = [[_normalize(r) for r in gts[k]] for k in keys]  # list of list-of-tokens
	hypotheses = [[_normalize(res[k][0])] for k in keys]          # list of list-of-tokens (single)
	smoothing = SmoothingFunction().method3

	# BLEU-n 使用不同的权重
	bleu1 = corpus_bleu(references, [h[0] for h in hypotheses], weights=(1, 0, 0, 0), smoothing_function=smoothing)
	bleu2 = corpus_bleu(references, [h[0] for h in hypotheses], weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
	bleu3 = corpus_bleu(references, [h[0] for h in hypotheses], weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing)
	bleu4 = corpus_bleu(references, [h[0] for h in hypotheses], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
	return bleu1, bleu2, bleu3, bleu4


# ---------------------- METEOR ----------------------
def compute_meteor(res: Dict[str, List[str]], gts: Dict[str, List[str]]) -> float:
	try:
		from nltk.translate.meteor_score import meteor_score
	except Exception:
		# 降级为基于 unigram 的 F1
		return _overlap_f1_corpus(res, gts)

	scores = []
	for k in res.keys():
		# NLTK meteor_score 预期: hypothesis 和 references 使用同一种类型
		# 这里统一做分词，传入 List[str]，避免 TypeError: expects pre-tokenized hypothesis
		cand_raw = res[k][0]
		refs_raw = gts[k]
		cand = _normalize(cand_raw)
		refs = [_normalize(r) for r in refs_raw]
		try:
			score = meteor_score(refs, cand)
		except LookupError:
			# 缺少 wordnet 等资源
			# 退回到基于字符串的重叠 F1 计算
			score = _overlap_f1(cand_raw, refs_raw)
		scores.append(score)
	return float(np.mean(scores)) if scores else 0.0


def _overlap_f1(cand: str, refs: List[str]) -> float:
	c = set(_normalize(cand))
	best = 0.0
	for r in refs:
		rr = set(_normalize(r))
		if not c and not rr:
			best = max(best, 1.0)
			continue
		if not c or not rr:
			continue
		p = len(c & rr) / max(1, len(c))
		r_ = len(c & rr) / max(1, len(rr))
		if p + r_ == 0:
			f1 = 0.0
		else:
			f1 = 2 * p * r_ / (p + r_)
		best = max(best, f1)
	return best


def _overlap_f1_corpus(res: Dict[str, List[str]], gts: Dict[str, List[str]]) -> float:
	return float(np.mean([_overlap_f1(res[k][0], gts[k]) for k in res.keys()])) if res else 0.0


# ---------------------- ROUGE-L ----------------------
def _lcs(X: List[str], Y: List[str]) -> int:
	# 经典 LCS DP
	m, n = len(X), len(Y)
	dp = [[0] * (n + 1) for _ in range(m + 1)]
	for i in range(1, m + 1):
		for j in range(1, n + 1):
			if X[i - 1] == Y[j - 1]:
				dp[i][j] = dp[i - 1][j - 1] + 1
			else:
				dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
	return dp[m][n]


def compute_rouge_l(res: Dict[str, List[str]], gts: Dict[str, List[str]]) -> float:
	scores = []
	for k in res.keys():
		cand = _normalize(res[k][0])
		best = 0.0
		for r in gts[k]:
			ref = _normalize(r)
			lcs = _lcs(cand, ref)
			if len(cand) == 0 or len(ref) == 0:
				f1 = 0.0
			else:
				p = lcs / len(cand)
				rec = lcs / len(ref)
				if p + rec == 0:
					f1 = 0.0
				else:
					f1 = 2 * p * rec / (p + rec)
			best = max(best, f1)
		scores.append(best)
	return float(np.mean(scores)) if scores else 0.0


# ---------------------- CIDEr (简化) ----------------------
def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
	counts = {}
	for i in range(len(tokens) - n + 1):
		ng = tuple(tokens[i:i+n])
		counts[ng] = counts.get(ng, 0) + 1
	return counts


def _tfidf_vector(counts: Dict[Tuple[str, ...], int], idf: Dict[Tuple[str, ...], float]) -> Tuple[Dict[Tuple[str, ...], float], float]:
	vec = {}
	norm_sq = 0.0
	for ng, tf in counts.items():
		w = (tf) * idf.get(ng, 0.0)
		vec[ng] = w
		norm_sq += w * w
	return vec, math.sqrt(norm_sq) if norm_sq > 0 else 1.0


def _cosine(vec1, norm1, vec2, norm2) -> float:
	# 稀疏向量点积
	if len(vec1) < len(vec2):
		v_small, v_large = vec1, vec2
	else:
		v_small, v_large = vec2, vec1
	dot = 0.0
	for k, w in v_small.items():
		if k in v_large:
			dot += w * v_large[k]
	denom = (norm1 * norm2) if norm1 * norm2 > 0 else 1.0
	return dot / denom


def compute_cider(res: Dict[str, List[str]], gts: Dict[str, List[str]], n: int = 4) -> float:
	# 构建 DF: 基于所有参考 captions 的 n-grams
	df = [{} for _ in range(n)]  # list of dict for each n
	eps = 1e-12
	for refs in gts.values():
		ref_ngrams_set = [set() for _ in range(n)]
		for r in refs:
			toks = _normalize(r)
			for i in range(n):
				ref_ngrams_set[i].update(_ngram_counts(toks, i+1).keys())
		for i in range(n):
			for ng in ref_ngrams_set[i]:
				df[i][ng] = df[i].get(ng, 0) + 1

	# 计算 idf
	M = len(gts)
	idf = [
		{ng: math.log((M + eps) / (df_i.get(ng, 0) + eps)) for ng in df_i.keys()}
		for df_i in df
	]

	scores = []
	for k in res.keys():
		cand_toks = _normalize(res[k][0])
		# 候选向量（拼接 1..n-gram 的 tf-idf）
		vec_c = {}
		norm_c_sq = 0.0
		for i in range(n):
			counts_c = _ngram_counts(cand_toks, i+1)
			vec_i, norm_i = _tfidf_vector(counts_c, idf[i])
			for ng, w in vec_i.items():
				vec_c[(i+1, ng)] = w
			norm_c_sq += norm_i * norm_i
		norm_c = math.sqrt(norm_c_sq) if norm_c_sq > 0 else 1.0

		# 与每个参考的相似度
		sims = []
		for r in gts[k]:
			ref_toks = _normalize(r)
			vec_r = {}
			norm_r_sq = 0.0
			for i in range(n):
				counts_r = _ngram_counts(ref_toks, i+1)
				vec_i, norm_i = _tfidf_vector(counts_r, idf[i])
				for ng, w in vec_i.items():
					vec_r[(i+1, ng)] = w
				norm_r_sq += norm_i * norm_i
			norm_r = math.sqrt(norm_r_sq) if norm_r_sq > 0 else 1.0
			sims.append(_cosine(vec_c, norm_c, vec_r, norm_r))

		score = float(np.mean(sims)) * 10.0 if sims else 0.0  # 与 COCO-Caption 结果同量级
		scores.append(score)

	return float(np.mean(scores)) if scores else 0.0


# ---------------------- 汇总接口 ----------------------
def compute_corpus_metrics(res: Dict[str, List[str]], gts: Dict[str, List[str]]) -> Dict[str, float]:
	assert res.keys() == gts.keys()
	b1, b2, b3, b4 = compute_bleu(res, gts)
	meteor = compute_meteor(res, gts)
	rouge_l = compute_rouge_l(res, gts)
	cider = compute_cider(res, gts)
	return {
		"BLEU-1": b1,
		"BLEU-2": b2,
		"BLEU-3": b3,
		"BLEU-4": b4,
		"METEOR": meteor,
		"ROUGE-L": rouge_l,
		"CIDEr": cider,
	}

