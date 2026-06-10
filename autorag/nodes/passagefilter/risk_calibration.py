"""Risk-controlled threshold calibration for passage filtering.

Adapted from *BalanceRAG: Joint Risk Calibration for Cascaded
Retrieval-Augmented Generation* (https://arxiv.org/abs/2605.20084v1).

BalanceRAG frames a filtering / routing threshold as an *operating point*
and certifies, at a target risk level ``alpha`` with confidence
``1 - delta``, the least-conservative threshold whose selection-conditioned
error rate is provably controlled.  It does so with Learn-then-Test style
calibration: a Hoeffding-Bentkus p-value per candidate operating point and a
fixed-sequence ("sequential graphical") test that controls the family-wise
error rate across the lattice of points.

This module ports that inference-time, model-free machinery so AutoRAG can
*derive* a passage-filter threshold from a labelled calibration set instead of
hand-setting it.  Two entry points are provided:

* :func:`calibrate_risk_threshold` - single-branch (1-D lattice) calibration,
  wired into :class:`~autorag.nodes.passagefilter.threshold_cutoff.ThresholdCutoff`.
* :func:`calibrate_joint_thresholds` - the paper's joint two-branch (2-D
  lattice) calibration for cascaded LLM-only / RAG routing.

What is *not* ported: the paper's neural uncertainty estimators and the LLM
backbones themselves.  We consume whatever scalar uncertainty / relevance
scores the surrounding pipeline already produces.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger("AutoRAG")


def _kl_bernoulli(a: float, b: float) -> float:
	"""KL divergence between two Bernoulli distributions, ``a`` and ``b``."""
	eps = 1e-12
	a = min(max(a, eps), 1 - eps)
	b = min(max(b, eps), 1 - eps)
	return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def hoeffding_bentkus_pvalue(risk_hat: float, n: int, alpha: float) -> float:
	"""Hoeffding-Bentkus p-value for the null hypothesis ``R(lambda) > alpha``.

	A small p-value lets us *reject* the null and certify that the operating
	point controls risk at level ``alpha``.  This is the Learn-then-Test
	p-value used by BalanceRAG to certify operating points.

	:param risk_hat: Empirical selection-conditioned risk in ``[0, 1]``.
	:param n: Number of accepted (selected) calibration examples.
	:param alpha: Target risk level.
	:return: A valid p-value in ``[0, 1]``. Returns ``1.0`` (cannot reject)
		when there is no evidence the true risk is below ``alpha``.
	"""
	if n <= 0:
		return 1.0
	if risk_hat >= alpha:
		# Empirical risk already exceeds the target; no evidence to reject.
		return 1.0

	# Bentkus tail bound via the binomial CDF.
	try:
		from scipy.stats import binom

		bentkus = math.e * float(binom.cdf(math.ceil(n * risk_hat), n, alpha))
	except ImportError:  # pragma: no cover - scipy is a hard dependency
		bentkus = 1.0

	# Hoeffding bound through the Bernoulli KL divergence (Chernoff form).
	hoeffding = math.exp(-n * _kl_bernoulli(min(risk_hat, alpha), alpha))

	return float(min(1.0, bentkus, hoeffding))


def _conditional_risk(
	scores: Sequence[float],
	labels: Sequence[int],
	threshold: float,
	reverse: bool,
) -> Tuple[float, int]:
	"""Selection-conditioned empirical risk for one threshold.

	An item is *accepted* (kept by the filter) when its score clears the
	threshold; the risk is the fraction of accepted items that are errors
	(``label == 0``).  Mirrors the keep-rule of ``ThresholdCutoff``.

	:return: ``(empirical_risk, num_accepted)``.
	"""
	if reverse:  # lower score is better -> keep items at or below threshold
		accepted = [lab for s, lab in zip(scores, labels) if s <= threshold]
	else:
		accepted = [lab for s, lab in zip(scores, labels) if s >= threshold]

	n = len(accepted)
	if n == 0:
		return 1.0, 0
	errors = sum(1 for lab in accepted if int(lab) == 0)
	return errors / n, n


@dataclass
class CalibrationResult:
	"""Outcome of a risk-controlled calibration."""

	threshold: float
	risk_hat: float
	pvalue: float
	num_accepted: int
	certified: bool
	target_risk: float
	delta: float
	lattice: List[Tuple[float, float, float]] = field(default_factory=list)


def _default_lattice(
	scores: Sequence[float], num_points: int, reverse: bool
) -> List[float]:
	"""Build a candidate-threshold lattice from the observed score range."""
	lo, hi = min(scores), max(scores)
	if math.isclose(lo, hi):
		return [lo]
	step = (hi - lo) / (num_points - 1)
	grid = [lo + step * i for i in range(num_points)]
	# Order from most conservative (safest, fewest accepted) to least so the
	# fixed-sequence test walks toward higher coverage and stops when it can
	# no longer certify the risk level.
	return sorted(grid, reverse=not reverse)


def calibrate_risk_threshold(
	scores: Sequence[float],
	labels: Sequence[int],
	target_risk: float,
	delta: float = 0.1,
	candidate_thresholds: Optional[Sequence[float]] = None,
	reverse: bool = False,
	num_points: int = 50,
) -> CalibrationResult:
	"""Certify the most permissive threshold whose risk is controlled.

	Implements BalanceRAG's single-branch calibration: the candidate
	thresholds form a 1-D lattice ordered from conservative to permissive; a
	fixed-sequence test walks the lattice, certifying each operating point with
	a Hoeffding-Bentkus p-value and stopping at the first point it cannot
	certify.  The last certified (most permissive) threshold is returned, which
	maximises coverage subject to the risk guarantee.

	:param scores: Per-item uncertainty / relevance scores on a labelled
		calibration set.
	:param labels: Binary correctness / relevance labels (``1`` good, ``0``
		error), aligned with ``scores``.
	:param target_risk: Target selection-conditioned risk ``alpha`` in ``(0, 1)``.
	:param delta: Test level; risk is controlled with confidence ``1 - delta``.
	:param candidate_thresholds: Optional explicit lattice. Defaults to a grid
		over the observed score range.
	:param reverse: If ``True``, lower scores are better (keep ``score <= t``).
	:return: A :class:`CalibrationResult`. ``certified`` is ``False`` when no
		operating point controls the risk; the safest threshold is returned.
	"""
	if not 0.0 < target_risk < 1.0:
		raise ValueError("target_risk must lie in (0, 1).")
	if len(scores) != len(labels):
		raise ValueError("scores and labels must have the same length.")
	if len(scores) == 0:
		raise ValueError("Cannot calibrate on an empty calibration set.")

	lattice = (
		list(candidate_thresholds)
		if candidate_thresholds is not None
		else _default_lattice(scores, num_points, reverse)
	)

	# Score every operating point, then run the fixed-sequence ("sequential
	# graphical") test in ascending-risk order so the most certifiable points
	# are visited first; the walk stops at the first point it cannot certify.
	# Higher coverage = more accepted items, i.e. a lower threshold when
	# ``reverse`` is False and a higher one when it is True.
	stats = []
	trace: List[Tuple[float, float, float]] = []
	for threshold in lattice:
		risk_hat, n = _conditional_risk(scores, labels, threshold, reverse)
		stats.append((threshold, risk_hat, n))
	stats.sort(key=lambda x: (x[1], -x[2]))

	best: Optional[CalibrationResult] = None
	for threshold, risk_hat, n in stats:
		pvalue = hoeffding_bentkus_pvalue(risk_hat, n, target_risk)
		trace.append((threshold, risk_hat, pvalue))
		if pvalue <= delta:
			# Certified; keep the one that accepts the most (highest coverage).
			if best is None or n > best.num_accepted:
				best = CalibrationResult(
					threshold=threshold,
					risk_hat=risk_hat,
					pvalue=pvalue,
					num_accepted=n,
					certified=True,
					target_risk=target_risk,
					delta=delta,
				)
		else:
			break

	if best is None:
		# Nothing could be certified; fall back to the safest operating point.
		threshold, risk_hat, n = stats[0]
		pvalue = hoeffding_bentkus_pvalue(risk_hat, n, target_risk)
		logger.warning(
			"Risk calibration could not certify any threshold at "
			"alpha=%.3f, delta=%.3f; falling back to the most conservative "
			"operating point (threshold=%.4f, empirical risk=%.3f).",
			target_risk,
			delta,
			threshold,
			risk_hat,
		)
		best = CalibrationResult(
			threshold=threshold,
			risk_hat=risk_hat,
			pvalue=pvalue,
			num_accepted=n,
			certified=False,
			target_risk=target_risk,
			delta=delta,
		)

	best.lattice = trace
	logger.info(
		"Risk-calibrated threshold=%.4f (empirical risk=%.3f, accepted=%d, "
		"certified=%s) at alpha=%.3f, delta=%.3f.",
		best.threshold,
		best.risk_hat,
		best.num_accepted,
		best.certified,
		target_risk,
		delta,
	)
	return best


@dataclass
class JointCalibrationResult:
	"""A certified cascaded operating point ``(llm_threshold, rag_threshold)``."""

	llm_threshold: float
	rag_threshold: float
	risk_hat: float
	coverage: float
	retrieval_rate: float
	pvalue: float
	certified: bool


def calibrate_joint_thresholds(
	llm_scores: Sequence[float],
	rag_scores: Sequence[float],
	labels_llm: Sequence[int],
	labels_rag: Sequence[int],
	target_risk: float,
	delta: float = 0.1,
	num_points: int = 20,
	reverse: bool = False,
) -> Optional[JointCalibrationResult]:
	"""Joint two-branch calibration for cascaded LLM-only / RAG routing.

	This ports BalanceRAG's headline contribution: rather than calibrating each
	stage independently (which is conservative), it searches the 2-D lattice of
	``(llm_threshold, rag_threshold)`` operating points jointly.  A query is
	answered by the LLM-only branch when its score clears ``llm_threshold``,
	escalated to the RAG branch when it does not, accepted there when the RAG
	score clears ``rag_threshold``, and abstained from otherwise.  Each point's
	system-level selection-conditioned risk is certified with a
	Hoeffding-Bentkus p-value under a fixed-sequence (sequential graphical)
	test ordered by predicted risk, controlling the family-wise error rate.

	Among certified points we return the one with the highest coverage, breaking
	ties toward lower retrieval usage - exactly the multi-objective the paper
	optimises (coverage and bounded retrieval calls).

	:param llm_scores: Confidence scores from the LLM-only branch.
	:param rag_scores: Confidence scores from the RAG branch.
	:param labels_llm: Correctness of the LLM-only answers (``1``/``0``).
	:param labels_rag: Correctness of the RAG answers (``1``/``0``).
	:param target_risk: Target system-level risk ``alpha``.
	:param delta: Test level; risk controlled with confidence ``1 - delta``.
	:return: The best certified operating point, or ``None`` if none qualifies.
	"""
	n = len(llm_scores)
	if not (len(rag_scores) == len(labels_llm) == len(labels_rag) == n):
		raise ValueError("All branch inputs must have the same length.")
	if n == 0:
		raise ValueError("Cannot calibrate on an empty calibration set.")

	llm_grid = _default_lattice(llm_scores, num_points, reverse)
	rag_grid = _default_lattice(rag_scores, num_points, reverse)

	def accepts(score: float, threshold: float) -> bool:
		return score <= threshold if reverse else score >= threshold

	# Enumerate every operating point with its empirical statistics.
	points = []
	for lt in llm_grid:
		for rt in rag_grid:
			errors = accepted = retrieved = 0
			for ls, rs, ll, rl in zip(
				llm_scores, rag_scores, labels_llm, labels_rag
			):
				if accepts(ls, lt):  # answered by LLM-only branch
					accepted += 1
					errors += int(int(ll) == 0)
				elif accepts(rs, rt):  # escalated to RAG and accepted
					retrieved += 1
					accepted += 1
					errors += int(int(rl) == 0)
				# else: abstain (not accepted)
			risk_hat = errors / accepted if accepted else 1.0
			points.append(
				{
					"llm": lt,
					"rag": rt,
					"risk": risk_hat,
					"coverage": accepted / n,
					"retrieval_rate": retrieved / n,
					"accepted": accepted,
				}
			)

	# Sequential graphical testing: order points by predicted risk (safest
	# first) so the family-wise error rate is controlled by a fixed sequence.
	points.sort(key=lambda p: p["risk"])
	certified = []
	for p in points:
		pvalue = hoeffding_bentkus_pvalue(p["risk"], p["accepted"], target_risk)
		if pvalue <= delta:
			p["pvalue"] = pvalue
			certified.append(p)
		else:
			break

	if not certified:
		logger.warning(
			"Joint calibration certified no operating point at alpha=%.3f, "
			"delta=%.3f.",
			target_risk,
			delta,
		)
		return None

	# Maximise coverage, then minimise retrieval usage.
	best = max(certified, key=lambda p: (p["coverage"], -p["retrieval_rate"]))
	return JointCalibrationResult(
		llm_threshold=best["llm"],
		rag_threshold=best["rag"],
		risk_hat=best["risk"],
		coverage=best["coverage"],
		retrieval_rate=best["retrieval_rate"],
		pvalue=best["pvalue"],
		certified=True,
	)
