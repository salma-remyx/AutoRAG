import pytest

# Import the *existing* call-site module to prove the calibration is wired in,
# not just self-tested.
from autorag.nodes.passagefilter import ThresholdCutoff
from autorag.nodes.passagefilter.risk_calibration import (
	calibrate_risk_threshold,
	calibrate_joint_thresholds,
	hoeffding_bentkus_pvalue,
)
from tests.autorag.nodes.passagefilter.test_passage_filter_base import (
	contents_example,
	ids_example,
	project_dir,
	previous_result,
)


# A separable calibration set: high scores (>= 0.7) are relevant (label 1),
# low scores (<= 0.3) are not (label 0). A risk-controlled threshold should
# land in the empty 0.3-0.7 gap. Enough samples to give the test power.
_LOW = [0.05 + 0.005 * i for i in range(50)]  # 0.05 .. 0.295, all irrelevant
_HIGH = [0.70 + 0.005 * i for i in range(50)]  # 0.70 .. 0.945, all relevant
CALIB_SCORES = _LOW + _HIGH
CALIB_LABELS = [0] * 50 + [1] * 50


def test_hoeffding_bentkus_pvalue_basics():
	# No evidence the true risk is below alpha -> cannot reject.
	assert hoeffding_bentkus_pvalue(risk_hat=0.5, n=20, alpha=0.1) == 1.0
	# Strong evidence (zero empirical risk over many samples) -> small p-value.
	assert hoeffding_bentkus_pvalue(risk_hat=0.0, n=100, alpha=0.2) < 0.05
	# More samples at the same empirical risk give a smaller (stronger) p-value.
	assert hoeffding_bentkus_pvalue(0.05, 200, 0.2) <= hoeffding_bentkus_pvalue(
		0.05, 50, 0.2
	)


def test_calibrate_risk_threshold_controls_risk():
	result = calibrate_risk_threshold(
		CALIB_SCORES, CALIB_LABELS, target_risk=0.2, delta=0.2
	)
	assert result.certified
	# Threshold must exclude the low-score (irrelevant) cluster.
	assert result.threshold > 0.3
	# Empirical selection-conditioned risk is within the target.
	assert result.risk_hat <= 0.2
	assert result.num_accepted > 0


def test_calibrate_risk_threshold_validates_inputs():
	with pytest.raises(ValueError):
		calibrate_risk_threshold([0.1, 0.2], [1], target_risk=0.1)
	with pytest.raises(ValueError):
		calibrate_risk_threshold([0.1], [1], target_risk=1.5)


def test_threshold_cutoff_uses_calibrated_threshold():
	"""The existing passage filter derives its cutoff from risk calibration."""
	instance = ThresholdCutoff(
		project_dir=project_dir, previous_result=previous_result, threshold=0.9
	)
	# Scores chosen so a calibrated cutoff (in the 0.3-0.7 gap) keeps exactly
	# the high-scoring passages and drops the low-scoring ones.
	scores = [[0.1, 0.8, 0.1, 0.5], [0.1, 0.2, 0.7, 0.3]]

	contents, ids, filtered_scores = instance._pure(
		contents_example,
		scores,
		ids_example,
		threshold=None,
		target_risk=0.2,
		calibration_scores=CALIB_SCORES,
		calibration_labels=CALIB_LABELS,
		risk_delta=0.2,
	)

	# Every kept score is above the calibrated cutoff; low scores are gone.
	assert filtered_scores[0] == [0.8]
	assert contents[0] == ["Paris is the capital of France."]
	assert filtered_scores[1] == [0.7]
	assert all(min(s) > 0.3 for s in filtered_scores)


def test_threshold_cutoff_requires_threshold_or_calibration():
	instance = ThresholdCutoff(
		project_dir=project_dir, previous_result=previous_result, threshold=0.9
	)
	with pytest.raises(ValueError):
		instance._pure(contents_example, [[0.1, 0.8]], ids_example)


def test_calibrate_joint_thresholds_cascade():
	# LLM-only branch is reliable when confident (high score); RAG rescues the
	# rest. Build a calibration set where the cascade can hit low risk.
	n = 40
	llm_scores = [0.9 if i % 2 == 0 else 0.2 for i in range(n)]
	rag_scores = [0.85] * n
	# LLM correct exactly when confident; RAG correct for the escalated ones.
	labels_llm = [1 if s >= 0.5 else 0 for s in llm_scores]
	labels_rag = [1] * n

	result = calibrate_joint_thresholds(
		llm_scores,
		rag_scores,
		labels_llm,
		labels_rag,
		target_risk=0.2,
		delta=0.2,
	)
	assert result is not None
	assert result.certified
	assert result.risk_hat <= 0.2
	assert 0.0 <= result.coverage <= 1.0
	assert 0.0 <= result.retrieval_rate <= 1.0
