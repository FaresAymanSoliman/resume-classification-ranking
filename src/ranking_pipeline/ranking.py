from typing import Any, Dict, List, Optional


DEFAULT_RANKING_WEIGHTS = {
    "hybrid_similarity_score": 0.75,
    "classification_score": 0.20,
    "label_match_score": 0.05,
}


def _safe_score(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Convert a score-like value into a float in [0, 1].
    Returns the default value if conversion fails.
    """
    if value is None:
        return default

    try:
        score = float(value)
    except (TypeError, ValueError):
        return default

    return max(0.0, min(1.0, score))


def _get_classification_score(candidate: Dict[str, Any]) -> Optional[float]:
    """
    Retrieve classifier confidence from the candidate record if present.
    """
    possible_keys = (
        "classification_score",
        "classifier_confidence",
        "prediction_probability",
        "class_probability",
        "probability",
        "confidence",
    )

    for key in possible_keys:
        if key in candidate:
            return _safe_score(candidate.get(key))

    return None


def _get_predicted_label(candidate: Dict[str, Any]) -> Optional[str]:
    """
    Retrieve the predicted label/category for a candidate if present.
    """
    possible_keys = (
        "predicted_label",
        "classification_label",
        "predicted_category",
        "category",
        "label",
    )

    for key in possible_keys:
        value = candidate.get(key)
        if value:
            return str(value).strip().lower()

    return None


def _get_target_labels(job_data: Optional[Dict[str, Any]]) -> List[str]:
    """
    Retrieve the expected job labels/categories from the job data if present.
    """
    if not job_data:
        return []

    possible_keys = (
        "target_label",
        "expected_label",
        "job_label",
        "job_category",
        "category",
        "target_labels",
    )

    labels = []

    for key in possible_keys:
        value = job_data.get(key)

        if isinstance(value, str) and value.strip():
            labels.append(value.strip().lower())

        elif isinstance(value, (list, tuple, set)):
            labels.extend(
                str(item).strip().lower()
                for item in value
                if str(item).strip()
            )

    # Remove duplicates while preserving order
    unique_labels = []
    seen = set()

    for label in labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)

    return unique_labels


def compute_label_match_score(
    candidate: Dict[str, Any],
    job_data: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Return:
    - 1.0 if candidate predicted label matches expected job label
    - 0.0 if labels exist but do not match
    - None if job label information is unavailable
    """
    target_labels = _get_target_labels(job_data)
    if not target_labels:
        return None

    predicted_label = _get_predicted_label(candidate)
    if not predicted_label:
        return 0.0

    return 1.0 if predicted_label in target_labels else 0.0


def compute_penalty_multiplier(candidate: Dict[str, Any]) -> float:
    """
    Apply penalties when required skill coverage or experience alignment is weak.
    """
    required_skill_coverage = _safe_score(
        candidate.get("required_skill_coverage"),
        default=1.0
    )
    experience_score = _safe_score(
        candidate.get("experience_score"),
        default=1.0
    )

    penalty = 1.0

    # Penalize low required skill match
    if required_skill_coverage is not None:
        if required_skill_coverage < 0.25:
            penalty *= 0.75
        elif required_skill_coverage < 0.50:
            penalty *= 0.88
        elif required_skill_coverage < 0.70:
            penalty *= 0.95

    # Small penalty for weak experience alignment
    if experience_score is not None and experience_score < 0.35:
        penalty *= 0.90

    return round(penalty, 4)


def compute_final_score(
    candidate: Dict[str, Any],
    job_data: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Compute the final ranking score using:
    - hybrid similarity score
    - classification confidence (if available)
    - label match score (if available)
    - penalty multiplier for weak requirements/experience
    """
    merged_weights = DEFAULT_RANKING_WEIGHTS.copy()
    if weights:
        merged_weights.update(weights)

    metrics = {
        "hybrid_similarity_score": _safe_score(
            candidate.get("hybrid_similarity_score"),
            default=0.0
        ),
        "classification_score": _get_classification_score(candidate),
        "label_match_score": compute_label_match_score(candidate, job_data),
    }

    available_metrics = {
        name: value
        for name, value in metrics.items()
        if value is not None
    }

    if not available_metrics:
        weighted_score = 0.0
    else:
        total_weight = sum(
            merged_weights[name]
            for name in available_metrics
            if name in merged_weights
        )

        if total_weight == 0:
            weighted_score = 0.0
        else:
            weighted_score = sum(
                available_metrics[name] * merged_weights[name]
                for name in available_metrics
                if name in merged_weights
            ) / total_weight

    penalty_multiplier = compute_penalty_multiplier(candidate)
    final_score = max(0.0, min(1.0, weighted_score * penalty_multiplier))

    return {
        "classification_score": (
            round(metrics["classification_score"], 4)
            if metrics["classification_score"] is not None
            else None
        ),
        "label_match_score": (
            round(metrics["label_match_score"], 4)
            if metrics["label_match_score"] is not None
            else None
        ),
        "penalty_multiplier": penalty_multiplier,
        "final_score": round(final_score, 4),
    }


def assign_match_band(final_score: float) -> str:
    """
    Convert a numeric score into a human-readable match level.
    """
    if final_score >= 0.80:
        return "excellent_match"
    if final_score >= 0.65:
        return "strong_match"
    if final_score >= 0.50:
        return "good_match"
    if final_score >= 0.35:
        return "moderate_match"
    return "low_match"


def build_match_explanation(
    candidate: Dict[str, Any],
    final_score: float
) -> List[str]:
    """
    Build simple explainable reasons for the ranking.
    """
    reasons = []

    matched_required = candidate.get("matched_required_skills", []) or []
    matched_preferred = candidate.get("matched_preferred_skills", []) or []

    required_coverage = _safe_score(candidate.get("required_skill_coverage"), default=0.0)
    text_similarity = _safe_score(candidate.get("text_similarity"), default=0.0)
    experience_score = _safe_score(candidate.get("experience_score"), default=0.0)

    if matched_required:
        reasons.append(f"matched {len(matched_required)} required skills")

    if matched_preferred:
        reasons.append(f"matched {len(matched_preferred)} preferred skills")

    if required_coverage is not None and required_coverage >= 0.70:
        reasons.append("strong required skill coverage")
    elif required_coverage is not None and required_coverage < 0.40:
        reasons.append("weak required skill coverage")

    if text_similarity is not None and text_similarity >= 0.50:
        reasons.append("high textual similarity with the job description")

    if experience_score is not None and experience_score >= 0.80:
        reasons.append("experience aligns well with the job requirement")
    elif experience_score is not None and experience_score < 0.35:
        reasons.append("experience appears below the job requirement")

    if final_score >= 0.80:
        reasons.append("overall profile is highly relevant")
    elif final_score < 0.35:
        reasons.append("overall profile is weak for this job")

    return reasons


def rank_candidates(
    similarity_results: List[Dict[str, Any]],
    job_data: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Rank candidates using similarity results plus optional classifier metadata.

    Parameters
    ----------
    similarity_results : list of dict
        Output from compute_similarity_scores() in similarity.py
    job_data : dict, optional
        Processed job data from job_processing.py
    weights : dict, optional
        Custom scoring weights
    top_k : int, optional
        Return only top_k candidates if provided

    Returns
    -------
    list of dict
        Ranked candidate records with:
        - final_score
        - rank
        - match_band
        - explanations
    """
    ranked_candidates = []
    required_skills = set((job_data or {}).get("required_skills", []) or [])

    for candidate in similarity_results:
        candidate_copy = dict(candidate)

        score_data = compute_final_score(
            candidate=candidate_copy,
            job_data=job_data,
            weights=weights
        )

        matched_required = set(candidate_copy.get("matched_required_skills", []) or [])
        missing_required = sorted(required_skills - matched_required) if required_skills else []

        candidate_copy.update(score_data)
        candidate_copy["missing_required_skills"] = missing_required
        candidate_copy["match_band"] = assign_match_band(candidate_copy["final_score"])
        candidate_copy["ranking_explanations"] = build_match_explanation(
            candidate_copy,
            candidate_copy["final_score"]
        )

        ranked_candidates.append(candidate_copy)

    ranked_candidates.sort(
        key=lambda item: (
            item.get("final_score", 0.0),
            item.get("required_skill_coverage", 0.0),
            item.get("text_similarity", 0.0),
            item.get("classification_score", 0.0) or 0.0,
        ),
        reverse=True
    )

    for rank_index, candidate in enumerate(ranked_candidates, start=1):
        candidate["rank"] = rank_index

    if top_k is not None:
        return ranked_candidates[:top_k]

    return ranked_candidates