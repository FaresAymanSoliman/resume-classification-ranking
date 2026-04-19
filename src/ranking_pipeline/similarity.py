import re
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .job_processing import normalize_text, extract_skills
except ImportError:
    from src.ranking_pipeline.job_processing import normalize_text, extract_skills


DEFAULT_SIMILARITY_WEIGHTS = {
    "text_similarity": 0.50,
    "required_skill_coverage": 0.30,
    "preferred_skill_coverage": 0.10,
    "experience_score": 0.10,
}


def _get_candidate_id(resume: Dict[str, Any], index: int) -> str:
    """
    Return the best available identifier for a resume.
    """
    for key in ("candidate_id", "resume_id", "id", "email", "name"):
        value = resume.get(key)
        if value:
            return str(value)
    return f"candidate_{index + 1}"


def _get_resume_text(resume: Dict[str, Any]) -> str:
    """
    Get resume text from the most likely field.
    """
    for key in ("clean_resume_text", "processed_text", "resume_text", "text", "raw_text"):
        value = resume.get(key)
        if value:
            return str(value)
    return ""


def _canonicalize_skill(skill: str) -> str:
    """
    Normalize a skill string into the same canonical form used in job processing.
    """
    normalized = normalize_text(str(skill))
    return normalized.replace(" ", "_").strip("_")


def _extract_resume_skills(resume: Dict[str, Any], normalized_text: str) -> List:
    """
    Extract skills from the resume text and merge them with any provided skill list.
    """
    detected_skills = set(extract_skills(normalized_text))

    provided_skills = resume.get("skills") or resume.get("extracted_skills") or []
    if isinstance(provided_skills, str):
        provided_skills = re.split(r"[,;/|]", provided_skills)

    for skill in provided_skills:
        canonical_skill = _canonicalize_skill(skill)
        if canonical_skill:
            detected_skills.add(canonical_skill)

    return sorted(detected_skills)


def _extract_resume_experience_years(
    resume: Dict[str, Any],
    raw_resume_text: str
) -> Optional:
    """
    Try to get total experience years from structured fields first.
    If unavailable, fall back to a simple regex over the resume text.
    """
    for key in ("years_experience", "experience_years", "total_experience_years"):
        value = resume.get(key)

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            match = re.search(r"(\d+(?:\.\d+)?)", value)
            if match:
                return float(match.group(1))

    text = raw_resume_text.lower()
    matches = []

    for match in re.finditer(r"(?<!\w)(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:years?|yrs?)\b", text):
        years = float(match.group(1))
        if 0 < years <= 40:
            matches.append(years)

    return max(matches) if matches else None


def build_resume_search_text(normalized_resume_text: str, resume_skills: List[str]) -> str:
    """
    Build a search representation for the resume.
    Repeat detected skills to make them more visible in TF-IDF matching.
    """
    skill_boost = " ".join(resume_skills)
    parts = [normalized_resume_text]

    if skill_boost:
        parts.append(skill_boost)
        parts.append(skill_boost)

    return " ".join(part for part in parts if part).strip()


def compute_text_similarity_scores(
    job_search_text: str,
    resume_search_texts: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 15000
) -> List:
    """
    Compute TF-IDF cosine similarity between the job and each resume.
    """
    if not resume_search_texts:
        return []

    documents = [(job_search_text or "").strip()] + [(text or "").strip() for text in resume_search_texts]

    if not any(documents):
        return [0.0] * len(resume_search_texts)

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=False,
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(documents)
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

        return [float(score) for score in similarities]

    except ValueError:
        return [0.0] * len(resume_search_texts)


def compute_skill_alignment(job_data: Dict[str, Any], resume_skills: List[str]) -> Dict[str, Any]:
    """
    Measure overlap between job skills and resume skills.
    """
    required_skills = set(job_data.get("required_skills") or [])
    preferred_skills = set(job_data.get("preferred_skills") or [])
    resume_skill_set = set(resume_skills)

    matched_required = sorted(required_skills.intersection(resume_skill_set))
    matched_preferred = sorted(preferred_skills.intersection(resume_skill_set))

    required_skill_coverage = (
        len(matched_required) / len(required_skills)
        if required_skills else 1.0
    )

    preferred_skill_coverage = (
        len(matched_preferred) / len(preferred_skills)
        if preferred_skills else 0.0
    )

    return {
        "matched_required_skills": matched_required,
        "matched_preferred_skills": matched_preferred,
        "required_skill_coverage": required_skill_coverage,
        "preferred_skill_coverage": preferred_skill_coverage,
    }


def compute_experience_score(
    job_data: Dict[str, Any],
    resume_years: Optional[float]
) -> Dict[str, Any]:
    """
    Compare resume experience against the minimum years required by the job.
    """
    experience_data = job_data.get("experience") or {}
    min_years_required = experience_data.get("min_years_required")

    if min_years_required is None:
        return {
            "resume_years": resume_years,
            "min_years_required": None,
            "experience_score": 1.0,
        }

    min_years_required = float(min_years_required)

    if resume_years is None:
        return {
            "resume_years": None,
            "min_years_required": min_years_required,
            "experience_score": 0.35,
        }

    score = min(1.0, float(resume_years) / max(min_years_required, 1.0))

    return {
        "resume_years": float(resume_years),
        "min_years_required": min_years_required,
        "experience_score": score,
    }


def prepare_resume_for_similarity(resume: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    Normalize a resume into a structure ready for similarity computation.
    """
    candidate_id = _get_candidate_id(resume, index)
    raw_resume_text = _get_resume_text(resume)
    normalized_resume_text = normalize_text(raw_resume_text)
    resume_skills = _extract_resume_skills(resume, normalized_resume_text)
    resume_years = _extract_resume_experience_years(resume, raw_resume_text)
    resume_search_text = build_resume_search_text(normalized_resume_text, resume_skills)

    return {
        "candidate_id": candidate_id,
        "original_resume": resume,
        "raw_resume_text": raw_resume_text,
        "normalized_resume_text": normalized_resume_text,
        "resume_skills": resume_skills,
        "resume_years": resume_years,
        "resume_search_text": resume_search_text,
    }


def compute_similarity_scores(
    job_data: Dict[str, Any],
    resumes: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Compute hybrid similarity metrics for a list of resumes against one job.

    Returns a list of enriched resume records with:
    - text_similarity
    - required_skill_coverage
    - preferred_skill_coverage
    - experience_score
    - hybrid_similarity_score

    This function does NOT rank/sort candidates.
    Ranking belongs in ranking.py.
    """
    merged_weights = DEFAULT_SIMILARITY_WEIGHTS.copy()
    if weights:
        merged_weights.update(weights)

    prepared_resumes = [
        prepare_resume_for_similarity(resume, index=i)
        for i, resume in enumerate(resumes)
    ]

    job_search_text = job_data.get("search_text") or job_data.get("clean_text") or ""

    resume_search_texts = [
        item["resume_search_text"]
        for item in prepared_resumes
    ]

    text_scores = compute_text_similarity_scores(job_search_text, resume_search_texts)

    results = []

    for prepared_resume, text_similarity in zip(prepared_resumes, text_scores):
        skill_data = compute_skill_alignment(job_data, prepared_resume["resume_skills"])
        experience_data = compute_experience_score(job_data, prepared_resume["resume_years"])

        hybrid_similarity_score = (
            merged_weights["text_similarity"] * text_similarity
            + merged_weights["required_skill_coverage"] * skill_data["required_skill_coverage"]
            + merged_weights["preferred_skill_coverage"] * skill_data["preferred_skill_coverage"]
            + merged_weights["experience_score"] * experience_data["experience_score"]
        )

        result = dict(prepared_resume["original_resume"])
        result.update({
            "candidate_id": prepared_resume["candidate_id"],
            "normalized_resume_text": prepared_resume["normalized_resume_text"],
            "resume_skills": prepared_resume["resume_skills"],
            "resume_years": prepared_resume["resume_years"],
            "text_similarity": round(text_similarity, 4),
            "matched_required_skills": skill_data["matched_required_skills"],
            "matched_preferred_skills": skill_data["matched_preferred_skills"],
            "required_skill_coverage": round(skill_data["required_skill_coverage"], 4),
            "preferred_skill_coverage": round(skill_data["preferred_skill_coverage"], 4),
            "experience_score": round(experience_data["experience_score"], 4),
            "hybrid_similarity_score": round(hybrid_similarity_score, 4),
        })

        results.append(result)

    return results