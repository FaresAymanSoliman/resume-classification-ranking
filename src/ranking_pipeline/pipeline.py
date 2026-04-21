from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from .job_processing import process_job_description
    from .similarity import compute_similarity_scores
    from .ranking import rank_candidates
except ImportError:
    from src.ranking_pipeline.job_processing import process_job_description
    from src.ranking_pipeline.similarity import compute_similarity_scores
    from src.ranking_pipeline.ranking import rank_candidates


def _validate_resumes(resumes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only valid resume records.
    A valid resume must be a dictionary and should contain at least one text-like field.
    """
    if not isinstance(resumes, list):
        raise TypeError("resumes must be a list of dictionaries")

    valid_resumes = []

    possible_text_keys = {
        "resume_text",
        "clean_resume_text",
        "processed_text",
        "text",
        "raw_text",
    }

    for index, resume in enumerate(resumes):
        if not isinstance(resume, dict):
            continue

        has_text = any(
            key in resume and str(resume.get(key)).strip()
            for key in possible_text_keys
        )

        if has_text:
            valid_resumes.append(resume)

    return valid_resumes


def run_ranking_pipeline(
    job_source: Union[str, Path],
    resumes: List[Dict[str, Any]],
    job_metadata: Optional[Dict[str, Any]] = None,
    similarity_weights: Optional[Dict[str, float]] = None,
    ranking_weights: Optional[Dict[str, float]] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    End-to-end ranking pipeline.

    Parameters
    ----------
    job_source : str or Path
        Job description text or file path.
    resumes : list of dict
        Resume records.
    job_metadata : dict, optional
        Extra job fields to merge into processed job data
        (e.g. target_label='data_science').
    similarity_weights : dict, optional
        Custom weights for similarity scoring.
    ranking_weights : dict, optional
        Custom weights for final ranking.
    top_k : int, optional
        Return only the top_k ranked candidates.

    Returns
    -------
    dict
        {
            "job_data": ...,
            "total_candidates": ...,
            "returned_candidates": ...,
            "ranked_candidates": [...]
        }
    """
    valid_resumes = _validate_resumes(resumes)

    if not valid_resumes:
        return {
            "job_data": {},
            "total_candidates": 0,
            "returned_candidates": 0,
            "ranked_candidates": [],
        }

    job_data = process_job_description(job_source)

    if job_metadata:
        job_data.update(job_metadata)

    similarity_results = compute_similarity_scores(
        job_data=job_data,
        resumes=valid_resumes,
        weights=similarity_weights,
    )

    ranked_candidates = rank_candidates(
        similarity_results=similarity_results,
        job_data=job_data,
        weights=ranking_weights,
        top_k=top_k,
    )

    return {
        "job_data": job_data,
        "total_candidates": len(valid_resumes),
        "returned_candidates": len(ranked_candidates),
        "ranked_candidates": ranked_candidates,
    }


def get_top_candidate(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Return the top-ranked candidate from the pipeline result.
    """
    ranked_candidates = result.get("ranked_candidates", [])
    if not ranked_candidates:
        return None
    return ranked_candidates[0]