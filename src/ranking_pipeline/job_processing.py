import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Union

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOPWORDS
except Exception:
    SKLEARN_STOPWORDS = set()


CUSTOM_STOPWORDS = {
    "ability", "able", "across", "also", "applicant", "applicants", "apply",
    "candidate", "candidates", "company", "currently", "demonstrated", "desired",
    "detail", "details", "environment", "experience", "experiences", "familiarity",
    "good", "great", "high", "highly", "ideal", "including", "job", "knowledge",
    "looking", "must", "need", "opportunity", "plus", "position", "preferred",
    "qualification", "qualifications", "requirement", "requirements",
    "responsibilities", "responsibility", "role", "seeking", "skill", "skills",
    "strong", "team", "tools", "use", "using", "various", "well", "will", "work",
    "working", "year", "years", "within", "build", "develop", "design", "support",
    "maintain", "understanding", "excellent", "etc"
}

STOPWORDS = set(SKLEARN_STOPWORDS).union(CUSTOM_STOPWORDS)

ALLOWED_SHORT_TOKENS = {
    "ai", "ml", "nlp", "cv", "bi", "ui", "ux", "qa", "etl", "elt",
    "api", "sql", "aws", "gcp", "git", "oop", "llm"
}


NORMALIZATION_PATTERNS = {
    r"(?<!\w)c\+\+(?!\w)": "cplusplus",
    r"(?<!\w)c#(?!\w)": "csharp",
    r"(?<!\w)\.net core(?!\w)": "dotnetcore",
    r"(?<!\w)asp\.net core(?!\w)": "aspdotnetcore",
    r"(?<!\w)asp\.net(?!\w)": "aspdotnet",
    r"(?<!\w)node\.js(?!\w)": "nodejs",
    r"(?<!\w)react\.js(?!\w)": "reactjs",
    r"(?<!\w)vue\.js(?!\w)": "vuejs",
    r"(?<!\w)next\.js(?!\w)": "nextjs",
    r"(?<!\w)power bi(?!\w)": "power_bi",
    r"(?<!\w)tableau prep(?!\w)": "tableau_prep",
    r"(?<!\w)sql server(?!\w)": "sql_server",
    r"(?<!\w)machine learning(?!\w)": "machine_learning",
    r"(?<!\w)deep learning(?!\w)": "deep_learning",
    r"(?<!\w)data science(?!\w)": "data_science",
    r"(?<!\w)data analysis(?!\w)": "data_analysis",
    r"(?<!\w)artificial intelligence(?!\w)": "artificial_intelligence",
    r"(?<!\w)natural language processing(?!\w)": "natural_language_processing",
    r"(?<!\w)computer vision(?!\w)": "computer_vision",
    r"(?<!\w)amazon web services(?!\w)": "aws",
    r"(?<!\w)google cloud platform(?!\w)": "gcp",
    r"(?<!\w)scikit learn(?!\w)": "scikit_learn",
    r"(?<!\w)problem solving(?!\w)": "problem_solving",
    r"(?<!\w)time management(?!\w)": "time_management",
    r"(?<!\w)team work(?!\w)": "teamwork",
    r"(?<!\w)team player(?!\w)": "teamwork",
    r"(?<!\w)restful api(?:s)?(?!\w)": "rest_api",
    r"(?<!\w)rest api(?:s)?(?!\w)": "rest_api",
}


SECTION_PATTERNS = {
    "requirements": [
        r"\brequirements?\b",
        r"\bmust[- ]have\b",
        r"\bminimum qualifications?\b",
        r"\bwhat we(?: are|'re)? looking for\b",
        r"\brequired skills?\b"
    ],
    "preferred": [
        r"\bpreferred\b",
        r"\bpreferred qualifications?\b",
        r"\bnice to have\b",
        r"\bbonus\b",
        r"\bplus\b",
        r"\bdesired\b"
    ],
    "responsibilities": [
        r"\bresponsibilit(?:y|ies)\b",
        r"\bduties\b",
        r"\bwhat you(?: will|'ll)? do\b",
        r"\brole overview\b",
        r"\bkey tasks?\b"
    ],
    "qualifications": [
        r"\bqualifications?\b",
        r"\bskills\b",
        r"\bexperience\b"
    ],
    "benefits": [
        r"\bbenefits\b",
        r"\bwhat we offer\b",
        r"\bwhy join\b"
    ]
}


SKILL_PATTERNS = {
    "python": [r"\bpython\b"],
    "java": [r"\bjava\b"],
    "javascript": [r"\bjavascript\b", r"\breactjs\b", r"\bnodejs\b"],
    "typescript": [r"\btypescript\b"],
    "cplusplus": [r"\bcplusplus\b"],
    "csharp": [r"\bcsharp\b"],
    "aspdotnet": [r"\baspdotnet\b"],
    "aspdotnetcore": [r"\baspdotnetcore\b", r"\bdotnetcore\b"],
    "sql": [r"\bsql\b"],
    "sql_server": [r"\bsql_server\b"],
    "mysql": [r"\bmysql\b"],
    "postgresql": [r"\bpostgresql\b", r"\bpostgres\b"],
    "mongodb": [r"\bmongodb\b"],
    "redis": [r"\bredis\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "scikit_learn": [r"\bscikit_learn\b", r"\bsklearn\b"],
    "tensorflow": [r"\btensorflow\b"],
    "pytorch": [r"\bpytorch\b"],
    "xgboost": [r"\bxgboost\b"],
    "lightgbm": [r"\blightgbm\b"],
    "machine_learning": [r"\bmachine_learning\b", r"\bml\b"],
    "deep_learning": [r"\bdeep_learning\b"],
    "data_science": [r"\bdata_science\b"],
    "data_analysis": [r"\bdata_analysis\b"],
    "statistics": [r"\bstatistics\b", r"\bstatistical\b"],
    "artificial_intelligence": [r"\bartificial_intelligence\b", r"\bai\b"],
    "natural_language_processing": [
        r"\bnatural_language_processing\b",
        r"\bnlp\b"
    ],
    "computer_vision": [r"\bcomputer_vision\b", r"\bcv\b"],
    "excel": [r"\bexcel\b"],
    "power_bi": [r"\bpower_bi\b"],
    "tableau": [r"\btableau\b", r"\btableau_prep\b"],
    "spark": [r"\bspark\b", r"\bpyspark\b"],
    "hadoop": [r"\bhadoop\b"],
    "git": [r"\bgit\b"],
    "github": [r"\bgithub\b"],
    "docker": [r"\bdocker\b"],
    "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "linux": [r"\blinux\b"],
    "aws": [r"\baws\b"],
    "azure": [r"\bazure\b"],
    "gcp": [r"\bgcp\b"],
    "flask": [r"\bflask\b"],
    "django": [r"\bdjango\b"],
    "fastapi": [r"\bfastapi\b"],
    "react": [r"\breact\b", r"\breactjs\b"],
    "nodejs": [r"\bnodejs\b"],
    "microservices": [r"\bmicroservices?\b"],
    "rest_api": [r"\brest_api\b"],
    "oop": [r"\boop\b", r"\bobject oriented\b"],
    "problem_solving": [r"\bproblem_solving\b"],
    "communication": [r"\bcommunication\b", r"\bcommunicate\b"],
    "teamwork": [r"\bteamwork\b", r"\bcollaboration\b", r"\bcollaborative\b"],
    "leadership": [r"\bleadership\b", r"\blead\b"],
    "time_management": [r"\btime_management\b"]
}


EDUCATION_PATTERNS = {
    "bachelor": [
        r"\bbachelor(?:'s|s)?\b",
        r"\bbsc\b",
        r"\bb\.sc\b"
    ],
    "master": [
        r"\bmaster(?:'s|s)?\b",
        r"\bmsc\b",
        r"\bm\.sc\b"
    ],
    "phd": [
        r"\bphd\b",
        r"\bdoctorate\b",
        r"\bdoctoral\b"
    ]
}


def load_job_description(source: Union[str, Path]) -> str:
    """
    Load job description from a file path or use the raw text directly.
    """
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8").strip()

    source_str = str(source).strip()

    if "\n" not in source_str:
        try:
            possible_path = Path(source_str)
            if possible_path.exists() and possible_path.is_file():
                return possible_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    return source_str


def normalize_text(text: str) -> str:
    """
    Normalize raw job text:
    - lowercase
    - preserve technical phrases as single tokens
    - remove noisy punctuation
    - collapse whitespace
    """
    text = text.lower()

    # Normalize common punctuation variants
    text = (
        text.replace("’", "'")
            .replace("–", "-")
            .replace("—", "-")
            .replace("/", " ")
            .replace("\\", " ")
    )

    for pattern, replacement in NORMALIZATION_PATTERNS.items():
        text = re.sub(pattern, replacement, text)

    # Keep letters, numbers, underscores, spaces
    text = re.sub(r"[^a-z0-9_\s\-]", " ", text)

    # Convert hyphenated normal words to spaces (keep normalized terms with underscores)
    text = re.sub(r"-", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize normalized text and remove noise.
    """
    tokens = re.findall(r"[a-z0-9_]+", text)

    filtered_tokens = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        if len(token) <= 2 and token not in ALLOWED_SHORT_TOKENS:
            continue
        filtered_tokens.append(token)

    return filtered_tokens


def _detect_section_heading(line: str) -> Optional[str]:
    """
    Detect whether a line is likely a section heading.
    """
    cleaned = line.strip().lower()
    cleaned = re.sub(r"^[\-\*\u2022\d\.\)\s]+", "", cleaned)
    cleaned = re.sub(r"[:\-–—]+$", "", cleaned)
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return None

    # Section headings are usually short
    if len(cleaned.split()) > 5:
        return None

    for section_name, patterns in SECTION_PATTERNS.items():
        if any(re.search(pattern, cleaned) for pattern in patterns):
            return section_name

    return None


def extract_sections(raw_text: str) -> Dict[str, str]:
    """
    Split the job description into logical sections if headings exist.
    """
    sections = {
        "summary": "",
        "requirements": "",
        "preferred": "",
        "responsibilities": "",
        "qualifications": "",
        "benefits": ""
    }

    current_section = "summary"

    for line in raw_text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        detected = _detect_section_heading(stripped)
        if detected:
            current_section = detected
            continue

        stripped = re.sub(r"^[\-\*\u2022]+", "", stripped).strip()
        sections[current_section] += " " + stripped

    return {key: value.strip() for key, value in sections.items() if value.strip()}


def extract_skills(clean_text: str, patterns: Dict[str, List[str]] = None) -> List[str]:
    """
    Extract canonical skill names from normalized text.
    """
    patterns = patterns or SKILL_PATTERNS
    found_skills = []

    for skill, regex_list in patterns.items():
        if any(re.search(regex, clean_text) for regex in regex_list):
            found_skills.append(skill)

    return sorted(found_skills)


def extract_education(clean_text: str) -> List[str]:
    """
    Extract education requirements from normalized text.
    """
    found = []

    for degree, regex_list in EDUCATION_PATTERNS.items():
        if any(re.search(regex, clean_text) for regex in regex_list):
            found.append(degree)

    return found


def extract_experience_requirements(raw_text: str) -> Dict[str, Optional[Union[int, List[Dict[str, Optional[int]]]]]]:
    """
    Extract years of experience mentioned in the job description.

    Returns:
        {
            "min_years_required": int | None,
            "max_years_mentioned": int | None,
            "mentions": [{"min": int, "max": int | None}, ...]
        }
    """
    text = raw_text.lower()

    patterns = [
        r"(?<!\w)(\d+)\s*(?:\+|plus)?\s*(?:-|to)?\s*(\d+)?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+)?experience\b",
        r"\bat least\s+(\d+)\s*(?:years?|yrs?)\b",
        r"\bminimum\s+of\s+(\d+)\s*(?:years?|yrs?)\b"
    ]

    mentions = []

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            groups = match.groups()

            if len(groups) == 2:
                min_year = int(groups[0]) if groups[0] else None
                max_year = int(groups[1]) if groups[1] else None
            else:
                min_year = int(groups[0]) if groups[0] else None
                max_year = None

            if min_year is not None:
                mentions.append({"min": min_year, "max": max_year})

    if not mentions:
        return {
            "min_years_required": None,
            "max_years_mentioned": None,
            "mentions": []
        }

    min_years_required = max(item["min"] for item in mentions if item["min"] is not None)
    max_years_mentioned = max(
        item["max"] if item["max"] is not None else item["min"]
        for item in mentions
    )

    return {
        "min_years_required": min_years_required,
        "max_years_mentioned": max_years_mentioned,
        "mentions": mentions
    }


def extract_keywords(clean_text: str, priority_terms: List[str], top_n: int = 20) -> List[str]:
    """
    Extract keywords for downstream ranking/similarity.
    Priority terms (skills) are placed first, then top frequent tokens.
    """
    tokens = tokenize_text(clean_text)
    counts = Counter(tokens)

    keywords = []
    seen = set()

    for term in priority_terms:
        if term not in seen:
            keywords.append(term)
            seen.add(term)

    for token, _ in counts.most_common(top_n * 2):
        if token not in seen:
            keywords.append(token)
            seen.add(token)
        if len(keywords) >= top_n:
            break

    return keywords[:top_n]


def build_search_text(
    clean_text: str,
    required_skills: List[str],
    preferred_skills: List[str],
    keywords: List[str],
    min_years_required: Optional[int]
) -> str:
    """
    Build weighted text for similarity scoring.
    Required skills are repeated more strongly than preferred skills.
    """
    boosted_terms = []

    # Strong boost for required skills
    for skill in required_skills:
        boosted_terms.extend([skill] * 4)

    # Medium boost for preferred skills
    for skill in preferred_skills:
        boosted_terms.extend([skill] * 2)

    # Mild boost for extra keywords
    for keyword in keywords[:10]:
        boosted_terms.append(keyword)

    if min_years_required is not None:
        boosted_terms.extend([f"{min_years_required}_years_experience"] * 2)

    return f"{clean_text} {' '.join(boosted_terms)}".strip()


def process_job_description(source: Union[str, Path], top_n_keywords: int = 20) -> Dict[str, object]:
    """
    Full job processing pipeline.

    Output is designed to be directly useful for ranking and similarity scoring.
    """
    raw_text = load_job_description(source)
    sections = extract_sections(raw_text)

    section_summary = sections.get("summary", "")
    section_requirements = sections.get("requirements", "")
    section_qualifications = sections.get("qualifications", "")
    section_preferred = sections.get("preferred", "")
    section_responsibilities = sections.get("responsibilities", "")

    full_text_for_processing = " ".join([
        section_summary,
        section_requirements,
        section_qualifications,
        section_preferred,
        section_responsibilities
    ]).strip() or raw_text

    requirements_text = " ".join([section_requirements, section_qualifications]).strip()
    preferred_text = section_preferred.strip()

    clean_full_text = normalize_text(full_text_for_processing)
    clean_requirements_text = normalize_text(requirements_text) if requirements_text else clean_full_text
    clean_preferred_text = normalize_text(preferred_text) if preferred_text else ""

    all_skills = extract_skills(clean_full_text)
    required_skills = extract_skills(clean_requirements_text) if clean_requirements_text else all_skills
    preferred_skills = extract_skills(clean_preferred_text) if clean_preferred_text else []

    # If no explicit requirements section exists, use all detected skills as required baseline
    if not required_skills:
        required_skills = all_skills.copy()

    experience = extract_experience_requirements(raw_text)
    education = extract_education(clean_full_text)

    priority_terms = required_skills + preferred_skills
    keywords = extract_keywords(
        clean_text=clean_requirements_text if clean_requirements_text else clean_full_text,
        priority_terms=priority_terms,
        top_n=top_n_keywords
    )

    search_text = build_search_text(
        clean_text=clean_full_text,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        keywords=keywords,
        min_years_required=experience["min_years_required"]
    )

    return {
        "raw_text": raw_text,
        "sections": sections,
        "clean_text": clean_full_text,
        "tokens": tokenize_text(clean_full_text),
        "all_skills": all_skills,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "experience": experience,
        "education": education,
        "keywords": keywords,
        "search_text": search_text
    }