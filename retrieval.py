"""
retrieval.py
------------
This is the "R" in RAG - Retrieval.

Plain-language explanation:
We take a list of job requirements (already extracted and cleaned by
generator.py's analyze_job_posting step - see that file for why) and a
resume, break the resume into individual bullet points, and for EACH
requirement ask: "which resume bullets are most similar in meaning to
this requirement?"

We use a technique called TF-IDF, which is simpler than it sounds:

  - TF (Term Frequency): how often a word appears in a piece of text
  - IDF (Inverse Document Frequency): how RARE that word is across all the text

Common words like "the" or "and" get a low score because they're everywhere.
Specific, meaningful words like "SQL" or "reconciliation" get a high score
because they don't show up everywhere - so when they DO match between a
requirement and a resume bullet, that's a real signal.

We turn every piece of text into a list of these word-importance scores
(a "vector"), and then measure the angle between two vectors using something
called "cosine similarity" - a number between 0 (completely unrelated) and
1 (identical). This is the same core idea real search engines and matching
tools use, just simplified.

VERSION HISTORY (kept here so the reasoning survives, not just the code):
v1: split the raw JD text on line breaks and treated every resulting line
    as a "requirement" - this included titles, requisition IDs, referral
    bonuses, etc. Too noisy.
v2: added a keyword-based filter (looks_like_requirement) to guess which
    lines were real requirements. Better, but still just guessing with
    regex - it let some marketing-copy lines through and could miss
    oddly-worded real requirements.
v3 (current): the JD is no longer split here at all. generator.py now asks
    Claude directly to read the whole posting and extract the real
    requirements, the company name, job title, salary and location - a
    judgment call Claude is much better suited for than pattern-matching.
    This file's job shrinks to what it's actually good at: measuring text
    similarity between a given list of requirements and resume bullets.
    The old keyword filter is kept below, unused by the main flow, purely
    as a safety net in generator.py in case the extraction call ever fails.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def split_into_lines(text):
    """
    Breaks a block of text into individual lines/bullets.
    We split on line breaks and bullet characters, and throw away
    anything too short to be meaningful (like a blank line or a header).
    Used for the resume (which is naturally a clean bullet list), and as
    a last-resort fallback for the JD if Claude's extraction ever fails.
    """
    raw_lines = re.split(r'[\n\r]+|(?:^|\n)[\-•\*]\s*', text)
    lines = [line.strip() for line in raw_lines if len(line.strip()) > 15]
    return lines


# --- Kept only as an emergency fallback (see generator.py) ---
REQUIREMENT_SIGNALS = [
    r"\byears?\b", r"\bexperience\b", r"\bdegree\b",
    r"\bbachelor", r"\bmaster", r"\bmba\b", r"\bphd\b",
    r"\bcertifi", r"\blicense", r"\bproficien", r"\bfamiliar",
    r"\bknowledge of\b", r"\bability to\b", r"\babilities\b", r"\bskills?\b",
    r"\brequired\b", r"\brequirement", r"\bmust have\b", r"\bmust be\b",
    r"\bpreferred\b", r"\bqualifi", r"\bexpertise\b", r"\bunderstanding of\b",
    r"\bsql\b|\bpython\b|\bexcel\b|\bjira\b|\bagile\b|\bscrum\b|\bpower ?bi\b",
]
BOILERPLATE_SIGNALS = [
    r"^title\s*:", r"^requisition\s*id", r"^job\s*id", r"^location\(s\)?\s*:",
    r"referral (program|reward|bonus)", r"^\$?\d[\d,]*\.\d{2}\b",
    r"equal opportunity", r"accommodat", r"inclusive and accessible",
    r"we are committed to", r"apply now", r"about (us|the bank|the team)\b",
]


def looks_like_requirement(line):
    """Emergency-fallback heuristic only - see module docstring."""
    lower = line.lower()
    for pattern in BOILERPLATE_SIGNALS:
        if re.search(pattern, lower):
            return False
    for pattern in REQUIREMENT_SIGNALS:
        if re.search(pattern, lower):
            return True
    return False


def fallback_extract_requirements(jd_text):
    """
    Only used if Claude's own extraction (analyze_job_posting in
    generator.py) fails for some reason - e.g. a network error. Keeps the
    app working in a degraded mode instead of crashing.
    """
    lines = split_into_lines(jd_text)
    filtered = [line for line in lines if looks_like_requirement(line)]
    return filtered if filtered else lines


def find_best_matches(requirements, resume_text, top_n=2):
    """
    For each requirement (a plain list of requirement strings, already
    extracted/cleaned upstream), find the top_n most similar resume
    bullets, along with a similarity score.

    Returns a list of dictionaries, one per requirement, like:
    {
        "requirement": "5+ years product management experience",
        "matches": [
            {"bullet": "Owned product roadmap for 11+ years...", "score": 0.42},
            {"bullet": "Led Agile delivery across three banks...", "score": 0.31}
        ]
    }
    """
    resume_lines = split_into_lines(resume_text)

    if not requirements or not resume_lines:
        return []

    all_lines = list(requirements) + resume_lines
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_lines)

    req_vectors = tfidf_matrix[:len(requirements)]
    resume_vectors = tfidf_matrix[len(requirements):]

    similarity_matrix = cosine_similarity(req_vectors, resume_vectors)

    results = []
    for i, requirement in enumerate(requirements):
        scores = similarity_matrix[i]
        top_indices = scores.argsort()[::-1][:top_n]
        matches = [
            {"bullet": resume_lines[idx], "score": round(float(scores[idx]), 3)}
            for idx in top_indices
        ]
        results.append({"requirement": requirement, "matches": matches})

    return results
