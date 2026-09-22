"""
generator.py
------------
This is the "G" in RAG - Generation.

Plain-language explanation:
Retrieval already did the hard work of finding WHICH resume bullets are
probably related to WHICH job requirements. But similarity scores alone
aren't smart enough to give an honest verdict - a resume bullet can share
a lot of the same words as a requirement without actually satisfying it.
This is exactly the kind of judgment call that a human (or an AI that can
actually reason) needs to make.

This file now does THREE jobs, in this order:

1. analyze_job_posting() - reads the raw pasted text ONCE and asks Claude:
   "Is this actually a job posting? If so, who's the employer, what's the
   title, what's the salary/location, and what are the REAL requirements
   (ignoring benefits blurbs, referral bonuses, mission statements, etc.)?"
   This replaces the old regex-guessing approach in retrieval.py with a
   judgment call Claude is much better suited to make, and it's also what
   lets the app reject junk input (a resume pasted by mistake, random
   text, an email) instead of pretending to analyze it.

2. get_all_verdicts() - takes the retrieval matches (produced by
   retrieval.py from the requirements extracted in step 1) and asks Claude
   for an honest Yes/Partial/No verdict on EVERY requirement IN ONE SINGLE
   API CALL, instead of one call per requirement. This is the main fix for
   the slowness you saw: a 40-line JD used to mean 40 separate round trips
   to Claude (each with its own network delay); now it's one call
   regardless of how many requirements there are. If that batched call
   ever fails to come back as valid data (rare, but networks are
   networks), the code automatically falls back to the slower one-by-one
   method rather than crashing - so the app stays fast in the normal case
   and merely slows down, instead of breaking, in the rare bad case.

3. calculate_match_score() - turns the Yes/Partial/No verdicts into a
   single percentage. We calculate this ourselves with plain arithmetic
   (Yes = 100%, Partial = 50%, No = 0%, averaged) rather than asking Claude
   to invent a percentage - that keeps the number explainable: you can
   always show exactly how it was calculated, which matters if this is
   ever a paid product.

This is real API usage - actual code talking to a real AI model over the
internet, the same way any AI-powered product you've ever used works
under the hood.
"""

import anthropic
import json

DEFAULT_MODEL = "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# STEP 1: Understand and validate the job posting, extract requirements
# ---------------------------------------------------------------------------

def build_analysis_prompt(jd_text):
    return f"""You are analyzing a piece of pasted text to determine if it is a genuine
job posting / job description, and if so, to extract structured information
from it.

Text to analyze:
\"\"\"
{jd_text}
\"\"\"

Do the following:
1. Decide whether this text is actually a job posting/job description
   (not a resume, not an email, not an article, not random text, not a
   question to an AI).
2. If it IS a job posting, extract:
   - company: the employer's name, or null if not stated
   - job_title: the role's title, or null if not stated
   - salary: the salary/compensation range exactly as stated, or null if
     no compensation is mentioned anywhere
   - location: the work location, or null if not stated
   - requirements: a list of ONLY the genuine requirements/qualifications
     a candidate needs (skills, years of experience, degrees,
     certifications, specific responsibilities that describe what the
     person must be able to do). Do NOT include: benefits, referral
     bonuses, equal-opportunity/accommodation statements, requisition or
     job ID numbers, "about us"/company marketing copy, or application
     instructions. Each requirement should be a short, self-contained
     sentence or phrase.

Respond ONLY in this exact JSON format, nothing else, no markdown formatting,
no code fences:
{{
  "is_job_description": true or false,
  "reason": "one short sentence explaining your decision - if false, say what the text actually looks like instead",
  "company": "..." or null,
  "job_title": "..." or null,
  "salary": "..." or null,
  "location": "..." or null,
  "requirements": ["...", "..."]
}}
"""


def analyze_job_posting(client, jd_text, model=DEFAULT_MODEL):
    """
    Single API call. Returns a dict with is_job_description, reason,
    company, job_title, salary, location, requirements.
    """
    prompt = build_analysis_prompt(jd_text)
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.content[0].text.strip()
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fail safe: treat an unparseable response as "couldn't confirm this
        # is a job posting" rather than crashing the app.
        data = {
            "is_job_description": False,
            "reason": "Could not analyze this text - please try pasting the job posting again.",
            "company": None,
            "job_title": None,
            "salary": None,
            "location": None,
            "requirements": [],
        }
    return data


# ---------------------------------------------------------------------------
# STEP 2: Honest verdicts - batched (fast) with a sequential fallback (safe)
# ---------------------------------------------------------------------------

def build_batch_verdict_prompt(retrieval_results):
    items_text = ""
    for i, item in enumerate(retrieval_results):
        bullets_text = "\n".join(
            f'   - "{m["bullet"]}" (text similarity score: {m["score"]})'
            for m in item["matches"]
        )
        items_text += f'\n{i + 1}. Requirement: "{item["requirement"]}"\n   Closest resume bullets found:\n{bullets_text}\n'

    prompt = f"""You are an honest, no-fabrication career counselor. Below are
{len(retrieval_results)} job requirements. For each one, the resume bullets a
text-matching system found most similar to it are also shown (a similarity
score in brackets, 0 = unrelated, 1 = identical - but the score is only a
rough text-overlap signal, not a judgment of real fit; that's YOUR job).

{items_text}

Rules for every requirement:
- Do not be generous. Do not assume things the resume doesn't actually say.
- A shared keyword does NOT mean the requirement is met.
- If the resume bullets only partially cover the requirement, say so plainly and explain what's missing.
- If the resume bullets don't genuinely address the requirement at all, say "No" plainly.

Respond ONLY with a JSON array, with EXACTLY {len(retrieval_results)} objects,
in the SAME ORDER as the numbered list above, nothing else, no markdown
formatting, no code fences:
[
  {{"verdict": "Yes" or "Partial" or "No", "reasoning": "one or two honest, specific sentences"}},
  ...
]
"""
    return prompt


def get_all_verdicts_batch(client, retrieval_results, model=DEFAULT_MODEL):
    """One API call for every requirement. Raises on any problem, so the
    caller (get_all_verdicts) can fall back to the safer sequential path."""
    prompt = build_batch_verdict_prompt(retrieval_results)
    max_tokens = min(8000, 250 * len(retrieval_results) + 300)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.content[0].text.strip()
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    verdicts = json.loads(cleaned)  # let this raise if malformed

    if not isinstance(verdicts, list) or len(verdicts) != len(retrieval_results):
        raise ValueError("Batch verdict response did not match expected shape.")

    full_results = []
    for item, verdict in zip(retrieval_results, verdicts):
        full_results.append({
            "requirement": item["requirement"],
            "matches": item["matches"],
            "verdict": verdict.get("verdict", "Error"),
            "reasoning": verdict.get("reasoning", ""),
        })
    return full_results


def build_verdict_prompt(requirement, matched_bullets):
    """Single-requirement prompt - only used by the sequential fallback."""
    bullets_text = "\n".join(
        f'- "{m["bullet"]}" (text similarity score: {m["score"]})' for m in matched_bullets
    )
    return f"""You are an honest, no-fabrication career counselor. You are given ONE job
requirement and the resume bullets that a text-matching system found to be
most similar to it. Judge, HONESTLY, whether the candidate's real experience
actually satisfies this requirement.

Job requirement:
"{requirement}"

Resume bullets found most similar (similarity score in brackets):
{bullets_text}

Rules:
- Do not be generous. Do not assume things the resume doesn't actually say.
- A shared keyword does NOT mean the requirement is met.
- If only partially covered, say so and explain what's missing.
- If not genuinely addressed at all, say "No" plainly.

Respond ONLY in this exact JSON format, nothing else, no markdown formatting:
{{"verdict": "Yes" or "Partial" or "No", "reasoning": "one or two honest, specific sentences"}}
"""


def get_all_verdicts_sequential(client, retrieval_results, model=DEFAULT_MODEL):
    """Slower, one-call-per-requirement fallback. Only used if the fast
    batched call above fails for some reason - keeps the app working
    rather than crashing."""
    full_results = []
    for item in retrieval_results:
        prompt = build_verdict_prompt(item["requirement"], item["matches"])
        try:
            response = client.messages.create(
                model=model, max_tokens=300, messages=[{"role": "user", "content": prompt}]
            )
            raw_text = response.content[0].text.strip()
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            verdict = json.loads(cleaned)
        except Exception as e:
            verdict = {"verdict": "Error", "reasoning": f"Could not get a verdict: {e}"}

        full_results.append({
            "requirement": item["requirement"],
            "matches": item["matches"],
            "verdict": verdict.get("verdict", "Error"),
            "reasoning": verdict.get("reasoning", ""),
        })
    return full_results


def get_all_verdicts(client, retrieval_results, model=DEFAULT_MODEL):
    """
    Public entry point app.py calls. Tries the fast batched approach first;
    only falls back to the slow, safe, one-by-one approach if the batch
    call fails or comes back malformed.
    """
    if not retrieval_results:
        return []
    try:
        return get_all_verdicts_batch(client, retrieval_results, model=model)
    except Exception:
        return get_all_verdicts_sequential(client, retrieval_results, model=model)


# ---------------------------------------------------------------------------
# STEP 3: Turn verdicts into one honest, explainable percentage
# ---------------------------------------------------------------------------

VERDICT_WEIGHTS = {"Yes": 1.0, "Partial": 0.5, "No": 0.0, "Error": 0.0}


def calculate_match_score(final_results):
    """
    Plain arithmetic, not an AI guess: Yes counts fully, Partial counts
    half, No counts zero - averaged across every requirement, as a
    percentage. Deterministic and easy to explain to anyone who asks
    "how was this number calculated?"
    """
    if not final_results:
        return 0
    total = sum(VERDICT_WEIGHTS.get(r["verdict"], 0.0) for r in final_results)
    return round((total / len(final_results)) * 100)
