# 🎯 Honest Fit Checker

A RAG (Retrieval-Augmented Generation) tool that gives an honest,
no-fabrication verdict on how well a resume actually matches a job
description — requirement by requirement, with reasoning, not just a
generic "you're a great fit!" summary.

Built as a hands-on project to learn how a real RAG pipeline works end to
end: not a tutorial copy-paste, but a working tool with its own design
decisions, bugs found through real testing, and fixes made because of what
that testing showed.

## What it does

1. You paste a job description and a resume.
2. Claude reads the job posting and checks it's actually a job description
   (not junk input), then extracts the company, job title, salary/location
   if stated, and the *real* requirements — ignoring boilerplate like
   referral bonuses, requisition IDs, and mission statements.
3. A TF-IDF + cosine similarity search (classic text-similarity math, no
   AI involved in this step) finds the resume bullets most relevant to
   each requirement.
4. Claude judges each requirement honestly as a genuine match, a partial
   match, or a gap — using the retrieved resume bullets as evidence, with
   explicit instructions not to be generous or assume anything the resume
   doesn't actually say.
5. The app shows an overall match percentage (calculated with plain
   arithmetic from the verdicts, not invented by the AI, so it's always
   explainable), a summary, and a full requirement-by-requirement
   breakdown you can expand.

## Why RAG, and why this order matters

**Retrieval happens first, then feeds into Generation** — not the other
way around. The text-similarity step (Retrieval) narrows down *which*
resume bullets are even worth showing Claude for a given requirement,
before Claude (Generation) makes the actual judgment call. Handing Claude
the whole resume for every single requirement would be slower, more
expensive, and noisier; Retrieval focuses the context first.

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   app.py     │─────▶│   retrieval.py    │─────▶│   generator.py    │
│ (Streamlit   │      │ (TF-IDF + cosine  │      │ (Claude API calls: │
│  interface)  │◀─────│  similarity)      │◀─────│  validate JD,      │
└─────────────┘      └──────────────────┘      │  judge fit,        │
                                                  │  score match)      │
                                                  └─────────────────┘
```

- **`app.py`** — the Streamlit web interface: text boxes, the button, and
  the results display.
- **`retrieval.py`** — the "R" in RAG. Pure text-similarity math (no AI
  model call) to find which resume bullets are most relevant to each
  requirement.
- **`generator.py`** — the "G" in RAG. All calls to the Claude API: job
  posting validation/extraction, the honest per-requirement verdicts, and
  the match-score calculation.

## A real bug found through testing (and why it matters)

The first working version split the job description on every line break
and treated *every* line as a "requirement" — including the job title, a
requisition ID, and an employee referral bonus blurb. That produced a
misleading "30 gaps out of 40" result, because most of those weren't real
requirements at all.

The fix went through two iterations:
1. First, a keyword-based filter (regex matching on words like "years",
   "degree", "certified") to guess which lines were real requirements.
2. Then, a better fix: let Claude read the whole posting and extract the
   real requirements directly, since that's a judgment call Claude is far
   better suited for than pattern-matching. The keyword filter is kept in
   `retrieval.py` only as a silent fallback if that extraction call ever
   fails.

This is also why the app now makes far fewer API calls than the first
version: instead of one Claude call *per requirement* (up to 40 separate,
slow round trips), it makes one call to validate/extract the job posting
and one batched call to judge every requirement at once — with an
automatic slower fallback only if that batched call ever fails.

## Running it locally

Requires Python 3.10+ and your own [Anthropic API
key](https://console.anthropic.com) (entered in the app itself — never
stored, never hardcoded in the code).

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

- **Streamlit** — web interface
- **scikit-learn** — TF-IDF vectorization and cosine similarity (Retrieval)
- **Anthropic API (Claude)** — job posting analysis and honest fit judgment (Generation)

## Author

Built by [Sushant Malhotra](https://www.linkedin.com/in/sushantmalhotra) —
Senior Product Manager, founder of [Vizenova](https://vizenova.ca).
