"""
app.py
------
This is the interface - it ties Retrieval (retrieval.py) and Generation
(generator.py) together into an actual website with text boxes and a
button, using Streamlit.

Plain-language explanation:
Streamlit turns a normal Python script into a website almost automatically.
Every time you write "st.something", that line becomes a visible piece of
a webpage - a text box, a button, a heading, whatever.

The overall flow, updated after real testing feedback:
  1. Get the resume and JD text from the user (two text boxes)
  2. When they click "Check My Fit":
     a. First, ONE fast Claude call reads the pasted JD text and decides:
        is this actually a job posting? If not, stop immediately and tell
        the user plainly - no fabricated analysis of junk input.
        If yes, it also extracts company / job title / salary / location
        and the real requirements (ignoring boilerplate).
     b. Retrieval finds the resume bullets most similar to each real
        requirement.
     c. ONE more Claude call judges every requirement at once (fast),
        with an automatic slower fallback only if that ever fails.
  3. Show company/title/salary/location and one honest match percentage
     right away. The full requirement-by-requirement breakdown is tucked
     behind a single "Show full breakdown" toggle instead of dumped on
     screen by default.
"""

import streamlit as st
import anthropic
from retrieval import find_best_matches
from generator import analyze_job_posting, get_all_verdicts, calculate_match_score

st.set_page_config(page_title="Honest Fit Checker", page_icon="🎯", layout="wide")

st.title("🎯 Honest Fit Checker")
st.markdown(
    "Paste a job description and a resume below. This tool will honestly tell you "
    "which requirements are genuinely met, partially met, or not met at all - "
    "no flattery, no fabrication."
)

# ---- API Key input ----
# We ask the user for their own Anthropic API key rather than hard-coding one.
# This is standard practice: never put a real API key directly in code that
# might end up on GitHub, since anyone could see and use it.
api_key = st.text_input(
    "Your Anthropic API key",
    type="password",
    help="Get one at console.anthropic.com. It's never stored - only used for this session."
)

col1, col2 = st.columns(2)
with col1:
    jd_text = st.text_area("Paste the Job Description here", height=300)
with col2:
    resume_text = st.text_area("Paste the Resume here", height=300)

if st.button("Check My Fit", type="primary"):
    if not api_key:
        st.error("Please enter your Anthropic API key above.")
    elif not jd_text or not resume_text:
        st.error("Please paste both a job description and a resume.")
    else:
        client = anthropic.Anthropic(api_key=api_key)

        # ---- STEP 1: Understand and validate what was pasted ----
        with st.spinner("Reading the job posting..."):
            try:
                analysis = analyze_job_posting(client, jd_text)
            except Exception as e:
                st.error(f"Couldn't reach Claude to analyze this text: {e}")
                st.stop()

        if not analysis.get("is_job_description", False):
            st.error(
                "⚠️ This doesn't look like a job description. "
                f"{analysis.get('reason', 'Please check what you pasted and try again.')}"
            )
            st.stop()

        requirements = analysis.get("requirements") or []
        if not requirements:
            st.warning(
                "This looks like a job posting, but no clear requirements could be "
                "found in it. Try pasting the full posting, including the "
                "qualifications/requirements section."
            )
            st.stop()

        # ---- Job posting summary card ----
        st.subheader("Job Posting Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Company", analysis.get("company") or "Not stated")
        s2.metric("Role", analysis.get("job_title") or "Not stated")
        s3.metric("Salary", analysis.get("salary") or "Not stated")
        s4.metric("Location", analysis.get("location") or "Not stated")

        # ---- STEP 2: Retrieval - find candidate matches using text similarity ----
        with st.spinner("Matching your resume against each requirement..."):
            retrieval_results = find_best_matches(requirements, resume_text, top_n=2)

        if not retrieval_results:
            st.warning("Could not find enough distinct lines in the resume to compare. Try adding more detail.")
            st.stop()

        # ---- STEP 3: Generation - ask Claude to make the honest judgment call ----
        with st.spinner(f"Asking Claude to judge {len(retrieval_results)} requirements honestly..."):
            final_results = get_all_verdicts(client, retrieval_results)

        # ---- Overall match score (calculated with plain arithmetic - see generator.py) ----
        score = calculate_match_score(final_results)
        if score >= 70:
            score_icon = "🟢"
        elif score >= 40:
            score_icon = "🟡"
        else:
            score_icon = "🔴"

        st.subheader("Overall Match Score")
        st.markdown(f"## {score_icon} {score}%")
        st.caption(
            "Calculated as: Yes = 100%, Partial = 50%, No = 0%, averaged across all "
            "requirements Claude judged. This is an AI-generated estimate based on the "
            "resume text provided, not a certified or guaranteed assessment."
        )

        # ---- Summary counts ----
        yes_count = sum(1 for r in final_results if r["verdict"] == "Yes")
        partial_count = sum(1 for r in final_results if r["verdict"] == "Partial")
        no_count = sum(1 for r in final_results if r["verdict"] == "No")

        m1, m2, m3 = st.columns(3)
        m1.metric("✅ Genuine Matches", yes_count)
        m2.metric("🟡 Partial Matches", partial_count)
        m3.metric("❌ Gaps", no_count)

        # ---- Detailed breakdown - hidden by default ----
        with st.expander(f"Show full requirement-by-requirement breakdown ({len(final_results)} requirements)"):
            for r in final_results:
                icon = {"Yes": "✅", "Partial": "🟡", "No": "❌", "Error": "⚠️"}.get(r["verdict"], "⚠️")
                with st.expander(f"{icon} {r['requirement']}"):
                    st.write(f"**Verdict:** {r['verdict']}")
                    st.write(f"**Why:** {r['reasoning']}")
                    st.write("**Closest resume bullets found:**")
                    for m in r["matches"]:
                        st.write(f"- \"{m['bullet']}\" (similarity: {m['score']})")

st.markdown("---")
st.caption("Built by Sushant Malhotra. This tool never stores your data - everything happens in this session only.")
