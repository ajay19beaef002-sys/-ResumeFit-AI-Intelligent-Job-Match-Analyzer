import streamlit as st
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.resume_extractor import extract_text_generic
from src.text_clean import clean_and_lemmatize, tokenize
from src.skills_ontology import load_skills_ontology, extract_skills
from presidio_analyzer import AnalyzerEngine

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("🧠 AI-Powered Resume Analyzer")
st.caption("Compute job fit score, see matched & missing skills, and get course recommendations.")

# Input widgets
mode = st.radio("Embedding Mode", ["SBERT (recommended)", "TF-IDF (baseline)"])
uploaded_resume = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
jd_text = st.text_area("Paste Job Description")
threshold = st.slider("Fit Score Threshold for Good Fit (%)", 0, 100, 70)

run = st.button("Analyze")

@st.cache_resource
def get_sbert():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def anonymize_text(text: str) -> str:
    """Anonymize PII using Presidio."""
    try:
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"], language="en")
        for result in results:
            text = text[:result.start] + "*" * (result.end - result.start) + text[result.end:]
        return text
    except Exception:
        return text  # Fallback to original text if anonymization fails

def plot_wordcloud(skills: list, title: str):
    """Generate and display word cloud."""
    if skills:
        wc = WordCloud(width=400, height=200, background_color='white').generate(' '.join(skills))
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        st.pyplot(fig)

def highlight_skills(text: str, matched: list, missing: list) -> str:
    """Highlight matched and missing skills in text."""
    for skill in matched:
        text = text.replace(skill, f'<span style="color:green; font-weight:bold">{skill}</span>')
    for skill in missing:
        text = text.replace(skill, f'<span style="color:red; font-weight:bold">{skill}</span>')
    return text

COURSE_MAP = {
    "python": ["Python for Everybody (Coursera)", "Learn Python - FreeCodeCamp (YouTube)"],
    "sql": ["SQL for Data Science (Coursera)", "SQL Tutorial for Beginners (YouTube)"],
    "machine learning": ["Machine Learning by Andrew Ng (Coursera)", "ML Crash Course (Google)"],
    "aws": ["AWS Certified Solutions Architect (Udemy)", "AWS Fundamentals (YouTube)"]
}

def suggest_courses(skills: list) -> dict:
    """Suggest courses for missing skills."""
    return {skill: COURSE_MAP.get(skill, ["No courses found"]) for skill in skills[:3]}

if run:
    if not uploaded_resume or not jd_text.strip():
        st.warning("Please upload a resume and paste a job description.")
        st.stop()

    # Process resume
    try:
        resume_bytes = uploaded_resume.read()
        with open(".tmp_resume", "wb") as f:
            f.write(resume_bytes)
        raw_resume = extract_text_generic(".tmp_resume" if uploaded_resume.name.endswith(".txt") else ".tmp_resume")
        if not raw_resume.strip():
            with open(".tmp_resume", "wb") as f:
                f.write(resume_bytes)
            raw_resume = extract_text_generic(uploaded_resume.name)
        os.remove(".tmp_resume")
    except Exception as e:
        st.error(f"Error processing resume: {str(e)}")
        st.stop()

    # Anonymize and preprocess
    raw_resume = anonymize_text(raw_resume)
    raw_jd = anonymize_text(jd_text)
    resume = clean_and_lemmatize(raw_resume)
    jd = clean_and_lemmatize(raw_jd)

    if not resume.strip() or not jd.strip():
        st.error("Resume or JD is empty after preprocessing.")
        st.stop()

    # Compute similarity
    ontology = load_skills_ontology()
    if mode.startswith("SBERT"):
        model = get_sbert()
        embs = model.encode([resume, jd])
        score = cosine_similarity([embs[0]], [embs[1]])[0,0]
    else:
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
        X = vec.fit_transform([resume, jd])
        score = cosine_similarity(X[0:1], X[1:2])[0,0]

    # Skill analysis
    res_tokens = tokenize(resume)
    jd_tokens = tokenize(jd)
    res_skills = extract_skills(res_tokens, ontology)
    jd_skills = extract_skills(jd_tokens, ontology)
    matched = sorted(res_skills & jd_skills)
    missing = sorted(jd_skills - res_skills)
    extras = sorted(res_skills - jd_skills)

    # Display results
    st.subheader(f"Fit Score: {score*100:.2f}%")
    st.write(f"Fit: {'Good' if score*100 > threshold else 'Poor'}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Matched Skills")
        st.write(", ".join(matched) or "-")
        plot_wordcloud(matched, "Matched Skills")
    with col2:
        st.markdown("### ❌ Missing Skills")
        st.write(", ".join(missing) or "-")
        plot_wordcloud(missing, "Missing Skills")

    st.markdown("### ➕ Extra Skills in Resume")
    st.write(", ".join(extras) or "-")

    st.markdown("### 📄 Job Description with Highlights")
    st.markdown(highlight_skills(jd_text.lower(), matched, missing), unsafe_allow_html=True)

    if missing:
        st.markdown("### 📚 Recommended Courses")
        courses = suggest_courses(missing)
        for skill, course_list in courses.items():
            st.write(f"**{skill.capitalize()}**: {', '.join(course_list)}")

    st.markdown("---")
    st.markdown("**Tips**: Tailor your resume to include missing skills, quantify achievements, and align with JD keywords.")
