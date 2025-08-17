import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.resume_extractor import extract_text_generic
from src.text_clean import clean_and_lemmatize, tokenize
from src.skills_ontology import load_skills_ontology, extract_skills

def compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """Compute TF-IDF cosine similarity."""
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X = vec.fit_transform([resume_text, jd_text])
    sim = cosine_similarity(X[0:1], X[1:2])[0,0]
    return float(sim)

def gap_analysis(resume_text: str, jd_text: str, ontology: dict):
    """Perform skill gap analysis."""
    res_tokens = tokenize(resume_text)
    jd_tokens = tokenize(jd_text)
    res_skills = extract_skills(res_tokens, ontology)
    jd_skills = extract_skills(jd_tokens, ontology)
    matched = sorted(res_skills & jd_skills)
    missing = sorted(jd_skills - res_skills)
    extras = sorted(res_skills - jd_skills)
    return matched, missing, extras

def main(args):
    try:
        raw_resume = extract_text_generic(args.resume)
        raw_jd = extract_text_generic(args.jd)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return

    resume = clean_and_lemmatize(raw_resume)
    jd = clean_and_lemmatize(raw_jd)

    if not resume.strip() or not jd.strip():
        print("Error: Resume or JD is empty after preprocessing.")
        return

    ontology = load_skills_ontology()
    sim = compute_tfidf_similarity(resume, jd)
    matched, missing, extras = gap_analysis(resume, jd, ontology)

    print(f"TF-IDF Fit Score: {sim*100:.2f}%")
    print("Matched Skills:", ", ".join(matched) or "-")
    print("Missing Skills:", ", ".join(missing) or "-")
    print("Extra Skills (in resume only):", ", ".join(extras) or "-")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="Path to resume file (pdf/docx/txt)")
    parser.add_argument("--jd", required=True, help="Path to job description (txt/pdf/docx)")
    args = parser.parse_args()
    main(args)
