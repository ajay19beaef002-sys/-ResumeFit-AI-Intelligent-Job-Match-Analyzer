import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.resume_extractor import extract_text_generic
from src.text_clean import clean_and_lemmatize, tokenize
from src.skills_ontology import load_skills_ontology, extract_skills

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def sbert_similarity(resume_text: str, jd_text: str) -> float:
    """Compute SBERT cosine similarity."""
    model = SentenceTransformer(MODEL_NAME)
    embs = model.encode([resume_text, jd_text])
    sim = cosine_similarity([embs[0]], [embs[1]])[0,0]
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
    sim = sbert_similarity(resume, jd)
    matched, missing, extras = gap_analysis(resume, jd, ontology)

    print(f"SBERT Fit Score: {sim*100:.2f}%")
    print("Matched Skills:", ", ".join(matched) or "-")
    print("Missing Skills:", ", ".join(missing) or "-")
    print("Extra Skills (in resume only):", ", ".join(extras) or "-")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--jd", required=True)
    args = parser.parse_args()
    main(args)
