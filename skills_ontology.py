import pandas as pd
from nltk.corpus import wordnet

CORE_SKILLS = {
    "python", "java", "c++", "c#", "sql", "nosql", "mysql", "postgresql", "mongodb",
    "pandas", "numpy", "scikit-learn", "sklearn", "matplotlib", "seaborn",
    "tensorflow", "pytorch", "keras", "nlp", "bert", "transformers",
    "machine learning", "deep learning", "statistics", "data analysis",
    "power bi", "tableau", "excel", "aws", "gcp", "azure",
    "spark", "hadoop", "airflow", "docker", "kubernetes", "git"
}

ALIASES = {
    "scikit-learn": {"sklearn"},
    "tensorflow": {"tf"},
    "pytorch": {"torch"},
    "machine learning": {"ml"}
}

def load_skills_ontology(file_path: str = "src/skills.csv") -> dict:
    """Load skills ontology from CSV or use default."""
    try:
        df = pd.read_csv(file_path)
        ontology = {row['skill']: set(row['aliases'].split(',')) for _, row in df.iterrows()}
        ontology.update(ALIASES)
        return ontology
    except FileNotFoundError:
        return ALIASES

def get_synonyms(skill: str) -> set:
    """Get synonyms for a skill using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(skill):
        synonyms.update(lemma.name().lower().replace('_', ' ') for lemma in syn.lemmas())
    return synonyms

def normalize_skill(skill: str, ontology: dict) -> str:
    """Normalize skill to canonical form."""
    s = skill.lower().strip()
    for k, vals in ontology.items():
        if s in vals or s == k:
            return k
    return s

def extract_skills(tokens: list[str], ontology: dict) -> set:
    """Extract skills from tokens using ontology and synonyms."""
    found = set()
    for tok in tokens:
        t = normalize_skill(tok, ontology)
        if t in CORE_SKILLS or t in ontology:
            found.add(t)
        # Check synonyms
        synonyms = get_synonyms(t)
        for syn in synonyms:
            if syn in CORE_SKILLS or syn in ontology:
                found.add(normalize_skill(syn, ontology))
    return found
