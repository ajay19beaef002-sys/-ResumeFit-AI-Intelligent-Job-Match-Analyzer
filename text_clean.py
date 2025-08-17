from __future__ import annotations
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams

# Ensure NLTK data
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english')) - {"sql", "python", "java", "c++", "c#"}
LEMM = WordNetLemmatizer()

def basic_clean(text: str) -> str:
    """Basic text cleaning: lowercase, keep tech tokens, normalize spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+.#/+_-]", " ", text)  # Keep tech tokens like c++, c#
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> list[str]:
    """Tokenize text into words."""
    return word_tokenize(text)

def clean_and_lemmatize(text: str, n_grams: int = 2) -> str:
    """Clean, lemmatize, and add n-grams for multi-word phrases."""
    text = basic_clean(text)
    tokens = [t for t in tokenize(text) if t not in STOPWORDS]
    lemmas = [LEMM.lemmatize(t) for t in tokens]
    # Add bigrams
    n_grams_list = [' '.join(gram) for gram in ngrams(tokens, n_grams) if all(t not in STOPWORDS for t in gram)]
    return ' '.join(lemmas + n_grams_list)
