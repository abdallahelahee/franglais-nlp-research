import spacy
import re
import string
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOP

# Load SpaCy models (optional for minimal preprocessing)
nlp_en = spacy.load("en_core_web_md")
nlp_fr = spacy.load("fr_core_news_md")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def preprocess_text(text, lang=None):
    """
    Minimal preprocessing: keep all words intact, remove only leading/trailing whitespace.
    Do not force lang-specific cleaning.
    """
    return text.strip()
