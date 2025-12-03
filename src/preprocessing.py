import spacy
import re
import string
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOP

# Load SpaCy models
nlp_en = spacy.load("en_core_web_md")
nlp_fr = spacy.load("fr_core_news_md")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def preprocess_text(text, lang="en"):
    text = clean_text(text)

    if lang == "en":
        doc = nlp_en(text)
        stop_words = EN_STOP
    else:
        doc = nlp_fr(text)
        stop_words = FR_STOP

    lemmatized = [
        token.lemma_
        for token in doc
        if token.lemma_ not in stop_words
        and token.lemma_.strip() != ""
        and token.is_alpha
    ]

    return " ".join(lemmatized)
