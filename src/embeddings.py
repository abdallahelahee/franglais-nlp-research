# src/embeddings.py

from sentence_transformers import SentenceTransformer

# Load a multilingual model (supports English + French)
_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def embed_texts(text_list):
    """
    Takes a list of preprocessed sentences and returns embeddings.
    """
    return _model.encode(text_list)
