from sentence_transformers import SentenceTransformer
import numpy as np

# Load a multilingual embedding model
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def embed_texts(texts):
    """
    Input: list of strings
    Output: numpy array of embeddings
    """
    embeddings = embed_model.encode(texts)
    return np.array(embeddings)
