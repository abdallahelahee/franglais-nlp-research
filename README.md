# Franglais NLP Research (French ↔ English)

A research project exploring **cross-lingual semantic analysis** using academic texts in French and English.

## Objectives
- Build a clean bilingual NLP dataset  
- Perform preprocessing (tokenization, lemmatization, stopwords)
- Apply topic modelling (LDA & BERTopic)
- Generate multilingual embeddings (Sentence-BERT)
- Visualize clusters and similarity relationships
- Build an interactive dashboard

## Project Structure
bilingual-nlp-research/
├── src/
│   ├── preprocessing.py
│   ├── topic_model.py
│   ├── embeddings.py
│   ├── visualize.py
│
├── notebooks/
│   ├── data-exploration.ipynb
│   ├── topic-modelling.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── dashboard/
│   └── app.py

##  Installation
pip install -r requirements.txt
