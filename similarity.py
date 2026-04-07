from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(question, paragraphs):
    q_embedding = embedding_model.encode([question])
    p_embeddings = embedding_model.encode(paragraphs)

    scores = cosine_similarity(q_embedding, p_embeddings)[0]
    return float(max(scores))