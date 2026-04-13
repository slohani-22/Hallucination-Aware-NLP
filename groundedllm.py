import re
import torch
from transformers import pipeline
from similarity import compute_similarity
from nli_checker import count_contradictions

DEVICE = 0 if torch.cuda.is_available() else -1
print("Aware LLM using:", "GPU" if DEVICE == 0 else "CPU")

qa_model = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=DEVICE
)

# Lower threshold for better coverage
QA_CONFIDENCE_THRESHOLD = 0.15


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def normalize_answer_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_answer(answer):
    answer = str(answer).strip(" .,;:-")
    answer = re.sub(r"\s+(of|in|on|at|for|to|from)$", "", answer.strip(), flags=re.IGNORECASE)

    # trim very long answers
    words = answer.split()
    if len(words) > 8:
        answer = " ".join(words[:6])

    return answer.strip(" .,;:-")


def retrieve_top_k_paragraphs(question, paragraphs, k=4):
    scored_paragraphs = []

    for p in paragraphs:
        score = compute_similarity(question, [p])
        scored_paragraphs.append((p, score))

    ranked = sorted(scored_paragraphs, key=lambda x: x[1], reverse=True)
    return [p[0] for p in ranked[:k]]


def build_candidate_contexts(question, top_paragraphs):
    candidate_contexts = []

    # Single paragraphs
    candidate_contexts.extend(top_paragraphs)

    # Paragraph pairs
    for i in range(len(top_paragraphs)):
        for j in range(i + 1, len(top_paragraphs)):
            candidate_contexts.append(top_paragraphs[i] + " " + top_paragraphs[j])

    # Sentence-level candidates
    all_sentences = []
    for para in top_paragraphs:
        all_sentences.extend(split_into_sentences(para))

    sentence_scores = []
    for s in all_sentences:
        score = compute_similarity(question, [s])
        sentence_scores.append((s, score))

    ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    top_sentences = [x[0] for x in ranked_sentences[:10]]

    # More sentence bundles for multi-hop coverage
    if len(top_sentences) >= 4:
        candidate_contexts.append(" ".join(top_sentences[:4]))
    if len(top_sentences) >= 6:
        candidate_contexts.append(" ".join(top_sentences[:6]))

    # Remove duplicates
    seen = set()
    unique_contexts = []
    for c in candidate_contexts:
        norm = c.strip()
        if norm and norm not in seen:
            seen.add(norm)
            unique_contexts.append(norm)

    return unique_contexts


def extract_best_answer(question, candidate_contexts):
    best_answer = ""
    best_score = 0.0
    best_context = ""

    for context in candidate_contexts:
        result = qa_model(
            question=question,
            context=context,
            truncation=True
        )

        answer = clean_answer(result["answer"])
        score = float(result["score"])

        # Skip empty or overly long noisy spans
        if not answer:
            continue
        if len(answer.split()) > 10:
            continue

        # Small preference for named entities / proper nouns
        adjusted_score = score
        if any(ch.isupper() for ch in answer):
            adjusted_score += 0.01

        # Prefer higher score
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_answer = answer
            best_context = context

        # If scores are close, prefer shorter cleaner answer
        elif abs(adjusted_score - best_score) < 0.03 and best_answer:
            if len(answer.split()) < len(best_answer.split()):
                best_answer = answer
                best_context = context

    return best_answer, best_score, best_context


def fallback_answer_from_context(top_paragraphs):
    if not top_paragraphs:
        return "Not enough evidence."

    sentences = split_into_sentences(top_paragraphs[0])
    if not sentences:
        return "Not enough evidence."

    fallback = clean_answer(sentences[0])

    if fallback:
        return fallback

    return "Not enough evidence."


def hallucination_aware_answer(question, paragraphs):
    top_paragraphs = retrieve_top_k_paragraphs(question, paragraphs, k=4)

    candidate_contexts = build_candidate_contexts(question, top_paragraphs)
    answer, qa_score, best_context = extract_best_answer(question, candidate_contexts)

    contradictions = count_contradictions(top_paragraphs)

    print("Contradictions Found:", contradictions)
    print("QA Confidence:", round(qa_score, 4))

    # Low-confidence fallback instead of direct rejection
    if qa_score < QA_CONFIDENCE_THRESHOLD:
        return fallback_answer_from_context(top_paragraphs)

    # Grounding check
    if normalize_answer_text(answer) not in normalize_answer_text(best_context):
        return fallback_answer_from_context(top_paragraphs)

    # Much softer contradiction rule
    if contradictions >= 2 and qa_score < 0.25:
        return fallback_answer_from_context(top_paragraphs)

    answer = clean_answer(answer)
    if not answer:
        return fallback_answer_from_context(top_paragraphs)

    return answer