import re
import torch
from transformers import pipeline
from normal_llm import generate_answer
from similarity import compute_similarity
from nli_checker import count_contradictions

DEVICE = 0 if torch.cuda.is_available() else -1
print("Aware GPT-2 using:", "GPU" if DEVICE == 0 else "CPU")

# QA anchor model
qa_model = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=DEVICE
)

# Constrained GPT-2 generator
gpt2_generator = pipeline(
    "text-generation",
    model="gpt2",
    device=DEVICE
)

SIMILARITY_THRESHOLD = 0.20
QA_CONFIDENCE_THRESHOLD = 0.15


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_answer(answer):
    answer = str(answer).strip(" .,;:-")
    answer = re.sub(r"\s+(of|in|on|at|for|to|from)$", "", answer, flags=re.IGNORECASE)

    words = answer.split()
    if len(words) > 8:
        answer = " ".join(words[:6])

    return answer.strip(" .,;:-")


def retrieve_top_k_paragraphs(question, paragraphs, k=4):
    scored = []
    for p in paragraphs:
        score = compute_similarity(question, [p])
        scored.append((p, score))

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return [p[0] for p in ranked[:k]]


def build_candidate_contexts(question, top_paragraphs):
    candidate_contexts = []

    # Single paragraphs
    candidate_contexts.extend(top_paragraphs)

    # Paragraph pairs
    for i in range(len(top_paragraphs)):
        for j in range(i + 1, len(top_paragraphs)):
            candidate_contexts.append(top_paragraphs[i] + " " + top_paragraphs[j])

    # Sentence bundles
    all_sentences = []
    for para in top_paragraphs:
        all_sentences.extend(split_into_sentences(para))

    sentence_scores = []
    for s in all_sentences:
        score = compute_similarity(question, [s])
        sentence_scores.append((s, score))

    ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    top_sentences = [x[0] for x in ranked_sentences[:10]]

    if len(top_sentences) >= 4:
        candidate_contexts.append(" ".join(top_sentences[:4]))
    if len(top_sentences) >= 6:
        candidate_contexts.append(" ".join(top_sentences[:6]))

    # deduplicate
    seen = set()
    unique_contexts = []
    for c in candidate_contexts:
        norm = c.strip()
        if norm and norm not in seen:
            seen.add(norm)
            unique_contexts.append(norm)

    return unique_contexts


def extract_anchor_answer(question, candidate_contexts):
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

        if not answer:
            continue
        if len(answer.split()) > 10:
            continue

        adjusted_score = score
        if any(ch.isupper() for ch in answer):
            adjusted_score += 0.01

        if adjusted_score > best_score:
            best_score = adjusted_score
            best_answer = answer
            best_context = context
        elif abs(adjusted_score - best_score) < 0.03 and best_answer:
            if len(answer.split()) < len(best_answer.split()):
                best_answer = answer
                best_context = context

    return best_answer, best_score, best_context


def generate_constrained_gpt2_answer(question, context, anchor_answer):
    """
    GPT-2 is forced to stay short and near the QA anchor.
    """
    prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Verified short answer: {anchor_answer}\n"
        f"Final short answer:"
    )

    result = gpt2_generator(
        prompt,
        max_new_tokens=8,
        do_sample=False,
        temperature=0.2,
        truncation=True,
        pad_token_id=50256
    )

    output = result[0]["generated_text"]

    if "Final short answer:" in output:
        answer = output.split("Final short answer:")[-1].strip()
    else:
        answer = output.strip()

    answer = clean_answer(answer)

    # fallback if GPT-2 rambles or copies weird text
    if not answer:
        return anchor_answer
    if len(answer.split()) > 8:
        return anchor_answer
    if "question" in answer.lower() or "context" in answer.lower():
        return anchor_answer

    return answer


def hallucination_aware_gpt2_answer(question, paragraphs):
    # Step 1: retrieve evidence
    top_paragraphs = retrieve_top_k_paragraphs(question, paragraphs, k=4)

    # Step 2: build richer candidate contexts
    candidate_contexts = build_candidate_contexts(question, top_paragraphs)

    # Step 3: extract QA anchor
    anchor_answer, qa_score, best_context = extract_anchor_answer(question, candidate_contexts)

    # Step 4: contradiction check (soft)
    contradictions = count_contradictions(top_paragraphs)
    print("Aware GPT2 - Contradictions Found:", contradictions)
    print("Aware GPT2 - QA Confidence:", round(qa_score, 4))

    # Low-confidence fallback
    if qa_score < QA_CONFIDENCE_THRESHOLD:
        if top_paragraphs:
            fallback = split_into_sentences(top_paragraphs[0])
            if fallback:
                return clean_answer(fallback[0])
        return "Not enough evidence."

    # Step 5: grounding check on anchor
    if normalize_text(anchor_answer) not in normalize_text(best_context):
        return "Answer not supported by context."

    # Step 6: soft contradiction handling
    if contradictions >= 2 and qa_score < 0.25:
        return "Evidence conflict detected."

    # Step 7: GPT-2 generates only a short final answer
    gpt2_answer = generate_constrained_gpt2_answer(question, best_context, anchor_answer)

    # Step 8: final semantic grounding check
    similarity_score = compute_similarity(gpt2_answer, [best_context])
    print("Aware GPT2 - Similarity Score:", round(similarity_score, 4))

    if similarity_score < SIMILARITY_THRESHOLD:
        return anchor_answer

    return gpt2_answer