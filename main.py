import re
import pandas as pd
from data_loader import load_data, get_paragraphs
from normal_llm import generate_answer
from aware_gpt2 import hallucination_aware_gpt2_answer
from groundedllm import hallucination_aware_answer


def token_f1(correct_answer, predicted_answer):
    gold_tokens = normalize_text(correct_answer).split()
    pred_tokens = normalize_text(predicted_answer).split()

    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    shared = sum(min(pred_counts.get(t, 0), gold_counts[t]) for t in gold_counts)

    if shared == 0:
        return 0.0

    precision = shared / len(pred_tokens)
    recall = shared / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a, b):
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(correct_answer, predicted_answer):
    gold_tokens = normalize_text(correct_answer).split()
    pred_tokens = normalize_text(predicted_answer).split()

    if not gold_tokens or not pred_tokens:
        return 0.0

    lcs = lcs_length(gold_tokens, pred_tokens)

    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower().strip()

    replacements = {
        " ii ": " 2 ",
        " iii ": " 3 ",
        " iv ": " 4 ",
        " v ": " 5 ",
        " world war ii ": " world war 2 ",
        " world war two ": " world war 2 ",
    }

    text = f" {text} "
    for old, new in replacements.items():
        text = text.replace(old, new)

    # remove punctuation except commas for list matching
    text = re.sub(r"[^\w\s,]", " ", text)

    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_name_variant(correct, predicted):
    """
    Count person-name variants as correct when the shorter name appears
    inside the longer one in the same token order.

    Examples:
    - 'roy orbison' vs 'roy kelton orbison' -> True
    - 'gregory hines' vs 'dancer gregory hines' -> True
    """
    c_tokens = correct.split()
    p_tokens = predicted.split()

    # Only apply this to answers that look like names / multiword entities
    if len(c_tokens) < 2 or len(p_tokens) < 2:
        return False

    # shorter vs longer token list
    if len(c_tokens) <= len(p_tokens):
        short_tokens = c_tokens
        long_tokens = p_tokens
    else:
        short_tokens = p_tokens
        long_tokens = c_tokens

    i = 0
    for token in long_tokens:
        if i < len(short_tokens) and token == short_tokens[i]:
            i += 1

    return i == len(short_tokens)


def inclusive_match(correct_answer, predicted_answer):
    correct = normalize_text(correct_answer)
    pred = normalize_text(predicted_answer)

    if not pred:
        return False

    # exact match
    if correct == pred:
        return True

    # substring / containment
    if pred in correct or correct in pred:
        return True

    # list-style answers: if correct is "a, b, c" and pred is "c"
    correct_parts = [x.strip() for x in correct.split(",") if x.strip()]
    if pred in correct_parts:
        return True

    # person / entity name variant matching
    if is_name_variant(correct, pred):
        return True

    return False


# Rejection phrases used by aware systems
REJECTION_PHRASES = [
    "not enough evidence",
    "evidence conflict detected",
    "answer not supported by context"
]


def is_rejection(answer):
    return normalize_text(answer) in [normalize_text(r) for r in REJECTION_PHRASES]


dataset = load_data(sample_size=200)
results = []

print("Running evaluation on 200 questions...\n")

for i, sample in enumerate(dataset):
    print(f"\n--- Processing Question {i+1}/200 ---")

    question = sample["question"]
    correct_answer = sample["answer"]
    paragraphs = get_paragraphs(sample)

    # System 1: Normal LLM — GPT-2, no verification
    normal_answer = generate_answer(question, paragraphs)

    # System 2: Hallucination-Aware LLM — GPT-2 + verification pipeline
    aware_gpt2_ans = hallucination_aware_gpt2_answer(question, paragraphs)

    # System 3: Grounded LLM — RoBERTa + full extraction pipeline
    aware_answer = hallucination_aware_answer(question, paragraphs)

    # --- Exact match scoring ---
    normal_exact        = normalize_text(correct_answer) == normalize_text(normal_answer)
    aware_gpt2_exact    = normalize_text(correct_answer) == normalize_text(aware_gpt2_ans)
    aware_exact         = normalize_text(correct_answer) == normalize_text(aware_answer)

    # --- Inclusive match scoring ---
    normal_inclusive        = inclusive_match(correct_answer, normal_answer)
    aware_gpt2_inclusive    = inclusive_match(correct_answer, aware_gpt2_ans)
    aware_inclusive         = inclusive_match(correct_answer, aware_answer)

    # --- Token-Level F1 scoring ---
    normal_f1       = token_f1(correct_answer, normal_answer)
    aware_gpt2_f1   = token_f1(correct_answer, aware_gpt2_ans)
    aware_f1        = token_f1(correct_answer, aware_answer)

    # --- ROUGE-L scoring ---
    normal_rouge        = rouge_l(correct_answer, normal_answer)
    aware_gpt2_rouge    = rouge_l(correct_answer, aware_gpt2_ans)
    aware_rouge         = rouge_l(correct_answer, aware_answer)

    # --- Rejection tracking ---
    aware_gpt2_rejected = is_rejection(aware_gpt2_ans)
    aware_rejected      = is_rejection(aware_answer)

    results.append({
        "Question":                             question,
        "HotpotQA Correct Answer":              correct_answer,

        # System 1 — Normal LLM
        "Normal LLM Answer":                    normal_answer,
        "Normal Exact Correct":                 normal_exact,
        "Normal Inclusive Correct":             normal_inclusive,
        "Normal Token F1":                      normal_f1,
        "Normal ROUGE-L":                       normal_rouge,

        # System 2 — Hallucination-Aware LLM (GPT-2 + Pipeline)
        "Aware GPT2 Answer":                    aware_gpt2_ans,
        "Aware GPT2 Exact Correct":             aware_gpt2_exact,
        "Aware GPT2 Inclusive Correct":         aware_gpt2_inclusive,
        "Aware GPT2 Token F1":                  aware_gpt2_f1,
        "Aware GPT2 ROUGE-L":                   aware_gpt2_rouge,
        "Aware GPT2 Rejected":                  aware_gpt2_rejected,

        # System 3 — Grounded LLM (RoBERTa + Pipeline)
        "Aware LLM Answer":                     aware_answer,
        "Aware Exact Correct":                  aware_exact,
        "Aware Inclusive Correct":              aware_inclusive,
        "Aware Token F1":                       aware_f1,
        "Aware ROUGE-L":                        aware_rouge,
        "Aware Rejected":                       aware_rejected,
    })

df = pd.DataFrame(results)

# --- Compute accuracy metrics ---
normal_exact_acc            = df["Normal Exact Correct"].mean() * 100
aware_gpt2_exact_acc        = df["Aware GPT2 Exact Correct"].mean() * 100
aware_exact_acc             = df["Aware Exact Correct"].mean() * 100

normal_inclusive_acc        = df["Normal Inclusive Correct"].mean() * 100
aware_gpt2_inclusive_acc    = df["Aware GPT2 Inclusive Correct"].mean() * 100
aware_inclusive_acc         = df["Aware Inclusive Correct"].mean() * 100

# --- Compute Token-Level F1 averages ---
normal_f1_avg       = df["Normal Token F1"].mean() * 100
aware_gpt2_f1_avg   = df["Aware GPT2 Token F1"].mean() * 100
aware_f1_avg        = df["Aware Token F1"].mean() * 100

# --- Compute ROUGE-L averages ---
normal_rouge_avg        = df["Normal ROUGE-L"].mean() * 100
aware_gpt2_rouge_avg    = df["Aware GPT2 ROUGE-L"].mean() * 100
aware_rouge_avg         = df["Aware ROUGE-L"].mean() * 100

# --- Compute rejection rates ---
aware_gpt2_rejection_rate   = df["Aware GPT2 Rejected"].mean() * 100
aware_rejection_rate        = df["Aware Rejected"].mean() * 100

# --- Build summary dataframe ---
summary_df = pd.DataFrame([
    {"Metric": "Normal LLM Exact Accuracy (%)",                   "Value": round(normal_exact_acc, 2)},
    {"Metric": "Hallucination-Aware LLM Exact Accuracy (%)",      "Value": round(aware_gpt2_exact_acc, 2)},
    {"Metric": "Grounded LLM Exact Accuracy (%)",                 "Value": round(aware_exact_acc, 2)},
    {"Metric": "",                                                 "Value": ""},
    {"Metric": "Normal LLM Inclusive Accuracy (%)",               "Value": round(normal_inclusive_acc, 2)},
    {"Metric": "Hallucination-Aware LLM Inclusive Accuracy (%)",  "Value": round(aware_gpt2_inclusive_acc, 2)},
    {"Metric": "Grounded LLM Inclusive Accuracy (%)",             "Value": round(aware_inclusive_acc, 2)},
    {"Metric": "",                                                 "Value": ""},
    {"Metric": "Normal LLM Token F1 (%)",                         "Value": round(normal_f1_avg, 2)},
    {"Metric": "Hallucination-Aware LLM Token F1 (%)",            "Value": round(aware_gpt2_f1_avg, 2)},
    {"Metric": "Grounded LLM Token F1 (%)",                       "Value": round(aware_f1_avg, 2)},
    {"Metric": "",                                                 "Value": ""},
    {"Metric": "Normal LLM ROUGE-L (%)",                          "Value": round(normal_rouge_avg, 2)},
    {"Metric": "Hallucination-Aware LLM ROUGE-L (%)",             "Value": round(aware_gpt2_rouge_avg, 2)},
    {"Metric": "Grounded LLM ROUGE-L (%)",                        "Value": round(aware_rouge_avg, 2)},
    {"Metric": "",                                                 "Value": ""},
    {"Metric": "Hallucination-Aware LLM Rejection Rate (%)",      "Value": round(aware_gpt2_rejection_rate, 2)},
    {"Metric": "Grounded LLM Rejection Rate (%)",                 "Value": round(aware_rejection_rate, 2)},
])

output_file = "LLM_Evaluation_Output_Final.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Results", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

# --- Print summary to console ---
print("\n========== EVALUATION COMPLETE ==========")
print(f"Output saved to: {output_file}\n")
print(f"{'System':<45} {'Exact':>8} {'Inclusive':>10} {'Token F1':>10} {'ROUGE-L':>9} {'Reject':>8}")
print("-" * 95)
print(f"{'Normal LLM (GPT-2)':<45} {normal_exact_acc:>7.2f}% {normal_inclusive_acc:>9.2f}% {normal_f1_avg:>9.2f}% {normal_rouge_avg:>8.2f}% {'N/A':>8}")
print(f"{'Hallucination-Aware LLM (GPT-2 + Pipeline)':<45} {aware_gpt2_exact_acc:>7.2f}% {aware_gpt2_inclusive_acc:>9.2f}% {aware_gpt2_f1_avg:>9.2f}% {aware_gpt2_rouge_avg:>8.2f}% {aware_gpt2_rejection_rate:>7.2f}%")
print(f"{'Grounded LLM (RoBERTa + Pipeline)':<45} {aware_exact_acc:>7.2f}% {aware_inclusive_acc:>9.2f}% {aware_f1_avg:>9.2f}% {aware_rouge_avg:>8.2f}% {aware_rejection_rate:>7.2f}%")
print("=" * 95)
