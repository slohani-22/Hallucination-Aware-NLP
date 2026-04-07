import torch
from transformers import pipeline

DEVICE = 0 if torch.cuda.is_available() else -1
print("NLI using:", "GPU" if DEVICE == 0 else "CPU")

nli_model = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    device=DEVICE
)

def count_contradictions(paragraphs):
    paragraphs = paragraphs[:2]   # 🔥 reduce noise
    contradictions = 0

    for i in range(len(paragraphs)):
        for j in range(i + 1, len(paragraphs)):

            result = nli_model(
                {
                    "text": paragraphs[i],
                    "text_pair": paragraphs[j]
                },
                truncation=True
            )

            if isinstance(result, list):
                label = result[0]["label"].upper()
                score = result[0]["score"]
            else:
                label = result["label"].upper()
                score = result["score"]

            # 🔥 only strong contradictions count
            if "CONTRADICTION" in label and score >= 0.90:
                contradictions += 1

    return contradictions