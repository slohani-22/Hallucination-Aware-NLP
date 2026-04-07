from datasets import load_dataset

def load_data(sample_size=200):
    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split=f"train[:{sample_size}]"
    )
    return dataset

def get_paragraphs(sample):
    sentences = sample["context"]["sentences"]

    paragraphs = []
    for i in range(len(sentences)):
        paragraph = " ".join(sentences[i])
        paragraphs.append(paragraph)

    return paragraphs