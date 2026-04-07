from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0
)

def generate_answer(question, paragraphs):

    context = " ".join(paragraphs[:2])

    prompt = f"""
    Answer the question based on the context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    result = generator(
        prompt,
        max_length=200,
        temperature=0.7,
        do_sample=True
    )

    output = result[0]["generated_text"]

    answer = output.split("Answer:")[-1].strip()

    return answer