from datasets import load_dataset

ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
example = hotpotqa["train"][0]

print(example)
