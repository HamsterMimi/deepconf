import json
from datasets import load_dataset

# Load dataset
dataset = load_dataset("MathArena/brumo_2025", split="train")
# MathArena/brumo_2025
# MathArena/aime_2025

# Convert to JSONL
with open("brumo_2025.jsonl", "w", encoding="utf-8") as f:
    for example in dataset:
        entry = {
            "question": example["problem"],
            "answer": str(example["answer"])
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Converted {len(dataset)} examples to brumo_2025")