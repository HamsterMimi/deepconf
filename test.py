from deepconf import DeepThinkLLM
import os
from transformers import AutoTokenizer
os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
# Initialize model

if __name__ == "__main__":
    model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    deep_llm = DeepThinkLLM(model=model_name,
                            gpu_memory_utilization=0.8,
                            max_model_len=2048)
    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    # Prepare prompt
    question = "What is the square root of 144?"

    messages = [
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Run offline mode with multiple voting
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="offline",
        budget=64,
        compute_multiple_voting=True,
    )

    # Evaluate results
    for method, method_result in result.voting_results.items():
        if method_result and method_result.get('answer'):
            print(f"{method}: {method_result['answer']}")
