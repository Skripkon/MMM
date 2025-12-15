from typing import Any

LLAVA_MODEL_ID: str = "llava-hf/llava-1.5-7b-hf"
QWEN_2_5_VL_MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE: str = "mps"

PREDICT_PRICE_PROMPT_TEMPLATE: str = """
You are given the product and your task is to predict the price of the product.

Title
{title}

Store
{store}

Description
{description}

Rating
{average_rating}

Main category
{main_category}

Details
{details}

Return the predicted price of the product in dollars and NOTHING ELSE.
""".strip()


GENERATE_TITLE_PROMPT_TEMPLATE: str = """
You are given the product and your task is to predict the title of the product.

Description
{description}

Store
{store}

Rating
{average_rating}

Price
{price}

Main category
{main_category}

Details
{details}

Return the predicted title of the product and NOTHING ELSE.
""".strip()


GENERATE_DESCRIPTION_PROMPT_TEMPLATE: str = """
You are given the product '{title}' from a category '{main_category}' and your task is to generate a description of the product. Return the description with approximately {n_words} words and NOTHING ELSE.
""".strip()

CHAT_TEMPLATE: list[dict[str, Any]] = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "###USER_PROMPT###"},
        {"type": "image"},
    ]
}]


LLM_AS_A_JUDGE_MODEL_ID: str = "Qwen/Qwen3-0.6B"
LLM_AS_A_JUDGE_PROMPT_TEMPLATE: str = """
Original {task}: '{original}'
Predicted {task}: '{predicted}'

How good is the predicted {task}? Rate it from 0 (predicted is completely wrong) to 10 (predicted is exactly the same as original). Return ONLY the score and nothing else.
""".strip()