from dataclasses import dataclass
import os

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import asyncio
import aiohttp
from tqdm.asyncio import tqdm as tqdm_asyncio

import json
from typing import Any, Literal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from consts import (
    LLAVA_MODEL_ID, QWEN_2_5_VL_MODEL_ID, DEVICE, PREDICT_PRICE_PROMPT_TEMPLATE,
    GENERATE_TITLE_PROMPT_TEMPLATE, GENERATE_DESCRIPTION_PROMPT_TEMPLATE, CHAT_TEMPLATE
)
from tqdm import tqdm

import ast
from copy import deepcopy

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


@dataclass
class ModelsLoader:

    llava_model: LlavaForConditionalGeneration | None = None
    llava_processor: AutoProcessor | None = None
    qwen_2_5_vl_model: Qwen2_5_VLForConditionalGeneration | None = None
    qwen_2_5_vl_processor: AutoProcessor | None = None

    def load_model_and_processor(self, model_id: str):
        if model_id == LLAVA_MODEL_ID:
            if self.llava_model is not None:
                return self.llava_model, self.llava_processor

            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.float16
            ).to(DEVICE)
            self.llava_model.eval()
            self.llava_processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            return self.llava_model, self.llava_processor

        elif model_id == QWEN_2_5_VL_MODEL_ID:
            if self.qwen_2_5_vl_model is not None:
                return self.qwen_2_5_vl_model, self.qwen_2_5_vl_processor

            self.qwen_2_5_vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.float16
            ).to(DEVICE)
            self.qwen_2_5_vl_model.eval()
            self.qwen_2_5_vl_processor = AutoProcessor.from_pretrained(model_id, max_pixels=256*28*28, use_fast=True, padding_side='left')
            return self.qwen_2_5_vl_model, self.qwen_2_5_vl_processor
        else:
            raise ValueError("Model not supported")

def batch_generate(
    images: list[Image.Image] | None,
    chats: list[list[dict[str, Any]]],
    processor: AutoProcessor,
    model: LlavaForConditionalGeneration | Qwen2_5_VLForConditionalGeneration,
    max_new_tokens: int = 100, do_sample: bool = False, temperature: float = 0.0, top_k: int = 50
    ):

    chat_template_prompts = processor.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)

    if images is not None:
        inputs = processor(
            images=images,
            text=chat_template_prompts,
            return_tensors='pt',
            padding=len(chats) > 1
        ).to(DEVICE)
    else:
        inputs = processor(
            text=chat_template_prompts,
            return_tensors='pt',
            padding=len(chats) > 1
        ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k
        )

    decoded_output = processor.batch_decode([
        output[i][len(inputs["input_ids"][i]):] for i in range(len(output))
    ], skip_special_tokens=True)

    return decoded_output


def predict(
    models_loader: ModelsLoader, model_id: str, benchmark: pd.DataFrame, inference_batch_size: int = 1,
    task: Literal["price", "title", "description"] = "price", use_images: bool = True, do_sample: bool = False, max_new_tokens: int = 100,
    temperature: float = 0.0, top_k: int = 50
    ):

    model, processor = models_loader.load_model_and_processor(model_id)
    model_id_name = model_id.split("/")[-1]
    if not use_images:
        filepath = f"data/results/{model_id_name}_{task}_no_images_predictions.json"
    else:
        filepath = f"data/results/{model_id_name}_{task}_predictions.json"

    cached_results = load_results(filepath)
    logger.info(f"Loaded {len(cached_results)} cached results for {model_id_name}")

    current_idx = 0
    logger.info(f"Predicting {task} for {len(benchmark) - len(cached_results)} samples")
    while current_idx < len(benchmark):
        uncached_samples = []
        
        while current_idx < len(benchmark) and len(uncached_samples) < inference_batch_size:
            row = benchmark.iloc[current_idx]
            parent_asin = row["parent_asin"]
            if parent_asin not in cached_results:
                uncached_samples.append(current_idx)
            current_idx += 1
        
        if len(uncached_samples) == 0:
            break
        
        samples = benchmark.iloc[uncached_samples]

        images = [Image.open(f"data/images/{sample['parent_asin']}.jpg") for sample in samples.to_dict(orient="records")]

        instructions = []
        for _, sample in samples.iterrows():
            details_str = ""
            for k, v in ast.literal_eval(sample["details"]).items():
                details_str += f"{k}: {v}\n"

            if task == "price":
                prompt = PREDICT_PRICE_PROMPT_TEMPLATE.format(
                    title=sample["title"],
                    description=sample["description"],
                    store=sample["store"],
                    average_rating=sample["average_rating"],
                    main_category=sample["main_category"],
                    details=details_str
                )
            elif task == "title":
                prompt = GENERATE_TITLE_PROMPT_TEMPLATE.format(
                    description=sample["description"],
                    store=sample["store"],
                    average_rating=sample["average_rating"],
                    price=sample["price"],
                    main_category=sample["main_category"],
                    details=details_str
                )
            elif task == "description":
                prompt = GENERATE_DESCRIPTION_PROMPT_TEMPLATE.format(
                    title=sample["title"],
                    main_category=sample["main_category"],
                    n_words=len(sample["description"].split())
                )
            else:
                raise ValueError(f"Task {task} not supported")

            instructions.append(prompt)

        chats = [deepcopy(CHAT_TEMPLATE) for _ in range(len(instructions))]

        for chat, instruction in zip(chats, instructions):
            chat[0]["content"][0]["text"] = instruction

        if use_images:
            outputs = batch_generate(images, chats, processor, model, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k)
        else:
            outputs = batch_generate(None, chats, processor, model, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k)

        # save results by `parent_asin`
        for parent_asin, output in zip(samples["parent_asin"], outputs):
            cached_results[parent_asin] = output

        dump_results(cached_results, filepath)

    logger.info(f"Saved {len(cached_results)} predictions for {model_id_name} to {filepath}")
    return cached_results


def dump_results(results: dict[str, str], filepath: str):
    with open(filepath, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


def load_results(filepath: str):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        return {}

def select_different_products_by_category(dataframe: pd.DataFrame, category: str, n: int) -> pd.DataFrame:
    """
    Selects up to n diverse products from the given category, maximizing diversity using cosine distance between embeddings.
    """
    category_products = dataframe[dataframe['filename'] == category]
    category_products = category_products.dropna(subset=['price', 'store', 'description'], how='any', inplace=False)

    if len(category_products) <= n:
        return category_products

    embeddings = np.stack(category_products['embeddings'].values)
    selected_indices = [0]
    selected_embeddings = [embeddings[0]]
    while len(selected_indices) < n:
        dists = cosine_distances(embeddings, np.vstack(selected_embeddings))
        min_dist = dists.min(axis=1)
        for idx in np.argsort(-min_dist):
            if idx not in selected_indices:
                selected_indices.append(idx)
                selected_embeddings.append(embeddings[idx])
                break
    return category_products.iloc[selected_indices].reset_index(drop=True)


async def _download_image(session, url, path, sem):
    async with sem:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    image_data = await resp.read()
                    with open(path, "wb") as handler:
                        handler.write(image_data)
        except Exception:
            pass


def download_images(dataframe: pd.DataFrame, image_column: str, save_dir: str = "data", max_concurrent: int = 64):
    """
    Downloads images from the given column and saves them to the local directory asynchronously using semaphores.
    """
    os.makedirs(save_dir, exist_ok=True)

    async def run():
        sem = asyncio.Semaphore(max_concurrent)
        tasks = []
        async with aiohttp.ClientSession() as session:
            for _, row in dataframe.iterrows():
                image_url = row[image_column]
                if not image_url:
                    continue
                parent_asin = row["parent_asin"]
                path = f"{save_dir}/{parent_asin}.jpg"
                if os.path.exists(path):
                    continue
                tasks.append(_download_image(session, image_url, path, sem))
            for f in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Downloading images"):
                await f

    try:
        loop = asyncio.get_running_loop()
        return asyncio.create_task(run())
    except RuntimeError:
        asyncio.run(run())


def parse_price(price_str: str) -> float | None:
    """Parses the price string and returns the price as a float. If parsing fails, returns None."""
    try:
        price: float = float(price_str.replace("$", ""))
        return price
    except ValueError:
        return None


def compute_metrics(ground_truth: list[float], predictions: list[float]) -> dict[str, float]:
    """Computes the metrics for the predictions."""
    return {
        "MAPE": round(100 * mean_absolute_percentage_error(ground_truth, predictions), 2),
        "MAE": round(mean_absolute_error(ground_truth, predictions), 2),
        "MSE": round(mean_squared_error(ground_truth, predictions), 2)
    }


def calculate_bleu_score(ground_truth: list[str], predictions: list[str]) -> float:
    """Calculates the BLEU score for the predictions."""
    if len(ground_truth) != len(predictions):
        raise ValueError("ground_truth and predictions must have the same length")
    
    smoothing = SmoothingFunction().method1
    scores = []
    
    for ref, pred in zip(ground_truth, predictions):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        scores.append(score)
    
    return round(float(np.mean(scores)), 3)



def calculate_rouge_score(ground_truth: list[str], predictions: list[str]) -> float:
    """Calculates the ROUGE score for the predictions."""
    if len(ground_truth) != len(predictions):
        raise ValueError("ground_truth and predictions must have the same length")
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    
    for ref, pred in zip(ground_truth, predictions):
        score = scorer.score(ref, pred)
        scores.append(score['rougeL'].fmeasure)
    
    return round(float(np.mean(scores)), 3)


def calculate_blue_and_rouge_scores(
        benchmark: pd.DataFrame,
        task: Literal["title", "description"],
        use_images: bool = True,
        models_to_eval: list[str] = [LLAVA_MODEL_ID, QWEN_2_5_VL_MODEL_ID]
        ) -> pd.DataFrame:
    """Calculates the BLEU and ROUGE-L scores for the predictions."""
    ground_truth = benchmark[task].tolist()
    predictions = []
    for model_id in models_to_eval:
        if use_images:
            fp = f"data/results/{model_id.split('/')[-1]}_{task}_predictions.json"
        else:
            fp = f"data/results/{model_id.split('/')[-1]}_{task}_no_images_predictions.json"
        predictions.append(list(load_results(fp).values()))

    assert len(predictions) == len(models_to_eval), "Predictions and models_to_eval have different sizes"

    bleu_scores = [calculate_bleu_score(ground_truth, predictions[i]) for i in range(len(models_to_eval))]
    rouge_scores = [calculate_rouge_score(ground_truth, predictions[i]) for i in range(len(models_to_eval))]

    results_table = pd.DataFrame({
        "Model": models_to_eval,
        "BLEU": bleu_scores,
        "ROUGE-L": rouge_scores
    })
    results_table.set_index("Model", inplace=True)
    return results_table
