import asyncio
import io
import logging
import os
import pickle
import re
import sys
from os import getenv
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import tree_sitter_python as tspython
from accelerate import Accelerator
from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message
from dotenv import load_dotenv
from lime.lime_text import LimeTextExplainer
from tree_sitter import Language, Parser

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from scripts.ai_dataset import AiDataset, DatasetType
from scripts.ai_model import AiModel, ModelType

load_dotenv()

TOKEN = getenv("BOT_TOKEN")

NAME = "codebert"
MODEL_NAME = "microsoft/codebert-base"
MODEL_TYPE = ModelType.ROBERTA
DATASET_TYPE = DatasetType.ROBERTA

# NAME = "deberta_base"
# MODEL_NAME = "microsoft/deberta-v3-base"
# MODEL_TYPE = ModelType.DEBERTA
# DATASET_TYPE = DatasetType.DEBERTA

# NAME = "deberta_small"
# MODEL_NAME = "microsoft/deberta-v3-xsmall"
# MODEL_TYPE = ModelType.DEBERTA
# DATASET_TYPE = DatasetType.DEBERTA


TREE_MODEL = None
with open("./src/notebooks/trees_path/random_forest_classifier_model.pkl", "rb") as f:
    TREE_MODEL = pickle.load(f)

AI_MODEL = None
AI_DATASET = None
checkpoint_path = f"./src/notebooks/checkpoints_path/{NAME}_best.pth.tar"

dp = Dispatcher()

ACCELERATOR = Accelerator(
    gradient_accumulation_steps=1,
)

PY_LANGUAGE = Language(tspython.language())
from collections import Counter

parser = Parser(PY_LANGUAGE)
node_types = set()
with open("./data/ast/node_types.txt", "r", encoding="utf-8") as f:
    node_types = [x.strip() for x in f.readlines()]


def walk_tree(node, types):
    types.append(node.type)
    for child in node.children:
        walk_tree(child, types)


def code_to_feature_vector(code):
    code = code.encode("utf-8")
    tree = parser.parse(code)
    types = []
    walk_tree(tree.root_node, types)
    counts = Counter(types)
    feature_vector = [counts.get(typ, 0) for typ in node_types]

    return feature_vector


def extract_python_code(res: str) -> str:
    try:
        return (
            re.findall(r"```python(.*?)```", res, re.DOTALL)[0]
            .encode("utf-8", errors="ignore")
            .decode("utf-8")
        )
    except:
        return res


def load_codebert():
    global AI_MODEL, AI_DATASET
    model_test = AiModel(ACCELERATOR.device, MODEL_NAME, MODEL_TYPE)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_test.load_state_dict(ckpt["state_dict"])
    AI_MODEL = model_test
    AI_DATASET = AiDataset(MODEL_NAME, DATASET_TYPE)
    return


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!")


def predict_proba_lime(text):
    data = AI_DATASET.tokenize_function({"text": text}, padding=True)

    tensor_keys = ["input_ids", "attention_mask"]
    for key in tensor_keys:
        data[key] = torch.tensor(data[key], dtype=torch.int64)

    with torch.no_grad():
        logits, _ = AI_MODEL(**data)
        logits = logits.reshape(-1)
        predictions = torch.sigmoid(logits)

    predictions_np = predictions.cpu().numpy()
    return np.array([[1 - x, x] for x in predictions_np])


def predict_proba_rfc(batch_texts):
    feature_vectors = []
    for text in batch_texts:
        feature_vector = code_to_feature_vector(text)
        feature_vectors.append(feature_vector)

    X_arr = np.array(feature_vectors)

    probas = TREE_MODEL.predict_proba(X_arr)
    return probas


@dp.message()
async def detector_handler(message: Message) -> None:
    code = extract_python_code(message.text)
    data = AI_DATASET.tokenize_function({"text": code})

    tensor_keys = ["input_ids", "attention_mask"]
    for key in tensor_keys:
        data[key] = torch.tensor([data[key]], dtype=torch.int64)

    with torch.no_grad():
        start = time()
        resp = torch.sigmoid(AI_MODEL(**data)[0])
        duration_llm = time() - start
        prob = resp.item()

    explainer = LimeTextExplainer(
        class_names=["human", "AI"],
    )

    try:
        exp = explainer.explain_instance(
            code,
            predict_proba_lime,
            num_features=10,
            num_samples=50,
        )

        significant_features = [(f, w) for f, w in exp.as_list()]
        fig = exp.as_pyplot_figure()
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        if significant_features:
            explanation_text = "Top features affecting the prediction:\n"
            for feature, weight in significant_features[:5]:
                explanation_text += f"{weight:.2f}: {feature}\n"
        else:
            explanation_text = "The model decision appears evenly distributed across many small features."

        photo = BufferedInputFile(buf.getvalue(), filename="lime_explanation.png")

        await message.reply_photo(
            photo,
            caption=f"LLM AI-generated probability ({MODEL_NAME}): {prob:.4f}\n\nElapsed time: {duration_llm:.4f}s\n\n{explanation_text}",
            parse_mode="Markdown",
        )

    except Exception as e:
        logging.error(f"Explanation failed: {str(e)}")
        await message.answer(
            f"LLM AI-generated probability ({MODEL_NAME}): {prob:.4f}\n\nElapsed time: {duration_llm:.4f}s\n\nCould not generate detailed explanation."
        )

    try:
        start = time()
        dt_pred = TREE_MODEL.predict_proba([code_to_feature_vector(code)])[0][1]
        duration_ast = time() - start

        explainerDT = LimeTextExplainer(
            class_names=["human", "AI"],
        )
        exp = explainerDT.explain_instance(
            code,
            predict_proba_rfc,
            num_features=10,
            num_samples=100,
        )

        significant_features = [(f, w) for f, w in exp.as_list()]
        fig = exp.as_pyplot_figure()
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        if significant_features:
            explanation_text = "Top features affecting the prediction:\n"
            for feature, weight in significant_features[:5]:
                explanation_text += f"{weight:.2f}: {feature}\n"
        else:
            explanation_text = "The model decision appears evenly distributed across many small features."

        photo = BufferedInputFile(buf.getvalue(), filename="lime_explanation_dtc.png")

        await message.reply_photo(
            photo,
            caption=f"DT AI-generated probability: {dt_pred:.4f}\n\nElapsed time: {duration_ast:.4f}s\n\n{explanation_text}",
            parse_mode="Markdown",
        )

    except Exception as e:
        logging.error(f"Explanation failed: {str(e)}")
        await message.answer(
            f"DT AI-generated probability: {dt_pred:.4f}\n\nElapsed time: {duration_ast:.4f}s\n\nCould not generate detailed explanation."
        )


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    load_codebert()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
