import asyncio
import logging
import re
import sys
from os import getenv

import torch
from accelerate import Accelerator
from ai_dataset import AiDataset, DatasetType
from ai_model import AiModel, ModelType
from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
import dotenv

from dotenv import load_dotenv

load_dotenv() 

TOKEN = getenv("BOT_TOKEN")
print(TOKEN)

MODEL_NAME = "microsoft/codebert-base"
MODEL_TYPE = ModelType.ROBERTA
DATASET_TYPE = DatasetType.ROBERTA

AI_MODEL = None
AI_DATASET = None
checkpoint_path = "./src/notebooks/checkpoints_path/detect_ai_model_best.pth.tar"

dp = Dispatcher()

ACCELERATOR = Accelerator(
    gradient_accumulation_steps=1,
)


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


@dp.message()
async def detector_handler(message: Message) -> None:
    tensor_keys = [
        "input_ids",
        "attention_mask",
    ]
    code = extract_python_code(message.text)
    data = AI_DATASET.tokenize_function({"text": code})
    for key in tensor_keys:
        data[key] = torch.tensor([data[key]], dtype=torch.int64)

    resp = torch.sigmoid(AI_MODEL(**data)[0])
    print(resp.item())

    await message.answer(f"{resp.item()}")


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    load_codebert()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
