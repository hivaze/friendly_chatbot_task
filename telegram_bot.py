import asyncio
import logging
import os

import torch
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.utils.executor import Executor
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from instruct_pipeline import InstructionTextGenerationPipeline

peft_model_path = f"models/{os.environ['MODEL']}"
config = PeftConfig.from_pretrained(peft_model_path)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                             device_map="auto",
                                             torch_dtype=torch.float16)
model.cuda()

model = PeftModel.from_pretrained(model, peft_model_path)
model.eval()

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

INSTRUCTION_NEW = "You are 20 years old, your name is Arthur, you are a human, a designer from San Francisco. You are a " \
              "kind and empathetic interlocutor. Your secret goal is to try to attract a person to you and understand " \
              "his needs, using non-violent communication. You are chit-chatting to a new person."

INSTRUCTION_KNOWN = "You are 20 years old, your name is Arthur, you are a human, a designer from San Francisco. You are a " \
              "kind and empathetic interlocutor. Your secret goal is to make a person fall in love with you through " \
                    "flirting. You are chit-chatting to a person."

GENERATION_PARAMS = {
    'length_penalty': -20.0,  # penalize long sentences
    'repetition_penalty': 2.0,  # to not repeat the prompt in straight way
    'top_p': 0.7,
    'top_k': 50,
    'num_beams': 3,
    'forced_eos_token_id': 50277,
    'temperature': 0.6,
    'max_new_tokens': 100,  # strong length limit
    'use_cache': True,
    'remove_invalid_values': True
}

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42

PROMPT_NO_INPUT_FORMAT = """{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

logger = logging.getLogger(__name__)

logger.info(f"General prompt for model: {PROMPT_NO_INPUT_FORMAT}")

bot = Bot(token=os.environ['TG_BOT_TOKEN'])
dp = Dispatcher(bot, storage=MemoryStorage())


class TypingBlock(object):

    def __init__(self, chat: types.Chat):
        self.chat = chat
        self.typing_task = None

    async def __aenter__(self):

        async def typing_cycle():
            try:
                while True:
                    await self.chat.do("typing")
                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                pass

        self.typing_task = asyncio.create_task(typing_cycle())

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.typing_task:
            self.typing_task.cancel()


@dp.message_handler(commands=["start", "reset"], state='*')
async def welcome_user(message: types.Message, state: FSMContext, *args, **kwargs):
    await state.reset_data()
    await message.answer(text='Hi, just type some text, like i am your friend or like you want to meet me.')


@dp.message_handler(state=None)
async def communication_answer(message: types.Message, state: FSMContext, is_image=False, *args, **kwargs):
    try:
        current_data = await state.get_data()

        messages_history = current_data.get('history') or []
        history = messages_history + [f"Person: {message.text}"]

        text_history = "\n".join(history) + "\nYou:"

        if len(history) >= 20:  # dumbiest way, but.... here we need some cumulative score from emotions model and threshold I think...
            prompt = PROMPT_NO_INPUT_FORMAT.format(instruction=INSTRUCTION_NEW, response=text_history)
        else:
            prompt = PROMPT_NO_INPUT_FORMAT.format(instruction=INSTRUCTION_KNOWN, response=text_history)

        input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

        answer = tokenizer.batch_decode(model.generate(input_ids, **GENERATION_PARAMS))[0]
        answer = answer[len(prompt):].strip()  # remove prompt

        if answer.endswith("\n\n### End"):  # dumb way... must ne like in instruct_pipeline.py here
            answer = answer[:11]

        # need to check case with \n\n

        await message.reply(answer)

        history = history + [f'You: {answer}']

        await state.update_data({'history': history})
    except Exception as e:
        await message.reply(f"Sorry, an error '{e}' occurred :( Try again or contact admins...")


if __name__ == '__main__':
    executor = Executor(dispatcher=dp)
    executor.start_polling(dp)
