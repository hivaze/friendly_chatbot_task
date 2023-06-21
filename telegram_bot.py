import asyncio
import os

import torch
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.utils.executor import Executor
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# from instruct_pipeline import InstructionTextGenerationPipeline

peft_model_path = f"models/{os.environ['MODEL']}"
config = PeftConfig.from_pretrained(peft_model_path)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, padding_side="left")
llm_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                             device_map="auto",
                                             load_in_8bit=True,
                                             torch_dtype=torch.bfloat16)
# model.cuda()

llm_model = PeftModel.from_pretrained(llm_model, peft_model_path)
llm_model.eval()

print('LLM model loaded')

emotional_model = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier", device_map='auto')
print('Emotional clf model loaded')

INTRO_PROMPT = "You are 20 years old, your name is Arthur, you are a friendly young man, designer from San Francisco. " \
               "You are a kind and empathetic interlocutor. You are talking to a new person. Below is an " \
               "instruction that describes a task."

INSTRUCTION_PROMPT_GENERAL = "Get to know a person by chit-chatting and providing long answers and questions. Try not " \
                             "to end the dialogue as long as possible. Complete a phrase, acting like an interlocutor."

INSTRUCTION_PROMPT_FLIRT = "Flirt with a person by chit-chatting and asking personal questions. " \
                           "Complete a phrase, acting like an interlocutor."

INSTRUCTION_PROMPT_FIX = "Try to understand why a person experiences negative emotions. And politely establish a good " \
                         "relationship with him. Complete a phrase, acting like an interlocutor."

GENERATION_PARAMS = {
    # 'length_penalty': -10.0,  # penalize long sentences
    # 'length_penalty': 100.0,  # penalize short sentences
    'repetition_penalty': 1.1,  # to not repeat the prompt in straight way
    'top_p': 0.9,
    'top_k': 100,
    'num_beams': 1,
    'eos_token_id': 187,  # until \n
    'forced_eos_token_id': 187,  # until \n
    'temperature': 0.7,
    'max_new_tokens': 100,  # strong length limit
    'use_cache': True,
    'remove_invalid_values': True
}

PROMPT_TEMPLATE = "{intro}\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"

bot = Bot(token=os.environ['TG_BOT_TOKEN'])
dp = Dispatcher(bot, storage=MemoryStorage())


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

        emotional_results = emotional_model(message.text, top_k=7)
        emotional_results = {elem['label']: elem['score'] for elem in emotional_results}  # dict

        # TODO: maybe exponential mean here for emotional scores
        positive_score = emotional_results['joy'] + emotional_results['surprise']
        negative_score = emotional_results['disgust'] + emotional_results['fear'] + emotional_results['anger']

        min_new_tokens = None

        if negative_score > 0.3:
            min_new_tokens = 15
            prompt = PROMPT_TEMPLATE.format(intro=INTRO_PROMPT,
                                            instruction=INSTRUCTION_PROMPT_FIX,
                                            response=text_history)
        elif positive_score < 0.5:
            min_new_tokens = 3
            prompt = PROMPT_TEMPLATE.format(intro=INTRO_PROMPT,
                                            instruction=INSTRUCTION_PROMPT_GENERAL,
                                            response=text_history)
        else:
            min_new_tokens = 10
            prompt = PROMPT_TEMPLATE.format(intro=INTRO_PROMPT,
                                            instruction=INSTRUCTION_PROMPT_FLIRT,
                                            response=text_history)

        print(f'Positive score: {positive_score}, negative_score: {negative_score}, chosen prompt: {prompt}')

        await message.chat.do("typing")

        input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

        answer = tokenizer.batch_decode(llm_model.generate(inputs=input_ids,
                                                           min_new_tokens=min_new_tokens,
                                                           **GENERATION_PARAMS))[0]  # one sequence at time
        answer = answer[len(prompt):].strip()  # remove prompt

        if answer.endswith("### End"):  # dumb way... must ne like in instruct_pipeline.py here
            answer = answer[:-7].strip()

        await message.reply(answer)

        print('Sending answer:', answer)

        history = history + [f'You: {answer}']

        await state.update_data({'history': history})
    except Exception as e:
        await message.reply(f"Sorry, an error '{e}' occurred :( Try again or contact admins...")


if __name__ == '__main__':
    executor = Executor(dispatcher=dp)
    executor.start_polling(dp)
