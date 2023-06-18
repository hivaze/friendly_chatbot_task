# AI Friend test task solution

Author: Sergey Bratchikov (hivaze)

## General approach

The main idea of the approach: we use prompts, we train a model, the license of which allows commercial use on high-quality empathic dialogues.

Note WIP notes are in raw_task_notes.md

### Used datasets

For training on relatively high-quality dialogues, 2 datasets were used:
- [empathetic_dialogues](https://huggingface.co/datasets/empathetic_dialogues)
- [daily_dialog](https://huggingface.co/datasets/daily_dialog)

The full pipeline of their pre-processing and gluing can be seen in the ai_friend_model_training notebook. In short: we remove the negative content that is already marked in them, leaving the neutral and positive in the main. After that, we train the model in the format of instructions, providing it with each dialogue in its entirety. \
The instruction chosen was: "You are a kind and empathetic interlocutor. Your secret goal is to try to attract a person to you and understand his needs, using non-violent communication. You are chit-chatting to a person."

Dataset empathetic_dialogues came from "Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset" (2019)
DailyDialog came from "DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset" (2017)

## Models

The databricks/dolly-v2-3b model was chosen, because according to the OpenLLM benchmark, this is almost the best model with a small size. Moreover, it has permission for commercial use. In addition to it, the 7b version was also trained, all the weights of LORA adapters are in models. Training process in notebook.

Finetuning was done with PEFT. And instructions setup.

Final metrics:

train steps - train loss - val loss \
1200    1.164300 	1.217549    dolly-v2-3b-lora \
500 	1.197700 	1.222749    dolly-v2-7b-lora

## Ice breaker

Main idea here was a CLM model with good dataset... and prompts.. \
But the best solution here seems to be PPOTrainer with a reward model from trl. I didn't do it... \
Ot we can calculate cumulative emotion score and do threshold...

## How to start

No docker for now (its easy, but too late...) Just `python3 telegram_bot.py` after requirements... \
And you need to setup 2 env vars `MODEL` name from models and `TG_BOT_TOKEN`.

## How to measure quality

FastChat and Vicuna propose to use GPT4 and ChatGPT, FastChat even have some code for it, but i didnt use any of it... \
BTW deploying of llms with fastchat can be done very quickly...

## Other stuff

Maybe some other ideas can be found in code and in raw_task_notes, but too late to write down it here...