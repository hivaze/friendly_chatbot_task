# AI Friend test task solution

Telegram bot (aiogram) with finetuned LLM and emotion classifier \
Author: Sergey Bratchikov (hivaze)

## General approach

The main idea of the approach: we use prompts, we train a model, the license of which allows commercial use on high-quality empathic dialogues.

Note WIP notes are in `raw_task_notes.md`

## Used datasets

For training on relatively high-quality dialogues, 2 datasets were used:
- [empathetic_dialogues](https://huggingface.co/datasets/empathetic_dialogues)
- [daily_dialog](https://huggingface.co/datasets/daily_dialog)

Merged version:
- [hivaze/emphatical_daily_dialogues](https://huggingface.co/datasets/hivaze/emphatical_daily_dialogues)

Overall 20k dialogues of different lengths in train, 2k in validation. See dataset card for more info.

The full pipeline of their pre-processing and gluing can be seen in the emphatical_dialogues_dataset notebook. \
In short: we remove the negative content that is already marked in them, leaving the neutral and positive in the main. After that, we train the model in the format of instructions, providing it with each dialogue in its entirety.

Prompts while training:
```
system_prompt = "You are a kind and empathetic interlocutor. You are talking to a person. Below is an instruction that describes a task. Write a response that appropriately completes the request."
instruction_prompt = "You try to chit-chat. Complete a phrase, acting like an interlocutor."
```

Dataset empathetic_dialogues came from "Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset" (2019)
DailyDialog came from "DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset" (2017)

## LLM Models

The `databricks/dolly-v2-3b` model was chosen, because according to the OpenLLM benchmark, this is almost the best model with a small size. And also this model was trained to follow the instructions, unlike the usual llama (most of other instuction-models use ShareGPT, which does not suit us, we do not create a ChatGPT clone with its bias). Following the instructions potentially means that you can control the model's responses in a dialog. \
Moreover, dolly has permission for commercial use. In addition to it, the 7b version was also trained, all the weights of LORA adapters are in models folder. \
Training process in notebook `lora_finetune.md`.

Other models and links can be found in `raw_task_notes.md`.

Finetuning was done with PEFT library and HF Trainer, params are in notebook. For 3b - 2 epochs, 7b - 1 epoch.

Final metrics:

train steps - train loss - val loss \
1206    1.363 	1.34    dolly-v2-3b-lora \
603 	1.235 	1.324    dolly-v2-7b-lora

Adapters weights are in models folder. Also on my hf account: https://huggingface.co/hivaze

Generation params:
```
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
    'min_new_tokens': 3, # or 15 or 10, depends on prompt and user emotion
    'use_cache': True,
    'remove_invalid_values': True
}
```

## Ice breaker

Main idea here was a CLM model with good dataset... and prompts.. \

For this i use scoring of users messages from pretrained model `michellejieli/emotion_text_classifier`

Two type of scores: positive and negative. So there are 3 instruction prompt types according to a task.

```
INSTRUCTION_PROMPT_GENERAL = "Get to know a person by chit-chatting and providing long answers and questions. Try not to end the dialogue as long as possible. Complete a phrase, acting like an interlocutor."
INSTRUCTION_PROMPT_FLIRT = "Flirt with a person by chit-chatting and asking personal questions. Complete a phrase, acting like an interlocutor."
INSTRUCTION_PROMPT_FIX = "Try to understand why a person experiences negative emotions. And politely establish a good relationship with him. Complete a phrase, acting like an interlocutor."
```

To switch them I use 2 thresholds:
```
negative_score > 0.3: INSTRUCTION_PROMPT_FIX \
positive_score > 0.5: INSTRUCTION_PROMPT_FLIRT \
else: INSTRUCTION_PROMPT_GENERAL
```

But the best solution here seems to be `PPOTrainer` with a reward model from trl. I didn't do it...

## How to start

Dockerfile was generated via ChatGPT :)

Build image:
```
docker build -t ai_friend_bot .
```

Run container:
```
docker run --gpus all -e MODEL=model_name -e TG_BOT_TOKEN=my_token ai_friend_bot
```

Note: that you need to setup 2 env vars `MODEL` name from models folder and `TG_BOT_TOKEN`.

Note: if you want cuda, you need to install nvidia-container-toolkit

## How to measure quality

FastChat and Vicuna propose to use GPT4 and ChatGPT, FastChat even have some code for it, but i didnt use any of it... \
BTW deploying of llms with fastchat can be done very quickly...

Another way is to measure perplexity with a stronger model, but it's not very revealing.

## Other stuff

Maybe some other ideas can be found in code and in raw_task_notes, but too late to write down it here...
