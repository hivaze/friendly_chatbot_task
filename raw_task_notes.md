# Some wip notes for the task:

## Any helpful datasets

https://huggingface.co/datasets/AlekseyKorshuk/erotic-books

https://huggingface.co/datasets/Ericwang/promptSentiment

https://huggingface.co/datasets/EleutherAI/lambada_openai

https://huggingface.co/datasets/michellejieli/friends_dataset

https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered

https://huggingface.co/datasets/empathetic_dialogues

https://huggingface.co/datasets/RUCAIBox/Open-Dialogue (compilation)

https://huggingface.co/datasets/allenai/prosocial-dialog

https://huggingface.co/datasets/alespalla/chatbot_instruction_prompts

https://huggingface.co/datasets/zetavg/ShareGPT-Processed


## Potenrialy good Models

According to some metrics from papers (and gpt4all benchmark, open_llm_leaderboard, etc):

OpenLLM leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

I can immediately make a remark that the models trained on ShareGPT will most likely have a bias towards ChatGPT and say that they are ethical and just do AI.

https://huggingface.co/nomic-ai/gpt4all-j
https://huggingface.co/nomic-ai/gpt4all-j-lora
https://huggingface.co/AlekseyKorshuk/results-gpt-j-lit-erotic?text=My+name+is+Julien+and+I+like+to
https://huggingface.co/vicgalle/gpt-j-6B-alpaca-gpt4
https://huggingface.co/microsoft/GODEL-v1_1-large-seq2seq - seq2seq goal oriented model (220M, 770M, 175B), but somewhere wheris gpt-j version

I remember that somewhere on HF there was a GPT-J model tuned on fanfiction and erotic, but I forgot its name...

1.3b:
https://huggingface.co/facebook/galactica-1.3b
https://huggingface.co/PygmalionAI/pygmalion-1.3b

3b:
https://huggingface.co/databricks/dolly-v2-3b
https://huggingface.co/lmsys/fastchat-t5-3b-v1.0
https://huggingface.co/Fredithefish/ScarletPajama-3B-HF
https://huggingface.co/hakurei/lit-6B
https://huggingface.co/facebook/blenderbot-3B?text=Hey+my+name+is+Mariama%21+How+are+you%3F

6b:
https://huggingface.co/hakurei/lit-6B

7b:
https://huggingface.co/mosaicml/mpt-7b
https://huggingface.co/tiiuae/falcon-7b
https://huggingface.co/decapoda-research/llama-7b-hf
https://huggingface.co/EleutherAI/gpt-j-6b
https://huggingface.co/chainyo/alpaca-lora-7b
https://huggingface.co/databricks/dolly-v2-7b
https://huggingface.co/eachadea/vicuna-7b-1.1

## Finetuning with lora

https://github.com/tloen/alpaca-lora/blob/main/finetune.py
https://huggingface.co/TheBloke/gpt4-alpaca-lora-30b-HF
https://github.com/Reason-Wang/flan-alpaca-lora/blob/main/train.py

## Full finetuning

https://github.com/hpcaitech/ColossalAI/blob/78509124d32b63b7fc36f6508e0576a326d51422/examples/language/opt/run_clm.py

## Other stuff

https://github.com/lm-sys/FastChat/ (An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and FastChat-T5. )
https://github.com/EleutherAI/pythia
https://github.com/microsoft/GODEL/tree/main
https://gpt4all.io/index.html - benchmarks table for many llms
https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt2-sentiment_peft.py (sentiment trl PPOTrainer)

## Judging

Use ChatGPT or GPT-4 for eval. More ifo here: https://lmsys.org/blog/2023-03-30-vicuna/
Stronger models than the basic? (2.8b -> 7b, 7b -> 13b, etc)

## Googd emotion detectors

https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
https://huggingface.co/michellejieli/emotion_text_classifier
https://huggingface.co/bdotloh/distilbert-base-uncased-empathetic-dialogues-context - empha

## Quantinization 4bit

Easy way - https://huggingface.co/blog/4bit-transformers-bitsandbytes
GPTQ - https://github.com/IST-DASLab/gptq

## Ice-Breaking

A set of prompts depending on the state of the dialogue (flirty/non flirty). \
You can determine the positivity of the dialogue using additional models. \
Or you can use PPOTrainer from trl to directly train model to follow emotional line

## Serving

Good and fast serving can be done with FastChat
Possible movements for better control: DeepSpeed Inference, Triton Inf. Server, ONNX, GPT-Q, ggml...
