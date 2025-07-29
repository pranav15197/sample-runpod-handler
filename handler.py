import runpod
import os
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import os
from huggingface_hub import login as hf_login


load_dotenv()


def handler(event):
    token = os.getenv("HUGGINGFACE_TOKEN")
    hf_login(token=token)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Variables (change if needed)
    BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    DATA_FILE = "sample.jsonl"  # your 1-row file
    OUTPUT_DIR = "./poc-output"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    dataset = load_dataset("json", data_files=DATA_FILE)["train"]
    dataset = dataset.select(range(1))  # use only 1 row

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def format(example):
        text = example["input"] + tokenizer.eos_token + example["output"] + tokenizer.eos_token
        enc = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc
    tokenized = dataset.map(format)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=1,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer
    )

    trainer.train()
    # model.save_pretrained(OUTPUT_DIR)
    # tokenizer.save_pretrained(OUTPUT_DIR)




# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })