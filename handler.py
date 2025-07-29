import os, tempfile, requests, json, runpod, shutil
from dotenv import load_dotenv
import torch, gc


import boto3
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from huggingface_hub import login as hf_login


def free_gpu():
    """Release every scrap of GPU RAM the worker may still hold."""
    gc.collect()              # 1. run Python garbage‑collector
    torch.cuda.empty_cache()  # 2. tell PyTorch to return cached blocks
    torch.cuda.ipc_collect()  # 3. clear inter‑process handles (safety)

load_dotenv()
s3 = boto3.client("s3")                     # needs AWS creds in env

# ---------- tiny helpers -------------------------------------------------- #
def download_to_tmp(url, suffix=".jsonl"):
    """Stream a file from `url` into a secure tmp path and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f, requests.get(url, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return path

def upload_dir(local_dir: str, bucket: str, key_prefix: str):
    """Upload every file under `local_dir` to s3://bucket/key_prefix/…"""
    for root, _, files in os.walk(local_dir):
        for name in files:
            full = os.path.join(root, name)
            rel  = os.path.relpath(full, local_dir)
            s3.upload_file(full, bucket, f"{key_prefix}/{rel}")

# ---------- main entry‑point ---------------------------------------------- #
def handler(event):
    """
    Expected JSON payload (event['input']):
    {
      "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
      "data_url":   "https://…/train.jsonl",
      "bucket":     "my‑lora‑bucket",
      "prefix":     "experiments",          # will produce experiments/<name>/*
      "adapter_name": "my_first_lora",
      "epochs": 3
    }
    """
    params = event.get("input", {})
    # 1. parse parameters with sane defaults -------------------------------
    base_model  = params.get("base_model",
                   "mistralai/Mistral-7B-Instruct-v0.2")
    data_url    = params["data_url"]        # mandatory
    bucket      = params["bucket"]          # mandatory
    prefix      = params.get("prefix", "adapters")
    adapter_nm  = params.get("adapter_name", "lora")
    epochs      = int(params.get("epochs", 3))

    # 2. one‑time setup -----------------------------------------------------
    hf_login(os.getenv("HUGGINGFACE_TOKEN"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 3. fetch the training file ------------------------------------------
    data_file = download_to_tmp(data_url, ".jsonl")

    # 4. build tokenizer & model ------------------------------------------
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        device_map="auto",
    )
    model.config.pad_token_id = tok.pad_token_id

    # 5. dataset pipe ------------------------------------------------------
    ds = load_dataset("json", data_files=data_file)["train"]
    def fmt(ex):
        txt = ex["input"] + tok.eos_token + ex["output"] + tok.eos_token
        enc = tok(txt, truncation=True, max_length=1024, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc
    ds = ds.map(fmt)

    # 6. apply LoRA --------------------------------------------------------
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64, lora_alpha=16, lora_dropout=0.05
    )
    model = get_peft_model(model, peft_cfg)

    # 7. training args & train --------------------------------------------
    out_dir = f"/tmp/{adapter_nm}"
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        num_train_epochs=epochs,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=1,
        save_strategy="no",
    )
    Trainer(model=model, args=args, train_dataset=ds, tokenizer=tok).train()

    # 8. save adapters & tokenizer locally --------------------------------
    model.save_pretrained(out_dir)   # LoRA only
    tok.save_pretrained(out_dir)

    # 9. push to S3 --------------------------------------------------------
    key_prefix = f"{prefix.rstrip('/')}/{adapter_nm}"
    upload_dir(out_dir, bucket, key_prefix)

    if "model" in locals():
        model.to("cpu")
    for obj in ("model", "trainer", "dataset", "tokenized"):
        if obj in locals():
            del locals()[obj]

    free_gpu()  

    return {
        "status": "ok",
        "s3_uri": f"s3://{bucket}/{key_prefix}/",
        "adapter_files": ["adapter_model.bin", "adapter_config.json"]
    }

# Needed by RunPod
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
