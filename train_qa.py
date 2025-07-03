import os
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TFTrainingArguments, TFTrainer
import tensorflow as tf

# Load dataset in the new format
data_files = {"train": "serbian_qa_mt5.json"}
dataset = load_dataset("json", data_files=data_files, field="data")

model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input"], max_length=256, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["target"], max_length=64, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, return_tensors="tf")

training_args = TFTrainingArguments(
    output_dir="./results_mt5",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=2e-5,
)

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("./serbian-mt5-qa-model")
    tokenizer.save_pretrained("./serbian-mt5-qa-model")
