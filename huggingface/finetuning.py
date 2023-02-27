# https://huggingface.co/docs/transformers/training

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import wandb

wandb.init(project='training-transformers')

dataset = load_dataset("yelp_review_full")
# DatasetDict({
#     train: Dataset({
#         features: ['label', 'text'],
#         num_rows: 650000
#     })
#     test: Dataset({
#         features: ['label', 'text'],
#         num_rows: 50000
#     })
# })

# Return input_ids, token_type_ids, attention_mask
# Attention mask is to attend/not as we don't have to attend padded tokens. Smart!
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Training using Hugginface Trainer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

# TrainingArguments doesn't evaluate. It only trains. Hence we need a metric. 
# High quality implementations are provided by huggingface
metric = evaluate.load("accuracy")

# All huggingface models return logits. We need to convert logits to predictions
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="output_dir", 
    evaluation_strategy="steps", 
    eval_steps=5, 
    save_steps=5, 
    logging_steps=5,
    logging_dir="logs_dir",
    num_train_epochs=1,
    logging_first_step=True,
    use_mps_device=True,
    learning_rate=3e-4,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()