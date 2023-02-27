# https://huggingface.co/docs/transformers/quicktour#trainer-a-pytorch-optimized-training-loop
# Say we want to train a model for sequence classification

from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import Trainer

# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=10)

# Obv, we can't put arbitrary number of classes and expect to have pretrained weights.

# From output: Some weights of DistilBertForSequenceClassification were not initialized 
# from the model checkpoint at distilbert-base-uncased and are newly initialized: 
# ['classifier.weight', 'pre_classifier.weight', 'pre_classifier.bias', 'classifier.bias']
#   (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
#   (classifier): Linear(in_features=768, out_features=10, bias=True)


# Get model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Hmm, interesting why this is the case even when we don't give num_labels. 
# I'd expect it to get same number of classes as pretrained and have same weights

# Get training arguments
training_args = TrainingArguments(
    output_dir="output_dir/",
    learning_rate=2e-5,
    per_device_train_batch_size=8, # Per GPU!!
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    use_mps_device=True, # Improves ~10x from 41 mins to 4 minutes!!
)

print(training_args.device)

# Get tokenizer for the same class. Tokenizer and autoclass are two parts of pipeline
# Also adds spl tokens if needed, like BOS, EOS, etc
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# We need data to train and test
# dataset is a dict with 3 arrow datasets - train, validation, and test
# dataset['train'] = Dataset({
#     features: ['text', 'label'],
#     num_rows: 8530
# })
dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT


def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])


# Creates new columns, input_ids and attention_mask
dataset = dataset.map(tokenize_dataset)

# It takes a batch and pads the batch
# By default, it pads to max length in batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Now create a trainer with model, arguments, dataset, tokenizer, and datacollator (data loader basically)
# You can customize the training loop behavior by subclassing the methods inside Trainer.
# This allows you to customize features such as the loss function, optimizer, and scheduler. 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP

trainer.train()

