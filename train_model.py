#
# 
# Inspired through: https://huggingface.co/docs/transformers/training
# and https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
# 
# 
# run python train_model.py --model_name_or_path "gpt2" --output_dir "/storage/ukp/work/matzken/language_detect" --learning_rate 2e-5 --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --num_train_epochs 2 --weight_decay 0.01

import datasets
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import argparse
import random
from collections import defaultdict
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
        required=True,
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=2, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1234, 
        help="A seed for reproducible training."
    )

    args = parser.parse_args()
    return args
    
def main():
    
    args = parse_args()
    random.seed(args.seed)
    print("Load corpus...")
    dataset_language = load_dataset("common_language")
    dataset_language = dataset_language.remove_columns(["client_id","audio", "path", "gender", "age"])
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,use_fast=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token


    id2label = {0: 'Arabic',
    1: 'Basque',
    2: 'Breton',
    3: 'Catalan',
    4: 'Chinese_China',
    5: 'Chinese_Hongkong',
    6: 'Chinese_Taiwan',
    7: 'Chuvash',
    8: 'Czech',
    9: 'Dhivehi',
    10: 'Dutch',
    11: 'English',
    12: 'Esperanto',
    13: 'Estonian',
    14: 'French',
    15: 'Frisian',
    16: 'Georgian',
    17: 'German',
    18: 'Greek',
    19: 'Hakha_Chin',
    20: 'Indonesian',
    21: 'Interlingua',
    22: 'Italian',
    23: 'Japanese',
    24: 'Kabyle',
    25: 'Kinyarwanda',
    26: 'Kyrgyz',
    27: 'Latvian',
    28: 'Maltese',
    29: 'Mongolian',
    30: 'Persian',
    31: 'Polish',
    32: 'Portuguese',
    33: 'Romanian',
    34: 'Romansh_Sursilvan',
    35: 'Russian',
    36: 'Sakha',
    37: 'Slovenian',
    38: 'Spanish',
    39: 'Swedish',
    40: 'Tamil',
    41: 'Tatar',
    42: 'Turkish',
    43: 'Ukranian',
    44: 'Welsh'}
    
    label2id = dict((v,k) for k,v in id2label.items())

    # Tokenize all texts and align the labels with them. - inspired from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['sentence'],
            padding=False,
            truncation=True,
            max_length=None,
        )
        tokenized_inputs["labels"] = examples['language']
        return tokenized_inputs


    tokenized_data = dataset_language.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)


    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=45, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
if __name__ == "__main__":
    main()