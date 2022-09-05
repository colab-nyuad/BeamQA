import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import pandas as pd
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import torch
from transformers import BartTokenizer, BartForConditionalGeneration , BartModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import numpy as np
import nltk
import mlflow
import re
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--hops', type=str, default='1')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--model', type=str, default='facebook/bart-base')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--decay', type=float, default=1.0)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=15)
parser.add_argument('--hop', type=str, default='1')
parser.add_argument('--mode', type=str, default='train_eval')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        l = []
        for i in examples["tag"]:
            l.append(tokenizer(i, max_length=100, truncation=True)["input_ids"])
    model_inputs["labels"] = l
    return model_inputs

if args.mode == 'train_eval':
    # train = pd.read_csv('../Data/QA_data/MetaQA/train_'+args.hop+'hop.txt', header=0, names=['QA', 'Ans', 'TAG'], delimiter='\t')
    # test = pd.read_csv('../Data/QA_data/MetaQA/test_'+args.hop+'hop.txt', header=0, names=['QA', 'Ans', 'TAG'], delimiter='\t')
    train = pd.read_csv('/storage/Embedkg/data/QA_data/MetaQA/train_' + args.hop + 'hop.txt', header=0, names=['QA', 'Ans', 'TAG'],
                    delimiter='\t')
    test = pd.read_csv('/storage/Embedkg/data/QA_data/MetaQA/test_' + args.hop + 'hop.txt', header=0, names=['QA', 'Ans', 'TAG'],
                   delimiter='\t')
    prop_set = get_set_relations(train, hop=2)
    print(len(prop_set), len(train))

    qa, rel = preprocess_data(train)
    train = pd.DataFrame({'text': qa, 'tag': rel})
    train.to_csv('../Data/Path_gen/train_bart_metaqa_'+args.hop+'hop.csv')
    qa, rel = preprocess_data(test)
    test = pd.DataFrame({'text': qa, 'tag': rel})
    test.to_csv('../Data/Path_gen/test_bart_metaqa_'+args.hop+'hop.csv')

    data_files = {"train": "../Data/Path_gen/train_bart_metaqa_"+args.hop+"hop.csv", "test": "../Data/Path_gen/test_bart_metaqa_"+args.hop+"hop.csv"}
    train_loader = load_dataset("csv", data_files=data_files, names=['text', 'tag'], header=0, index_col=0)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    print('tokenizer before ',len(tokenizer))
    tokenizer.add_tokens([i for i in prop_set])
    print('tokenizer after adding relations as tokens ',len(tokenizer))

    tokenized_datasets = train_loader.map(preprocess_function, batched=True)
    model = BartForConditionalGeneration.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


    mlflow.end_run()
    training_args = TrainingArguments(prediction_loss_only = True,
                                      report_to=None,
                                      num_train_epochs=args.epochs,
                                      output_dir='output',
                                      eval_steps = 100,
                                      logging_steps = 50,
                                      save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
                                      load_best_model_at_end=True,
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size,
                                      evaluation_strategy="steps",)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    data_test = test
    data_test.loc[data_test['tag'].isnull(), 'tag'] = 'unknown'
    data_test['text'] = data_test['text'].apply(lambda w: preprocess_sentence(w))
    run_evaluation(model, data_test,tokenizer)

if args.mode =='eval':
    train = pd.read_csv('/storage/Embedkg/data/QA_data/MetaQA/train_' + args.hop + 'hop.txt', header=0,
                        names=['QA', 'Ans', 'TAG'],
                        delimiter='\t')

    prop_set = get_set_relations(train, hop=int(args.hop))
    model = BartForConditionalGeneration.from_pretrained(args.model)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model.resize_token_embeddings(len(tokenizer))

    data_test = pd.read_csv('../Data/Path_gen/test_bart_metaqa_'+args.hop+'hop.csv')
    data_test.loc[data_test['tag'].isnull(), 'tag'] = 'unknown'
    data_test['text'] = data_test['text'].apply(lambda w: preprocess_sentence(w))
    run_evaluation(model, data_test,tokenizer)





