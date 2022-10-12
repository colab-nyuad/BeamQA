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
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--hops', type=str, default='1')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--model', type=str, default='facebook/bart-base')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=15)
parser.add_argument('--decay', type=float, default= 0.01)
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

with open('../Data/Path_gen/all_props.pkl', 'rb') as f:
    prop_set = pickle.load(f)

if args.mode == 'train_eval':
    unseen = pd.read_csv('../Data/Path_gen/1hop-synth-t5-unseen-only-small.csv', header=0, names=['QA', 'TAG'])
    train_ext = pd.read_csv('../Data/Path_gen/train_noc.txt', header=0, names=['QA', 'TAG'], sep='\t')


    train_ext = train_ext.append(unseen).reset_index(drop=True)
    print(len(train_ext))
    train_ext.to_csv('../Data/Path_gen/train_extended.txt', sep='\t')

    data_files = {"train": "../Data/Path_gen/train_extended.txt",
                  "test": "/home/ubuntu/farah/Path_gen/test_noc.txt"}

    train_loader = load_dataset("csv", data_files=data_files, names=['text', 'tag'], delimiter='\t', header=0)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")



    print('tokenizer before ',len(tokenizer))
    tokenizer.add_tokens(prop_set)
    print('tokenizer after adding relations as tokens ',len(tokenizer))
    # Tokenize input and target
    tokenized_datasets = train_loader.map(preprocess_function, batched=True)
    model = BartForConditionalGeneration.from_pretrained(args.model)
    ### Freeze the encoder of BART
    freeze_params(model.get_encoder())  ## freeze the encoder
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
                                      weight_decay=args.decay,
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

    data_test = pd.read_csv('../Data/QA_data/WQSP/test_wqsp.txt', names=['text', 'no', 'tag'],
                            delimiter='\t')
    ## count the samples with nan targets to exclude them when computing the hits@1 and hits@3
    to_exclude = len(data_test[data_test['tag'].isnull()])
    data_test.loc[data_test['tag'].isnull(), 'tag'] = 'unknown'
    data_test['text'] = data_test['text'].apply(lambda w: preprocess_sentence(w))

    data_test["tag"] = data_test.tag.apply(lambda w: w.replace('|', ' '))
    org_tar = data_test["tag"].values
    org_tar = ['|'.join(t.split(' ')) for t in org_tar]
    data_test["tag"] = data_test.tag.apply(lambda w: w.split(' '))
    print('LEN TEST ', len(data_test))
    data_test["tag"] = data_test.tag.apply(lambda w: ' '.join(w))
    run_evaluation(model, data_test,tokenizer,to_exclude,args.hops)

if args.mode =='eval':
    model = BartForConditionalGeneration.from_pretrained(args.model)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model.resize_token_embeddings(len(tokenizer))

    data_test = pd.read_csv('../Data/Path_gen/test_bart_metaqa_'+args.hop+'hop.csv')
    to_exclude = len(data_test[data_test['tag'].isnull()])
    data_test.loc[data_test['tag'].isnull(), 'tag'] = 'unknown'
    data_test['text'] = data_test['text'].apply(lambda w: preprocess_sentence(w))
    paths , scores, hop_scores = run_evaluation(model, data_test,tokenizer,to_exclude,args.hops)





