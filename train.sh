#!/bin/bash

cd MetaQA
python main.py --hops 2 --kg_type full --mode train-BeamQA --batch_size 128 --do_batchnorm True --do_dropout True --embedding_dim 400 --lr 0.005 --validate_every 5 --epochs 10