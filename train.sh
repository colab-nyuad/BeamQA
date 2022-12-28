#!/bin/bash

cd MetaQA
#python main.py --hops 3 --kg_type half --mode train-BeamQA --batch_size 128 --do_batchnorm True --do_dropout True --embedding_dim 400 --lr 0.0005 --validate_every 10
# --epochs 10 --gpu 1

#python main.py --hops 1 --kg_type half --mode train-BeamQA --batch_size 128 --do_batchnorm True --do_dropout True --embedding_dim 400 --lr 0.005 --validate_every 40
#--epochs 10 --gpu 1

#python main.py --hops 2 --kg_type half --mode train-BeamQA --batch_size 128 --do_batchnorm True --do_dropout True --embedding_dim 400 --lr 0.0005 --validate_every 40
# --epochs 10 --gpu 1

python main.py --hops 1 --kg_type full --mode train-BeamQA --batch_size 128 --do_batchnorm True --do_dropout True --embedding_dim 400 --lr 0.005 --validate_every 10
--epochs 10 --gpu 1

#cd WQSP
#python main.py --hops 2 --kg_type half --mode train-BeamQA --batch_size 128 --do_batchnorm False --do_dropout False --embedding_dim 400 --lr 0.005 --validate_every 10 --epochs 10 --gpu 1