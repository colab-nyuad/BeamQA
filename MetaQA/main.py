import argparse
import os
from utils import *
from Model import Model
from train_eval import evaluate_beamQA,train
from loaders import DatasetMetaQA_all_hops
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--hops', type=str, default='1')
parser.add_argument('--validate_every', type=int, default=5)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--kg_type', type=str, default='half')
parser.add_argument('--labelsmoothing', type=float, default=0.0)
parser.add_argument('--mode', type=str, default='BeamQA')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--do_batchnorm', type=bool, default=True)
parser.add_argument('--do_dropout', type=bool, default=True)
parser.add_argument('--decay', type=float, default=1.0)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=400)
parser.add_argument('--topk', type=int, default=20)

args = parser.parse_args()

device = 'cuda:'+str(args.gpu)
## KG path
kg_model_path = '../Data/Graph_data/MetaQA/MetaQA_'+args.kg_type+'/'
## best kg embedding model (obtained with libKge)
kg_model_name = 'checkpoint_best.pt'
nx_graph_path ='../Data/Graph_data/MetaQA/MetaQA-'+args.kg_type+'.gpickle'

embedding_matrices , entity2idx, rel2idx , idx2rel = get_embeddings(kg_model_path,kg_model_name)
model = Model(embedding_matrices,args.dropout,args.do_batchnorm,args.do_dropout).to(device)

train_data_path = '../Data/QA_data/MetaQA/train_'+str(args.hops)+'hop.txt'
test_data_path = '/storage/Embedkg/data/QA_data/MetaQA/test_'+str(args.hops)+'hop.txt'
### Graph created using networx
nx_graph = '../Data/Graph_data/MetaQA/MetaQA-'+ args.kg_type +'.gpickle'

if args.mode == 'BeamQA':
    test_score = evaluate_beamQA(model=model, data_path=test_data_path,
                                 entity2idx=entity2idx, rel2idx=rel2idx,
                                 device=device, hops=args.hops, nx_graph_path=nx_graph, topk=args.topk)

elif args.mode =='train-BeamQA':
    train_data = process_text_file(train_data_path)
    dataset1 = DatasetMetaQA_all_hops(data=train_data,  entity2idx=entity2idx,rel2idx=rel2idx)
    data_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    loss_func = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,args.decay)
    optimizer.zero_grad()
    ### The following are weights to use for the loss, should be changed with depending on the experiment
    #KG full 2hop,1hop weights [10,1]
    #KG full 3hop weights [100,1]
    #KG half 1hop [30 ,1]
    #KG half 2hop [100,1]
    #KG half 3hop [50,1]
    loss_weights = [10,1]
    for epoch in range(args.epochs):
        for phase in range(args.validate_every):
            train(model,data_loader,loss_func,loss_weights,optimizer,scheduler,args.batch_size,epoch,device)

        test_score = evaluate_beamQA(model=model, data_path=test_data_path,
                                  entity2idx=entity2idx, rel2idx=rel2idx,
                                  device=device, hops=args.hops, nx_graph_path=nx_graph,topk=args.topk)

else: print('The mode entered is wrong')
