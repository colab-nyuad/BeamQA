import argparse
import os
from utils import *
from Model import Model
from train_eval import evaluate_beamQA,train
from loaders import Datasetwqsp
from torch.utils.data import DataLoader
import pandas as pd
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--hops', type=str, default='2')
parser.add_argument('--validate_every', type=int, default=20)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--kg_type', type=str, default='half')
parser.add_argument('--mode', type=str, default='BeamQA')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--do_batchnorm', type=bool, default=False)
parser.add_argument('--do_dropout', type=bool, default=False)
parser.add_argument('--decay', type=float, default=1.0)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--topk', type=int, default=20)
args = parser.parse_args()

device = 'cuda:'+str(args.gpu)
kg_model_path = '../Data/Graph_data/Freebase/'
kg_model_name = 'checkpoint_fb_' + args.kg_type + '.pt'
nx_graph = '../Data/Graph_data/Freebase/FB_'+ args.kg_type +'.gpickle'

embedding_matrices , entity2idx, rel2idx , idx2rel = get_embeddings(kg_model_path,kg_model_name,args.embedding_dim)
model = Model(embedding_matrices,args.dropout,args.do_batchnorm,args.do_dropout).to(device)

train_data_path =  '../Data/QA_data/WQSP/train_wqsp.csv'
test_data_path = '../Data/QA_data/WQSP/test_wqsp.txt'
beam_data = '../Data/Path_gen/outputs/wqsp_predictions_wscores.txt'

if args.mode == 'BeamQA':
    test_score = evaluate_beamQA(model=model, data_path=test_data_path, beam_data=beam_data,
                                 entity2idx=entity2idx, rel2idx=rel2idx,
                                 device=device, hops=args.hops, nx_graph_path=nx_graph, topk=args.topk)

elif args.mode =='train-BeamQA':
    train_data = pd.read_csv(train_data_path, header=0, names=['head', 'text', 'rel', 'ans'])
    train_data = train_data.values
    print('Train length ', len(train_data))
    train_data = Datasetwqsp(train_data, entity2idx, rel2idx)
    data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle_data, num_workers=args.num_workers)

    klloss = torch.nn.KLDivLoss(reduction='sum')
    def loss_func(scores, targets):
        return klloss(
            F.log_softmax(scores, dim=1), F.normalize(targets.float(), p=1, dim=1)
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,args.decay)
    optimizer.zero_grad()
    loss_weights = [50,1]
    for epoch in range(args.epochs):
        for phase in range(args.validate_every):
            train(model,data_loader,loss_func,loss_weights,optimizer,scheduler,args.batch_size,epoch,device)

        test_score = evaluate_beamQA(model=model, data_path=test_data_path,beam_data=beam_data,
                              entity2idx=entity2idx, rel2idx=rel2idx,
                              device=device, hops=args.hops, nx_graph_path=nx_graph,topk= args.topk)
#
else: print('The mode entered is wrong')
