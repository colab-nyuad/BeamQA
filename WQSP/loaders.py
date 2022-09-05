from torch import nn
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import os
import torch
from torch.utils.data import Dataset
import ast

class Datasetwqsp(Dataset):
    def __init__(self, data, entity2idx, rel2idx):
        self.data = data
        self.entity2idx = entity2idx
        self.rel2idx = rel2idx

    def __len__(self):
        return len(self.data)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index):
        data_point = self.data[index]
        head = data_point[0]
        head_id = self.entity2idx[head.strip()]
        tail_ids = []
        for tail_name in data_point[3].split('|'):
            tail_name = tail_name.strip()
            tail_ids.append(self.entity2idx[tail_name] if tail_name in self.entity2idx else 0)
        tail_onehot = self.toOneHot(tail_ids)
        rel = ast.literal_eval(data_point[2])[0]
        relations = [self.rel2idx[r] if r in self.rel2idx else len(self.rel2idx)-1 for r in rel.split('|')]
        if len(relations) == 1: relations.append(len(self.rel2idx)-1)
        return torch.tensor(head_id, dtype=torch.long), tail_onehot, torch.tensor(relations)


class CheckpointLoader():
    def __init__(self,embedding_path):
        self.embedding_path = embedding_path

    def load_libkge_checkpoint(self,path,dim):
        checkpoint = os.path.join(self.embedding_path,path)
        kge_checkpoint = load_checkpoint(checkpoint)
        kge_model = KgeModel.create_from(kge_checkpoint)
        kge_model.eval()

        entities_dict = '{}/entity_ids.del'.format(self.embedding_path)
        bias = True if kge_model._entity_embedder.dim > dim else False
        entity2idx, idx2entity, self.embedding_matrix, self.bh, self.bt = self.extract_embeddings(kge_model._entity_embedder,
                                                                                             entities_dict, bias=bias)

        relation_dict = '{}/relation_ids.del'.format(self.embedding_path)
        rel2idx, idx2rel, self.relation_matrix, _, _ = self.extract_embeddings(kge_model._relation_embedder, relation_dict)
        return entity2idx, rel2idx ,  self.embedding_matrix ,self.relation_matrix


    def extract_embeddings(self,embedder, inst_dict, bias=False):
        if hasattr(embedder, 'base_embedder'):
            embedder = embedder.base_embedder

        inst2idx = {}
        idx2inst = {}
        embedding_matrix = []
        bh = []
        bt = []
        idx = 0

        with open(inst_dict, 'r') as f:
            for line in f.readlines():
                line = line[:-1].split('\t')
                inst_id = int(line[0])
                inst_name = line[1]

                inst2idx[inst_name] = idx
                idx2inst[idx] = inst_name
                entry = embedder._embeddings(torch.LongTensor([inst_id]))[0]
                if bias:
                    bt.append(entry[-1:])
                    bh.append(entry[-2:-1])
                    entry = entry[:-2]

                embedding_matrix.append(entry)
                idx += 1

        return inst2idx, idx2inst, embedding_matrix, bh, bt

