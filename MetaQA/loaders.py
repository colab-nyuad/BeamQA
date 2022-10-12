from torch import nn
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import os
import torch
from torch.utils.data import Dataset

class DatasetMetaQA_all_hops(Dataset):
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
        head_id = self.entity2idx[data_point[0].strip()]
        path = [self.rel2idx[rel_name] for rel_name in data_point[2]]
        tail_ids = []
        for tail_name in data_point[1]:
            tail_name = tail_name.strip()
            tail_ids.append(self.entity2idx[tail_name])
        tail_onehot = self.toOneHot(tail_ids)
        head_id = torch.tensor(head_id,dtype=torch.long)
        return head_id, tail_onehot, torch.tensor(path)

class CheckpointLoader():
    ''' Load graph embeddings, relation, and entities index '''
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
        rel2idx, idx2rel, self.relation_matrix, _, _ = self.extract_rel_embeddings(kge_model._relation_embedder, relation_dict)
        return entity2idx, rel2idx ,  self.embedding_matrix ,self.relation_matrix


    def extract_embeddings(self,embedder, inst_dict, bias=False):
        'Extract entity embeddings and entity to index mappings'
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

    def extract_rel_embeddings(self,embedder, inst_dict, bias=False):
        'Extract relation embeddings and relation to index mappings'

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
                idx += 1
        ### add reverse relation (add '_inv' to relations) before extracting embeddings
        def add_inv_rel(rel2idx):
            keyz = list(rel2idx.keys())
            for r in range(len(rel2idx)):
                rel2idx[keyz[r] + "_inv"] = len(rel2idx)
            return rel2idx
        inst2idx = add_inv_rel(inst2idx)

        for i,j in enumerate(inst2idx.keys()):
            entry = embedder._embeddings(torch.LongTensor([i]))[0]
            embedding_matrix.append(entry)

        return inst2idx, idx2inst, embedding_matrix, bh, bt
