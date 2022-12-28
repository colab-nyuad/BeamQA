import torch
from loaders import  CheckpointLoader
import networkx as nx

def get_embeddings(path,model_name):
    ckp = CheckpointLoader(path)
    entity2idx, rel2idx, embedding_matrix, embedding_matrix_rel = ckp.load_libkge_checkpoint(model_name,dim=400)

    embedding_matrix_rel.append(torch.zeros(embedding_matrix_rel[0].shape[0]))  ### this is a padding embedding
    print('Ent ', len(embedding_matrix), len(embedding_matrix_rel), embedding_matrix[0].shape,
          embedding_matrix_rel[0].shape)

    embedding_matrices = [embedding_matrix, embedding_matrix_rel]
    idx2rel = {k:v for v,k in rel2idx.items()}
    idx2rel[len(idx2rel)] = 'None'
    rel2idx['None'] = len(rel2idx)

    return embedding_matrices , entity2idx, rel2idx , idx2rel

def process_text_file(text_file):
    data_file = open(text_file, 'r')
    data_array = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        question = data_line[0].split('[')
        question_2 = question[1].split(']')
        head = question_2[0].strip()
        ans = data_line[1].split('|')
        path = data_line[2].split('|')
        data_array.append([head, ans, path])
    return data_array[1:]


def load_graph(path):
    nx_graph = nx.read_gpickle(path)
    return nx_graph

