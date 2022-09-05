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

def process_text_file(text_file, split=False):
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

def data_generator(data, word2ix, entity2idx, rel2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        ques = data_sample[1].strip()
        question = ques.split(' ')
        encoded_question = [word2ix[word.strip()] for word in question]
        if len(data_sample) == 4:
            path = [rel2idx[rel_name.strip()] for rel_name in data_sample[3]]
        else:
            path = None
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long), torch.tensor(encoded_question, dtype=torch.long), ans, torch.tensor(
            len(encoded_question), dtype=torch.long), data_sample[1], path , ques


def load_graph(path):
    nx_graph = nx.read_gpickle(path)
    return nx_graph

