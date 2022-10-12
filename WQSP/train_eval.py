import pandas as pd
from utils import process_text_file,load_graph
from beamQA import path_finder_rec
from tqdm import tqdm
import torch
import ast

def data_generator(data,beam_data, entity2idx):
    df2 = pd.read_csv(beam_data,index_col=0,delimiter='\t',header=0,names=['qa','rel','scores','hopscores'])
    df2 = df2.values
    for i in range(len(data)):
        data_sample = data[i]
        h = data_sample[0].strip().split('[')
        head = h[1].split(']')[0]
        if head in entity2idx : head = entity2idx[head]
        else : head = -1
        if str(data_sample[1]) != 'nan' :
            ans = data_sample[1].split('|')
        else : ans = 'nan'
        scores = ast.literal_eval(df2[i][2])
        scores = [float(a) for a in scores]
        relations = df2[i][1].split('|')
        yield torch.tensor(head, dtype=torch.long), ans, relations , scores

def evaluate_beamQA(data_path,beam_data, device, model, entity2idx, rel2idx,hops,nx_graph_path,topk):
    model.eval()
    test_data = pd.read_csv(data_path, sep='\t', names=['qa', 'ans', 'rel'])
    data = test_data.values
    data_gen = data_generator(data=data,beam_data=beam_data,entity2idx=entity2idx)
    total_correct = 0

    idx2entity = {v:k for k,v in entity2idx.items()}
    num_hops = int(hops.split('hop')[0])
    print('Hops ',num_hops)
    nx_graph = load_graph(nx_graph_path)

    loader = tqdm(range(len(data)))
    with torch.no_grad():
        for i in loader :
            d = next(data_gen)
            head = d[0].to(device)
            ans = d[1]
            scorez = d[3]
            predicted_chains = d[2]
            if head.item() in idx2entity:
                h = idx2entity[head.item()]
            else:
                continue
            predicted_entity, max_score = path_finder_rec(h, predicted_chains, scorez, model,entity2idx, idx2entity, rel2idx, nx_graph, device,topk)
            if predicted_entity :
                if predicted_entity in ans:
                    total_correct += 1
            loader.set_postfix(sample=i, hits_1=(total_correct / (i + 1)))
        accuracy = total_correct / len(data)
        print('Hits@1 ',accuracy)
        return accuracy


def train(model,data_loader,loss_func,loss_weights,optimizer,scheduler,batch_size,epoch,device):
    model.train()
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    running_loss = 0
    for i_batch, a in enumerate(loader):
        model.zero_grad()
        positive_head = a[0].to(device)
        target = a[1].to(device)  # positive tail
        relations = a[2].to(device)
        pred1, pred2 = model(positive_head, relations)
        a,b = loss_weights[0],loss_weights[1]
        loss =  a * loss_func(pred1, target) +  b *loss_func(pred2, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loader.set_postfix(Loss=running_loss / ((i_batch + 1) * batch_size), Epoch=epoch)
        loader.set_description('{}'.format(epoch))
        loader.update()
    scheduler.step()


