import pandas as pd
from utils import process_text_file,load_graph
from beamQA import path_finder_rec
from tqdm import tqdm
import torch
import ast

def data_generator(data, entity2idx, rel2idx,hops):
    df2 = pd.read_csv('../Data/Path_gen/outputs/predictions_metaqa_'+hops+'hop_wscores.txt',index_col=0,delimiter='\t') ## the path generated using path_generation module
    df2 = df2.values
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()] #extract head id

        path = [rel2idx[rel_name.strip()] if  rel_name in rel2idx else 0 for rel_name in data_sample[2] ] # extract relation id

        if type(data_sample[1]) is str:
            ans = entity2idx[data_sample[1]]
        else:
            ans = [entity2idx[entity.strip()] for entity in list(data_sample[1])]

        scores = ast.literal_eval(df2[i][2]) #path scores
        scores = [float(a) for a in scores]
        beam_paths = df2[i][1].split('|')  #paths generated
        yield torch.tensor(head, dtype=torch.long), ans,path , beam_paths,scores


def evaluate_beamQA(data_path, device, model, entity2idx, rel2idx,hops,nx_graph_path,topk):
    model.eval()
    data = process_text_file(data_path)
    data_gen = data_generator(data=data, entity2idx=entity2idx, rel2idx=rel2idx,hops=hops)
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
            scorez = d[4]
            predicted_chains = d[3]
            h = idx2entity[head.item()]
            predicted_chains = [h for h in predicted_chains if len(h.split(' ')) == num_hops]
            predicted_entity, max_score = path_finder_rec(h, predicted_chains, scorez, model, entity2idx, idx2entity, rel2idx, nx_graph,topk=topk, device=device)
            if predicted_entity :
                if entity2idx[predicted_entity] in ans:
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
        target = a[1].to(device)
        relations = a[2].to(device)
        pred1, pred2 = model(positive_head, relations)
        a,b = loss_weights[0],loss_weights[1]
        loss =  a * loss_func(pred1, target)  + b * loss_func(pred2, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loader.set_postfix(Loss=running_loss / ((i_batch + 1) * batch_size), Epoch=epoch)
        loader.set_description('{}'.format(epoch))
        loader.update()
    scheduler.step()


