import torch

def check(head, rel,model,rel2idx,entity2idx,idx2entity,nx_graph,device,topk = 50 ,retscore = False):
    if head not in entity2idx or rel not in rel2idx:
        return [(None, None)]
    s_id = entity2idx[head]
    rel_id = rel2idx[rel]
    s = torch.Tensor([s_id]).long().to(device)            # subject indexes
    p = torch.Tensor([rel_id]).long().to(device)          # relation indexes
    scores = model.another_forward(s, p)        # scores of all objects for (s,p,?)
    # scores = torch.softmax(scores ,dim=1)
    edgeidx = torch.tensor([entity2idx[i[1]]
                            for i in nx_graph.out_edges(head, data='data')
                            if i[2] == rel and i[1] in entity2idx]).long().to(device)

    scores.index_fill_(1, edgeidx, 1)
    scores.index_fill_(1, s, 0)
    sc, o = torch.topk(scores, topk, largest=True, dim=-1)  # index of highest-scoring objects
    ans = [idx2entity[ent] for ent in o.tolist()[0]]
    answr_score = dict(zip(ans, sc.tolist()[0]))
    if retscore:
        return [(k, v) for k, v in sorted(answr_score.items(), key=lambda item: item[1], reverse=True)]
    return [k for k, v in sorted(answr_score.items(), key=lambda item: item[1], reverse=True)]


def check_rec(prev_return, rel,head,topK,model,entity2idx,idx2entity,rel2idx,nx_graph,device):

    if rel not in rel2idx or any(headname not in entity2idx
                                 for headname in list(zip(*prev_return))[0]):
        return [(None, None)]
    entity_score = []
    for prev_entity, prev_score in prev_return:
        for entity, score in check(prev_entity, rel,model,rel2idx,entity2idx,idx2entity,nx_graph,device,topk = topK, retscore = True):
            if entity !=  head: entity_score.append((entity, score * prev_score ))
    return sorted(entity_score, key= lambda x: x[1], reverse=True)[:topK]


def path_finder_rec(headname, chains,scorez,model,entity2idx,idx2entity,rel2idx,nx_graph,device):
    max_score = 0
    predicted_entity = ''
    predicted_path = ''
    topK = 10
    idx = 0
    tops = []
    for path,pscore in zip(chains,scorez):
        path = path.split()
        prev_return = [(headname, 1)]
        for j , path_i in enumerate(path):
            prev_return = check_rec(prev_return, path_i, headname, topK, model, entity2idx, idx2entity,
                                    rel2idx, nx_graph, device)
            entity, score = prev_return[0]
        if entity and score * pscore > max_score:
            predicted_entity = entity
            max_score = score * pscore
            tops = prev_return
        idx += 1
    return predicted_entity, max_score ,tops