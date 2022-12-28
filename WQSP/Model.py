
import torch
from torch import nn, Tensor



class Model(nn.Module):

    def __init__(self, pretrained_embeddings,dropout_val,do_batchnorm,do_dropout,freeze=False):
        super().__init__()
        self.encoder_head = nn.Embedding.from_pretrained(torch.stack(pretrained_embeddings[0]), freeze=freeze)
        self.encoder_rel = nn.Embedding.from_pretrained(torch.stack(pretrained_embeddings[1]), freeze=freeze)
        self.dropout = torch.nn.Dropout(dropout_val)
        self.batchnorm1 = nn.BatchNorm1d(2048)

        self.dropout_rel = torch.nn.Dropout(dropout_val)
        self.do_batchnorm = do_batchnorm
        self.do_dropout = do_dropout

        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn2 = torch.nn.BatchNorm1d(2)

    def forward(self, head, relation) -> Tensor:
        head = self.encoder_head(head)
        relations = self.encoder_rel(relation)
        hops = relations.shape[1]
        if hops == 1: R = relations.squeeze(1)
        if hops == 2:
            R = self.new_operation(relations[:, 0, :], relations[:, 1, :])
        if hops == 3:
            R = self.new_operation(relations[:, 0, :], relations[:, 1, :])
            R = self.new_operation(R, relations[:, 2, :])

        pred = self.ComplEx(head, R)
        for r in range(relations.shape[1]):
            rp1 = relations[:, r, :]
            pred2 = self.ComplEx(head, rp1)
            new_head = pred2.argmax(dim=1)
            head = self.encoder_head(new_head)
        return pred, pred2

    def another_forward(self, head, relation) -> Tensor:
        head = self.encoder_head(head)
        relations = self.encoder_rel(relation)
        pred = self.ComplEx(head, relations)
        return pred

    def extract(self, emb):
        real, imaginary = torch.chunk(emb, 2, dim=1)
        return real, imaginary

    def new_operation(self, r1, r2):
        re_r1, im_r1 = self.extract(r1)
        re_r2, im_r2 = self.extract(r2)
        re_r = re_r1 * re_r2 - im_r1 * im_r2
        im_r = re_r1 * im_r2 + im_r1 * re_r2
        r = torch.cat([re_r, im_r], dim=1)
        return r

    def get_score_ranked(self, preds):
        top2 = torch.topk(preds, k=2, largest=True, sorted=True)
        return top2, None

    # inspired by https://github.com/malllabiisc/EmbedKGQA
    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batchnorm :
            head = self.bn0(head)
        if self.do_dropout :
            head = self.dropout(head)
            relation = self.dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.encoder_head.weight, 2, dim=1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batchnorm :
            score = self.bn2(score)
        if self.do_dropout : score = self.dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))
        return score


