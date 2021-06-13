
from dgl.nn.pytorch.conv import TAGConv
import torch
import torch.nn as nn
from common import *
import dgl
import torch.nn.functional as F
from dataloader import DataLoader

class UnPool(nn.Module):
    def __init__(self, id=0):
        super(UnPool, self).__init__()
        self.idx = id

    def forward(self, graph: dgl.DGLGraph, feat, pool_idx_, edge_idx_):
        edge_idx = edge_idx_[self.idx]
        if not isinstance(pool_idx, torch.Tensor):
            pool_idx = torch.LongTensor(pool_idx)
        # if not isinstance(feat, torch.Tensor):
        #     print(True)
        new_vs = 0.5 * torch.sum(feat[pool_idx], dim=1)
        feat = torch.cat((feat, new_vs), 0)
        es_num = graph.number_of_edges()
        graph.remove_edges([i for i in range(es_num)])
        t = {i: new_vs[i] for i in range(new_vs.shape[0])}
        graph.add_nodes(new_vs.shape[0])
        src, dst = tuple(zip(*edge_idx))
        graph.add_edges(src, dst)
        graph.add_edges(dst, src)
        graph.ndata['feat'] = feat
        return graph, feat

class Deformation(nn.Module):
    def __init__(self, feed_dict: dict, opt):
        super(TestGCN, self).__init__()
        self.opt = opt
        self.layers = nn.ModuleList()
        self.in_feats = opt.sad_in_feats
        self.out_feats = opt.sad_out_feats
        self.hidden = opt.sad_hidden_size
        self.layers.append(TAGConv(in_feats=self.in_feats, out_feats=self.hidden, activation=F.relu))
        for _ in range(12):
            self.layers.append(TAGConv(in_feats=self.hidden, out_feats=self.hidden, activation=F.relu))
        self.layers.append(TAGConv(in_feats=self.hidden, out_feats=self.out_feats, activation=lambda x: x))
        self.layers.append(DeformationUnpool(id=0))
        # Block one
        self.layers.append(TAGConv(in_feats=self.in_feats, out_feats=self.hidden, activation=F.relu))
        for _ in range(12):
            self.layers.append(TAGConv(in_feats=self.hidden, out_feats=self.hidden, activation=F.relu))
        self.layers.append(TAGConv(in_feats=self.hidden, out_feats=self.out_feats, activation=lambda x: x))
        self.layers.append(DeformationUnpool(id=1))
        # Block two
        self.layers.append(TAGConv(in_feats=self.in_feats, out_feats=self.hidden, activation=F.relu))
        for _ in range(12):
            self.layers.append(TAGConv(in_feats=self.hidden, out_feats=self.hidden, activation=F.relu))
        self.layers.append(TAGConv(in_feats=self.hidden, out_feats=self.out_feats, activation=lambda x: x))

    def forward(self, g: dgl.DGLGraph, features, pool_idx, edge_idx):
        h = features
        pool = [10, 22]
        eltwise = [2, 4, 6, 8, 10, 12]
        hidden = []
        for i in range(14):
            h = self.layers[i](g, h)
            hidden.append(h)
            if i in eltwise:
                h = 0.5 * (hidden[-2] + h)
        out1 = h
        g, h = self.layers[14](g, h, pool_idx, edge_idx)
        out1_2 = h
        eltwise = [17, 19, 21, 23, 25, 27]
        hidden = []

        for i in range(15, 29):
            h = self.layers[i](g, h)
            hidden.append(h)
            if i in eltwise:
                h = 0.5 * (hidden[-2] + h)
        out2 = h
        g, h = self.layers[29](g, h, pool_idx, edge_idx)  # pool
        out2_2 = h
        eltwise = [32, 34, 36, 38, 40, 42]
        hidden = []
        for i in range(30, 44):
            h = self.layers[i](g, h)
            hidden.append(h)
            if i in eltwise:
                h = 0.5 * (hidden[-2] + h)

        return out1, out2, h, out1_2, out2_2


class DeformationUnpool(nn.Module):
    def __init__(self):
        super(TestUnpool, self).__init__()
        self.layer = UnPool(id=0)

    def forward(self, g: dgl.DGLGraph, features, pool_idx, edge_idx):
        g, h = self.layer(g, features, pool_idx, edge_idx)
        return h

