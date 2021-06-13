import sys
from geometrics import *
from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()
from pytorch3d.structures import Meshes
import pytorch3d
from common import build_graph
from deform import Deformation
import torch
import os
from DES.model_nets import finemodel
from rec_data import Tree
import utils
from config import add_train_paser
from argparse import ArgumentParser
from rec_data import PartNetDataset
import numpy as np
import wandb


def unit(tensor):
    # print(tensor)
    tensor = tensor.cpu().clone().detach()
    length = torch.norm(tensor, p=2, dim=1)
    zero_index = torch.where(length == 0)


    re_length = torch.reciprocal(length)

    re_length[re_length == float('inf')] = 0

    length = re_length.reshape((-1, 1)).repeat((1, 3))

    return torch.mul(tensor, length)

def point_loss(pred, v, normal, edge):
    # print("edge : ", edge)
    eidx1= edge[:, 0].clone().detach().long().cuda()
    eidx2= edge[:, 1].clone().detach().long().cuda()
    # eidx1 = torch.tensor(edge[:, 0]).long().cuda()
    # eidx2 = torch.tensor(edge[:, 1]).long().cuda()
    nod1 = torch.index_select(pred, 0, eidx1)
    nod2 = torch.index_select(pred, 0, eidx2)
    e = nod1 - nod2
    pred = torch.unsqueeze(pred, dim=0).cuda()
    v = torch.unsqueeze(v, dim=0).cuda()
    dist_1, dist_2, idx1, idx2 = chamfer_dist(pred, v) # idx2 - pred ; idx1 - v

    p2p_loss = (torch.mean(dist_1) + torch.mean(dist_2)) * 30
    
    idx2 = torch.squeeze(idx2).long()
    print('normal shape ', normal.shape)
    normal = torch.index_select(normal, 0, idx2)
    normal = torch.index_select(normal, 0, eidx1)
    
    cosine = torch.abs(torch.sum(torch.mul(unit(normal), unit(e)), 1)) * 0.5

    normal_loss = torch.mean(cosine) * 1
    return p2p_loss, normal_loss

def coord(v, lap_idx):
    vertices = torch.cat((v, torch.zeros((1, 3)).cuda()), 0)
    indices = torch.tensor(lap_idx[:, :19]).cuda()
    weights = torch.tensor(lap_idx[:, -1], dtype=torch.float32).cuda()
    weights = torch.reciprocal(weights).reshape((-1, 1)).repeat((1, 3))
    num_pts, num_indices = indices.shape[0], indices.shape[1]
    indices = indices.reshape((-1, )).long()
    
    vertices = torch.index_select(vertices, 0, indices)

    vertices = vertices.reshape((num_pts, num_indices, 3))
    laplace = torch.sum(vertices, 1)
    laplace = v - torch.mul(laplace, weights)
    
    return laplace

def laplace_loss(pred, v, lap_idx):
    # print('pre ', pred.shape, ' v shape ', v.shape)
    pred = pred[0]
    v = v[0]
    # print('# laplac : ', pred.shape, v.shape)
    # print(lap_idx.shape)
    lap1 = coord(pred, lap_idx)
    lap2 = coord(v, lap_idx)
    laplace_loss = torch.mean(torch.sum(torch.pow(lap1 - lap2, 2), 1)) * 15
    # print("laplace_loss is ", laplace_loss)
    move_loss = torch.mean(torch.sum(torch.pow(pred - v, 2), 1)) * 1
    return laplace_loss, move_loss


def get_edge(v, f):
    edge = []
    for ff in f:
        ed1 = sorted([ff[0], ff[1]])
        ed2 = sorted([ff[1], ff[2]])
        ed3 = sorted([ff[0], ff[2]])
        if ed1 not in edge:
            edge.append(ed1)
        if ed2 not in edge:
            edge.append(ed2)
        if ed3 not in edge:
            edge.append(ed3)
    return np.array(edge)
def get_lap(v, e):
    lap = {}
    for ee in e:
        v1 = int(ee[0])
        v2 = int(ee[1])
        if v1 not in lap.keys():
            lap[v1] = [v2]
        else:
            lap[v1].append(v2)
        if v2 not in lap.keys():
            lap[v2] = [v1]
        else:
            lap[v2].append(v1)
    assert(len(lap) == v.shape[0])
    if len(lap) != v.shape[0]:
        exit()
    laplacian = []
    for i in range(v.shape[0]):
        seq = sorted(set(lap[i]))
        num_v = len(seq)
        if num_v > 20:
            print('laplacian is wrong!')
            exit()
        laplacian.append(list(seq) + [v.shape[0]] * (19 - num_v) + [num_v])

    return np.array(laplacian)
    
    
    
def subdivision(v, f):
    v2id = {}
    for i in range(v.shape[0]):
        v2id[str(list(v[i]))] = i
    nf = []
    for ff in f:
        v1 = int(ff[0])
        v2 = int(ff[1])
        v3 = int(ff[2])

        new_v1 = (v[v1] + v[v2]) * 0.5
        new_v2 = (v[v2] + v[v3]) * 0.5
        new_v3 = (v[v1] + v[v3]) * 0.5
        # new_v1
        if str(list(new_v1)) not in v2id:
            new_v1_idx = v.shape[0]
            v2id[str(list(new_v1))] = new_v1_idx
            v = np.vstack((v, new_v1))
            
        else:
            new_v1_idx = v2id[str(list(new_v1))]
        # new_v2
        if str(list(new_v2)) not in v2id:
            new_v2_idx = v.shape[0]
            v2id[str(list(new_v2))] = new_v2_idx
            v = np.vstack((v, new_v2))
        else:
            new_v2_idx = v2id[str(list(new_v2))]
        
        if str(list(new_v3)) not in v2id:
            new_v3_idx = v.shape[0]
            v2id[str(list(new_v3))] = new_v3_idx
            v = np.vstack((v, new_v3))
        else:
            new_v3_idx = v2id[str(list(new_v3))]
            
        nf.append([v1, new_v1_idx, new_v3_idx])
        nf.append([v2, new_v1_idx, new_v2_idx])
        nf.append([v3, new_v2_idx, new_v3_idx])
        nf.append([new_v1_idx, new_v2_idx, new_v3_idx])
    
    return v, np.array(nf)

def subdivision_once(v, f):
    new_v, new_f = subdivision(v, f)
    new_e = get_edge(new_v, new_f)
    new_lap = get_lap(new_v, new_e)
    return new_v, new_f, new_e, new_lap

def collate_feats(b):
    return list(zip(*b))

def unit(tensor):
    # print(tensor)
    tensor = tensor.cpu().clone().detach()
    length = torch.norm(tensor, p=2, dim=1)
    zero_index = torch.where(length == 0)


    re_length = torch.reciprocal(length)

    re_length[re_length == float('inf')] = 0

    length = re_length.reshape((-1, 1)).repeat((1, 3))

    return torch.mul(tensor, length)



def init_strucutrenet(category):
    Tree.load_category_info(category)


def main(config):
    if config.wandb:
        wandb.init(config=config, project="PIPELINE")
    
    init_strucutrenet(config.category)
    dataset = PartNetDataset(config.obb_root, config.deform_root,
                       config.img_root, config.object_list)
    device = torch.device(config.device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                             collate_fn=collate_feats)
    img_encoder = finemodel('pretrained/fullmodel.pkl')
    models = utils.get_model_module(config.model_version)
    encoder = models.RecursiveEncoder(
        config, variational=True, probabilistic=False)
    decoder = models.RecursiveDecoder(config)
    img_encoder.to(device)
    encoder.to(device)
    decoder.to(device)
    decoder.load_state_dict(
        torch.load('pretrained/180_net_decoder.pth'))
    encoder.load_state_dict(
        torch.load('pretrained/180_net_encoder.pth'))
    for param in decoder.parameters():
        param.requires_grad = False
    for param in encoder.parameters():
        param.requires_grad = False
    deform = Deformation(config)
    deform.to(device)
    img_opt = torch.optim.Adam(img_encoder.parameters(),
                                 lr=config.img_lr
                                 )
    deform_opt = torch.optim.Adam(deform.parameters(),
                               lr=config.deform_lr)
    for epoch in range(config.epoches):
        for i, (objs, names, imgs, edxs, gts, laps, normals, faces) in enumerate(dataloader):

            diff = torch.zeros(1, device=device)
            deform_loss = torch.zeros(1, device=device)
            laplacian = torch.zeros(1, device=device)
            pointloss = torch.zeros(1, device=device)
            moveloss = torch.zeros(1, device=device)
            normloss = torch.zeros(1, device=device)
            obbs = []
            fs = []
            for obj, img, in zip(objs, imgs):
                img = img.reshape(1, 3, 224, 224).cuda()
                obj.to(device)
                img_code = img_encoder(img)  # 256
                box_code = encoder.encode_structure(obj=obj)
                diff += torch.sqrt(torch.sum((img_code - box_code)
                                             ** 2) / box_code.shape[1])
                if epoch >= config.front_epoch:
                    gen_obb = decoder.decode_structure(
                        z=img_code.detach(), max_depth=config.max_tree_depth)
                    boxes = obj.boxes(leafs_only=True)
                    v, f = utils.gen_obb_mesh(boxes)
                    obbs.append(v)
                    fs.append(f)

            if epoch >= config.front_epoch:
                for obb, f, edx, gt, lap, normal, face, name in zip(
                    obbs, fs, edxs, gts, laps, normals, faces, names
                ):
                    
                    vs = obb
                    f = f -1 
                    es = get_edge(vs, f)
                    lap1_idx = get_lap(vs, es)
                    v2, f2, e2,lap2_idx = subdivision_once(vs, f)
                    v3, f3, e3, lap3_idx = subdivision_once(v2, f2)    
                    vs = torch.tensor(vs)
                    es = torch.tensor(es).long()
                    e2 = torch.tensor(e2).long()
                    e3 = torch.tensor(e3).long()
                    f = torch.tensor(f)
                    f2 = torch.tensor(f2)
                    f3 = torch.tensor(f3)
                    lap2_idx = torch.Tensor(lap2_idx)
                    lap3_idx = torch.Tensor(lap3_idx)
                    
                    g = build_graph(vs, es)
                    g.ndata['feat'] = vs
                    vs = vs.cuda()
                    es = es.cuda()
                    e2 = e2.cuda()
                    e3 = e3.cuda()
                    normal = torch.Tensor(normal).to(device)
                    out1, out2, out, out1_2, out2_2 = deform(
                        g, vs, [es, e2], [e2, e3])
                    
                    
                    vs = vs.unsqueeze(0).cuda()
                    out1 = out1.unsqueeze(0).cuda()
                    out2 = out2.unsqueeze(0).cuda()
                    out = out.unsqueeze(0).cuda()
                    out1_2 = out1_2.unsqueeze(0).cuda()
                    out2_2 = out2_2.unsqueeze(0).cuda()
                    gt = torch.Tensor(gt).unsqueeze(0).cuda()
                    pts_1 ,normal_loss_1 = point_loss(out1[0], gt[0], normal, es)
                    pts_2, normal_loss_2 = point_loss(out2[0], gt[0], normal, e2)
                    pts_3, normal_loss_3 = point_loss(out[0], gt[0], normal, e3)
                    pts =  pts_1 + pts_2 + pts_3
                    
                    normalloss = normal_loss_1 + normal_loss_2 + normal_loss_3
                    
                    laplace_1, mov1 = laplace_loss(out1, vs, lap1_idx)
                    laplace_2, mov2 = laplace_loss(out2, out1_2, lap2_idx)
                    laplace_3, mov3 = laplace_loss(out, out2_2, lap3_idx)
                    lap_loss = laplace_1 + laplace_2 + laplace_3                    
                    m = mov1 + mov2 + mov3
                    
                    laplacian += lap_loss
                    moveloss += m
                    pointloss += pts
                    normloss += normalloss
            
            diff = diff / len(objs)
            if epoch >= config.front_epoch:
                laplacian = laplacian / len(objs)
                pointloss = pointloss / len(objs)
                moveloss = moveloss / len(objs)
                normloss = normloss / len(objs)
                deform_loss += config.lap_wlaplacian + config.pts_w * pointloss + config.mov_w * moveloss + config.norm_wnormloss
                deform_opt.zero_grad()
                deform_loss.backward()
                deform_opt.step()
            img_opt.zero_grad()
            diff.backward()
            img_opt.step()
            
            
            if config.wandb:
                wandb.log({'img loss': diff.item(), 'deform loss': deform_loss.item(), \
                    'laplacian loss': laplacian.item(), 'pointloss': pointloss.item(),\
                         'movloss': moveloss.item(),'normloss':normloss.item()})
            print('Epoch :[{0}/{1}], batch: [{2}/{3}], the img diff is {4}, the deform loss is {5}'.format(
                epoch, config.epoches, i, len(dataloader) , diff.item(), deform_loss.item()) )
               
        if (epoch+1) % 10 == 0:     
           torch.save(img_encoder.state_dict(), '{1}/img_encoder_{0}.pkl'.format(epoch, config.save_state))
           if epoch >= config.front_epoch:
               torch.save(deform.state_dict(), '{1}/deform_{0}.pkl'.format(epoch, config.save_state))
                   


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_train_paser(parser)
    config = parser.parse_args()
    main(config)