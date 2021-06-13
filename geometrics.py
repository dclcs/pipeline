import open3d as o3d
import torch
import copy
from chamfer_distance import ChamferDistance
import numpy as np
import copy
import matplotlib.pyplot as plt

# from p2mesh import sample_surface
# from pytorch3d.ops import sample_points_from_meshes
# from pytorch3d.structures import Meshes

chamfer_dist = ChamferDistance()

from tri_distance import TriDistance

tri_dist = TriDistance()


class Plane():
    def __init__(self, point, direction):
        self.Point = point.clone()
        self.Direction = direction / torch.sqrt(torch.sum(direction ** 2, dim=-1)).unsqueeze(-1)

    def IsAbove(self, q):
        return torch.bmm(q.unsqueeze(1), self.Point.unsqueeze(-1)).view(-1) <= 0

    def Project(self, point):
        orig = self.Point
        v = point - orig
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        nx, ny, nz = self.Direction[:, 0], self.Direction[:, 1], self.Direction[:, 2]
        dist = (vx * nx + vy * ny + vz * nz).unsqueeze(-1)
        projected_point = point - dist * self.Direction
        return projected_point


class edge():
    def __init__(self, a, b):
        self.A = a.clone()
        self.B = b.clone()
        self.Delta = b - a

    def PointAt(self, t):
        return self.A + t.unsqueeze(-1) * self.Delta

    def LengthSquared(self):
        return torch.sum(self.Delta ** 2, dim=-1)

    def Project(self, p):
        vec = p - self.A
        vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
        nx, ny, nz = self.Delta[:, 0], self.Delta[:, 1], self.Delta[:, 2]
        vec = (vx * nx + vy * ny + vz * nz)
        return vec / self.LengthSquared()


def sample(vertices, faces, num_samples: int, eps: float = 1e-10):
    # print(vertices.shape, faces.shape)
    dist_uni = torch.distributions.Uniform(
        torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())

    # calculate area of each face
    x1, x2, x3 = torch.split(torch.index_select(
        vertices, 0, faces[:, 0]) - torch.index_select(
        vertices, 0, faces[:, 1]), 1, dim=1)
    y1, y2, y3 = torch.split(torch.index_select(
        vertices, 0, faces[:, 1]) - torch.index_select(
        vertices, 0, faces[:, 2]), 1, dim=1)
    a = (x2 * y3 - x3 * y2) ** 2
    b = (x3 * y1 - x1 * y3) ** 2
    c = (x1 * y2 - x2 * y1) ** 2
    Areas = torch.sqrt(a + b + c) / 2
    # percentage of each face w.r.t. full surface area
    Areas = Areas / (torch.sum(Areas) + eps)

    # define descrete distribution w.r.t. face area ratios caluclated
    cat_dist = torch.distributions.Categorical(Areas.view(-1))
    face_choices = cat_dist.sample([num_samples])

    # from each face sample a point
    select_faces = faces[face_choices]
    v0 = torch.index_select(vertices, 0, select_faces[:, 0])
    v1 = torch.index_select(vertices, 0, select_faces[:, 1])
    v2 = torch.index_select(vertices, 0, select_faces[:, 2])
    u = torch.sqrt(dist_uni.sample([num_samples]))
    v = dist_uni.sample([num_samples])
    points = (1 - u) * v0 + (u * (1 - v)) * v1 + u * v * v2

    return points, face_choices


def calc_point_to_line(p, triangles, point_options, device):
    a, b, c = triangles
    counter_p = torch.zeros(p.shape).to(device)

    EdgeAb = edge(a, b)
    EdgeBc = edge(b, c)
    EdgeCa = edge(c, a)

    uab = EdgeAb.Project(p)
    uca = EdgeCa.Project(p)
    ubc = EdgeBc.Project(p)

    TriNorm = torch.cross(a - b, a - c)
    TriPlane = Plane(EdgeAb.A, TriNorm)

    # type 1
    cond = (point_options == 1)
    counter_p[cond] = EdgeAb.A[cond]

    # type 2
    cond = (point_options == 2)
    counter_p[cond] = EdgeBc.A[cond]

    # type 3
    cond = (point_options == 3)
    counter_p[cond] = EdgeCa.A[cond]

    # type 4
    cond = (point_options == 4)
    counter_p[cond] = EdgeAb.PointAt(uab)[cond]

    # type 5
    cond = (point_options == 5)
    counter_p[cond] = EdgeBc.PointAt(ubc)[cond]

    # type 6
    cond = (point_options == 6)
    counter_p[cond] = EdgeCa.PointAt(uca)[cond]

    # type 0
    cond = (point_options == 0)
    counter_p[cond] = TriPlane.Project(p)[cond]

    distances = torch.mean(torch.sum((counter_p - p) ** 2, dim=-1))

    return distances


def batch_sample(verts, faces, device, num=14000):
    dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
    batch_size = verts.shape[0]

    x1, x2, x3 = torch.split(torch.index_select(verts, 1, faces[:, 0]) - torch.index_select(verts, 1, faces[:, 1]), 1,
                             dim=-1)

    y1, y2, y3 = torch.split(torch.index_select(verts, 1, faces[:, 1]) - torch.index_select(verts, 1, faces[:, 2]), 1,
                             dim=-1)
    # print("y3 is ", y3)
    a = (x2 * y3 - x3 * y2) ** 2
    b = (x3 * y1 - x1 * y3) ** 2
    c = (x1 * y2 - x2 * y1) ** 2
    Areas = torch.sqrt(a + b + c) / 2
    Areas = Areas.squeeze(-1) / torch.sum(Areas, dim=1)  # percentage of each face w.r.t. full surface area

    # define descrete distribution w.r.t. face area ratios caluclated
    choices = None

    for A in Areas:

        if choices is None:
            # print('choice is none')
            choices = torch.multinomial(A, num, True)  # list of faces to be sampled from
        else:
            choices = torch.cat((choices, torch.multinomial(A, num, True)))
    select_faces = faces[choices].view(verts.shape[0], 3, num)

    face_arange = verts.shape[1] * torch.arange(0, batch_size).unsqueeze(-1).expand(batch_size, num).to(device)
    select_faces = select_faces + face_arange.unsqueeze(1)

    select_faces = select_faces.view(-1, 3)
    flat_verts = verts.view(-1, 3)

    xs = torch.index_select(flat_verts, 0, select_faces[:, 0])
    ys = torch.index_select(flat_verts, 0, select_faces[:, 1])
    zs = torch.index_select(flat_verts, 0, select_faces[:, 2])
    u = torch.sqrt(dist_uni.sample_n(batch_size * num))
    v = dist_uni.sample_n(batch_size * num)

    points = (1 - u) * xs + (u * (1 - v)) * ys + u * v * zs
    points = points.view(batch_size, num, 3)

    return points


def batch_point_to_surface(pred_vert, faces, gt_points, device, num=14000, f1=False, save_sample=False):
    no_sampling = False
    if no_sampling:
        gt_points = gt_points.unsqueeze(0).to(device)
        pred_vert = pred_vert.unsqueeze(0).to(device)
        print('pred_vert shape ', pred_vert.shape)
        print('gt shape ', gt_points.shape)

        dist_1, dist_2, id_g, id_p = chamfer_dist(gt_points, pred_vert)
        loss = torch.mean(dist_1) + torch.mean(dist_2)
        return loss

    faces = torch.Tensor(faces)
    faces = faces.long().to(device)

    gt_points = gt_points.unsqueeze(0).to(device)
    pred_vert = pred_vert.unsqueeze(0).to(device)
    batch_size = pred_vert.shape[0]

    pred_points = batch_sample(pred_vert, faces, num=num, device=device)

    #####
    # ptp
    #####

    dist_1, dist_2, id_g, id_p = chamfer_dist(gt_points, pred_points)

    #####
    # ptp
    #####
    tri1 = torch.index_select(pred_vert, 1, faces[:, 0])  # todo adj
    tri2 = torch.index_select(pred_vert, 1, faces[:, 1])
    tri3 = torch.index_select(pred_vert, 1, faces[:, 2])
    _, point_options, index = tri_dist(gt_points, tri1, tri2, tri3)

    tri_sets = [tri1, tri2, tri3]
    point_options = point_options.view(-1)
    points_range = tri1.shape[1] * torch.arange(0, batch_size).to(device).unsqueeze(-1).expand(batch_size,
                                                                                               gt_points.shape[1])
    index = (index.long() + points_range).view(-1)

    for i, t in enumerate(tri_sets):
        t = t.view(-1, 3)
        tri_sets[i] = torch.index_select(t, 0, index)

    dist_2 = calc_point_to_line(gt_points.contiguous().view(-1, 3), tri_sets, point_options, device)

    ratio = dist_1.data.cpu().numpy() / (dist_2.data.cpu().numpy())
    loss = (torch.mean(dist_1) + torch.mean(dist_2))
    ####
    # f1
    ####
    if f1 and save_sample:
        dist_to_pred = torch.sqrt(
            torch.sum((.57 * pred_counters - .57 * gt_points.contiguous().view(-1, 3)) ** 2, dim=1)).view(batch_size,
                                                                                                          -1)
        dist_to_gt = torch.sqrt(torch.sum((.57 * gt_counters - .57 * pred_points.view(-1, 3)) ** 2, dim=1)).view(
            batch_size, -1)

        f_score_1 = 0
        th = 1e-2
        for i in range(dist_to_pred.shape[0]):
            recall = float(torch.where(dist_to_pred[i] <= th)[0].shape[0]) / float(num)
            precision = float(torch.where(dist_to_gt[i] <= th)[0].shape[0]) / float(num)
            f_score_1 += 2 * (precision * recall) / (precision + recall + 1e-8)

        f_score_1 = 100 * f_score_1 / (batch_size)
        # visualization(gt_points, pred_points, dist_to_pred, dist_to_gt,max_distance=th)
        f_score_2 = 0
        th = 1e-4
        for i in range(dist_to_pred.shape[0]):
            recall = float(torch.where(dist_to_pred[i] <= th)[0].shape[0]) / float(num)
            precision = float(torch.where(dist_to_gt[i] <= th)[0].shape[0]) / float(num)
            f_score_2 += 2 * (precision * recall) / (precision + recall + 1e-8)

        f_score_2 = 100 * f_score_2 / (batch_size)
        # visualization(gt_points, pred_points, dist_to_pred, dist_to_gt,max_distance=th)

        # return loss, f_score
        # f_score_1 = 0
        # f_score_2 = 1
        # return loss, [f_score_1, f_score_2]
    else:
        return loss

def batch_calc_edge(verts, faces):
    faces = torch.Tensor(faces)
    verts = verts.unsqueeze(0)
    # faces = faces[0]
    faces = faces.long().cuda()
    # get vertex loccations of faces
    p1 = torch.index_select(verts, 1, faces[:, 0])
    p2 = torch.index_select(verts, 1, faces[:, 1])
    p3 = torch.index_select(verts, 1, faces[:, 2])

    # get edge lentgh
    e1 = p2 - p1
    e2 = p3 - p1
    e3 = p2 - p3

    edge_length = (torch.sum(e1 ** 2, -1).mean() + torch.sum(e2 ** 2, -1).mean() + torch.sum(e3 ** 2, -1).mean()) / 3.


    return edge_length