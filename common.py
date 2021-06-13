import dgl
import numpy as np

def write_ply(vertices, path="test.ply"):
    ply_num = vertices.shape[0]
    ply_header = "ply\nformat ascii 1.0\nelement vertex {0}\nproperty float x\nproperty float y\nproperty float z\nend_header\n".format(
        ply_num)
    ply = open(path, "w+")
    ply.write(ply_header)
    for v in vertices:
        ply.write("{0} {1} {2}\n".format(v[0], v[1], v[2]))


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def build_mesh_graph(mesh):
    g = dgl.DGLGraph()
    g.add_nodes(mesh.vs.shape[0])
    edge_list = [tuple(eitem) for eitem in mesh.es]
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    return g


def build_graph(v, e):
    # print(type(v), type(e))
    # print(e.shape, v.shape)
    # e = e.numpy()[0]
    # v = v.numpy()[0]

    g = dgl.DGLGraph()
    g.add_nodes(v.shape[0])
    edge_list = [tuple(item) for item in e]
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    return g
