import pyvista as pv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import os.path as osp
from torch_geometric.data import Data, Batch, Dataset
import torch_geometric.data
from model import *
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from torch_geometric.nn import Sequential, GCNConv
import os
import shutil
from feature_descriptors import *
import re
import vtk
import shutil 
import numpy as np
import os
import pyvista as pv
import sys
# sys.path.insert(0, '/home/pandu/Panresearch_local/PPP_Utility')
from os import listdir
import pdb
import torch_geometric
import torch
import mesh2graph_utility as mutl
from tqdm import tqdm
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader
import vtk
import os.path as osp
from torch.nn import Linear, Parameter
from torch_geometric.utils import degree

# for matplotlib
params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)


# the output label is pressure and wss. 
allmax = torch.tensor([129561.04688,1055.88342])
allmin = torch.tensor([92763.81250,0.55870])

def normalize(data, maxmin=None):
    # max, min  = torch.max(data ,dim = 0)[0], torch.min(data ,dim = 0)[0]
    allmax, allmin = maxmin[0], maxmin[1]
    norm_data = (data-allmin)/(allmax-allmin)*2-1
    return norm_data
def denormalize(data, maxmin = None):
    allmax, allmin  = maxmin[0], maxmin[1]
    denorm_data = (data+1)/2*(allmax-allmin)+allmin
    return denorm_data
# max_edge is for encoding the input nodal feature. 
max_edge = 0.00169

# preprocess the data 
from torch_geometric.data import Data
from torch_sparse import SparseTensor


# Overwrite "Data" class to ensure correct batching for pooling hierarchy
class MultiscaleData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None,
                 y=None, pos=None, normal=None, face=None, **kwargs):

        super(MultiscaleData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, pos=pos, normal=normal, face=face, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if 'batch' in key:
            return int(value.max()) + 1

        # Batch edges and polygons as before
        elif key == 'edge_index' or key == 'face':
            return self.num_nodes

        # Batch scales correctly
        elif 'scale' in key and ('cluster_map' in key or 'edge_index' in key):
            return self[key[:6] + '_cluster_map'].max() + 1

        elif 'scale' in key and 'sample_index' in key:
            if int(key[5]) == 0:
                return self.num_nodes
            else:
                return self['scale' + str(int(key[5]) - 1) + '_sample_index'].size(dim=0)

        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'edge_index' in key or 'face' in key:
            return -1
        else:
            return 0

import torch
from torch_cluster import radius_graph
from torch_geometric.nn import fps
import potpourri3d as pp3d
import numpy as np


class HeatSamplingCluster(object):
    """Compute a hierarchy of vertex clusters using farthest point sampling and the vector heat method. For correct
    batch creation, overwrite "torch_geometric.data.Data.__inc__()".

    Args:
        ratios (list): Ratios for farthest point sampling relative to the previous scale hierarchy.
        radii (list): Maximum radii for the creation of radius graphs on each scale.
        loop (bool): Whether to construct self-loop edges.
    """

    def __init__(self, ratios, radii, loop=False):
        self.ratios = ratios
        self.radii = radii
        self.loop = loop

    def __call__(self, data):

        vertices = data.pos
        for i, (ratio, radius) in enumerate(zip(self.ratios, self.radii)):

            if ratio == 1:
                cluster = torch.arange(vertices.shape[0])  # trivial cluster
                edges = radius_graph(vertices, radius, loop=self.loop)
                indices = torch.arange(vertices.shape[0])  # trivial indices

            else:
                # Sample a subset of vertices
                indices = fps(vertices, ratio=ratio)
                indices, _ = indices.sort()  # increases stability

                # Assign cluster indices to geodesic-nearest vertices via the vector heat method
                solver = pp3d.PointCloudHeatSolver(vertices)  # bless Nicholas Sharp the G.O.A.T.
                cluster = solver.extend_scalar(indices.tolist(), np.arange(indices.numel()))
                cluster = cluster.round().astype(np.int64)  # round away smoothing

                # Identify the corresponding vertex subset (discard dropped vertices)
                unique, cluster = torch.unique(torch.from_numpy(cluster), return_inverse=True)
                vertices = vertices[indices[unique]]

                # Connect vertices that are closer together than "radius"
                edges = radius_graph(vertices, radius, loop=self.loop)

                # Indices for scale visualisation
                indices = indices[unique]

            data['scale' + str(i) + '_cluster_map'] = cluster  # assigns a cluster number to each fine-scale vertex
            data['scale' + str(i) + '_edge_index'] = edges  # edges of the coarse-scale graph
            data['scale' + str(i) + '_sample_index'] = indices  # which fine-scale vertices are part of the coarse scale

        return data

    def __repr__(self):
        return '{}(ratios={}, radii={}, loop={})'.format(self.__class__.__name__, self.ratios, self.radii, self.loop)

# transforamtion of the dataset: 
# transforms 
def myTransform(data):
    # transform classes 
    # f2e = torch_geometric.transforms.FaceToEdge(remove_faces=False)
    gn = torch_geometric.transforms.GenerateMeshNormals()
    # HeatSamplingCluster([1., 0.3, 0.1], [0.04, 0.08, 0.2], loop=True),
    hsc = HeatSamplingCluster([1, 0.3, 0.1], [0.0014, 0.0028, 0.007], loop=True)
    
    # propagate:
    # print(data)
    # MultiscaleData(y=[4000], pos=[4000, 3], face=[3, 7939], stmdist=[4000, 1], featr2=[4000, 3, 3, 3], featr3=[4000, 3, 3, 3], featr4=[4000, 3, 3, 3], featr5=[4000, 3, 3, 3], featr6=[4000, 3, 3, 3], featr7=[4000, 3, 3, 3])
    # data = f2e(data)
    # print(data)
    # MultiscaleData(y=[4000], pos=[4000, 3], face=[3, 7939], stmdist=[4000, 1], featr2=[4000, 3, 3, 3], featr3=[4000, 3, 3, 3], featr4=[4000, 3, 3, 3], featr5=[4000, 3, 3, 3], featr6=[4000, 3, 3, 3], featr7=[4000, 3, 3, 3], edge_index=[2, 23878])
    data = gn(data)
    # print(data)
    # MultiscaleData(y=[4000], pos=[4000, 3], face=[3, 7939], stmdist=[4000, 1], featr2=[4000, 3, 3, 3], featr3=[4000, 3, 3, 3], featr4=[4000, 3, 3, 3], featr5=[4000, 3, 3, 3], featr6=[4000, 3, 3, 3], featr7=[4000, 3, 3, 3], edge_index=[2, 23878], norm=[4000, 3])
    data = hsc(data)
    # print(data)
    # MultiscaleData(y=[4000], pos=[4000, 3], face=[3, 7939], stmdist=[4000, 1], featr2=[4000, 3, 3, 3], featr3=[4000, 3, 3, 3], featr4=[4000, 3, 3, 3], featr5=[4000, 3, 3, 3], featr6=[4000, 3, 3, 3], featr7=[4000, 3, 3, 3], edge_index=[2, 23878], norm=[4000, 3], scale0_cluster_map=[4000], scale0_edge_index=[2, 4000], scale0_sample_index=[4000], scale1_cluster_map=[4000], scale1_edge_index=[2, 869], scale1_sample_index=[869], scale2_cluster_map=[869], scale2_edge_index=[2, 74], scale2_sample_index=[74])
    return data



snum = 89
my_dataset = []
# label_collect = []
for i in tqdm(range(snum)):
    temp_data = torch.load('../../data/vertex/processed/data_{:d}.pt'.format(i))
    # print(temp_data)
    # print(temp_data.edge_index[0,:].view(3,-1).size())
    # label_collect.append(temp_data.pos)
    my_data = MultiscaleData(pos=temp_data.pos,
                            y=normalize(temp_data.y, maxmin = [allmax, allmin]),  # for pressure and tau
                            #  y=temp_data.y[:,1:],  # for wss
                            edge_index=temp_data.edge_index,
                            # normal = temp_data.norm,
                            face = temp_data.face,
                            # face = torch.transpose(temp_data.edge_index[0,:].view(3,-1),0 ,1),
                            stmdist = temp_data.stmdist/torch.max(temp_data.stmdist),
                            featr2 = temp_data.featr2,
                            featr3 = temp_data.featr3,
                            featr4 = temp_data.featr4,
                            featr5 = temp_data.featr5,
                            featr6 = temp_data.featr6,
                            featr7 = temp_data.featr7,
                            )
    my_dataset.append(myTransform(my_data))
    
# label_collect = torch.stack(label_collect)
# label_max, label_min = torch.max(label_collect),torch.min(label_collect)
print('got total {:d} of samples in the dataset'.format(len(my_dataset)))


from compare import *
device = 'cuda'
model = CompareSAGE(out_channel=2) # for pressure 
model.to(device)

expi = 1
# this is pressure dataset
cut = int(0.9*snum)
data_size = snum
batch_size = int(snum/3)
print('batch size is:', batch_size)
device = 'cuda'
train_dataset = my_dataset[:cut]
valid_dataset = my_dataset[cut:]
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size = 1, shuffle=True)

lr = 0.01
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.linspace(0,6000,301,dtype = int), gamma=0.926)

def train(model, device, train_loader, optimizer, criterion, scheduler = None):
    # define trainloader
    
    model.train()
    train_ins_error = []
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        labels = batch.y
        # print('debug:',output,labels)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        train_ins_error.append(loss)
        return torch.mean(torch.stack(train_ins_error))


n_iters = 5001
train_ins_error = []
for epoch in tqdm(range(n_iters)):
    temp_error= train(model, device, train_loader, optimizer,criterion, scheduler = scheduler)
    train_ins_error.append(temp_error)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.plot(torch.stack(train_ins_error).detach().cpu().numpy())
plt.savefig('./result/trainerror.png')

pickid=1
mesh_label = my_dataset[pickid].y
mesh_label_denorm = denormalize(my_dataset[pickid].y, maxmin = [allmax, allmin])  # for pressure and tau
myinput = my_dataset[pickid].to(device)
mesh_output = model(myinput)

# print(mesh_output.size())
mesh_output_denorm = denormalize(mesh_output.detach().cpu(), maxmin = [allmax, allmin]) # for pressure and tau
print('relative error',torch.norm(mesh_output_denorm-mesh_label_denorm)/torch.norm(mesh_label_denorm))
template_mesh = pv.read('../../data/vertex/raw/mesh{:d}.vtp'.format(pickid))
# print(mesh_label_denorm.size(), mesh_output_denorm.size())
# torch.norm(mesh_label_denorm)
# torch.norm(mesh_output_denorm)
# print(mesh_output_denorm)
template_mesh.point_data['pp'] = np.array(mesh_output_denorm[:,0])
template_mesh.point_data['pt'] = np.array(mesh_output_denorm[:,1])
template_mesh.save('./result/mesh_output{:d}.vtp'.format(pickid))

pickid=88
mesh_label = my_dataset[pickid].y
mesh_label_denorm = denormalize(my_dataset[pickid].y, maxmin = [allmax, allmin])  # for pressure and tau
myinput = my_dataset[pickid].to(device)
mesh_output = model(myinput)

# print(mesh_output.size())
mesh_output_denorm = denormalize(mesh_output.detach().cpu(), maxmin = [allmax, allmin]) # for pressure and tau
print('relative error',torch.norm(mesh_output_denorm-mesh_label_denorm)/torch.norm(mesh_label_denorm))
template_mesh = pv.read('../../data/vertex/raw/mesh{:d}.vtp'.format(pickid))
# print(mesh_label_denorm.size(), mesh_output_denorm.size())
# torch.norm(mesh_label_denorm)
# torch.norm(mesh_output_denorm)
# print(mesh_output_denorm)



template_mesh.point_data['pp'] = np.array(mesh_output_denorm[:,0])
template_mesh.point_data['pt'] = np.array(mesh_output_denorm[:,1])
template_mesh.save('./result/mesh_output{:d}.vtp'.format(pickid))


