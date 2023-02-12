from torch_geometric.data import Dataset, Data
from feature_descriptors import *
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
params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)
import os
import shutil
from feature_descriptors import *
import re
import vtk
# path = "../../data/cases/dataset1/"
# file_name_list = os.listdir(path)
# file_name_list = sorted(file_name_list, key=lambda x:float(re.findall("(\d+)",x)[0]))
# N = len(file_name_list); print('total num of data:',N)

# filenames = []
# mesh_list = []
# for i in range(N):
#     outvtpfile = '../../data/cases/dataset1/'+file_name_list[i]+'/Mapped_Blade_Surface.vtp'
#     # vtk2vtp(invtkfile, outvtpfile,binary=False)
#     mesh_list.append(pv.read(outvtpfile))
#     filenames.append(outvtpfile)

# for i in range(N):
#     mesh_list[i].save('../../data/vertex/raw/mesh{:d}.vtp'.format(i))



def cellArrayDivider(input_array):
    N = len(input_array)
    cursor = 0
    head_id = []
    segs = []
    
    while(cursor<N):
        head_id.append(cursor)
        segs.append(input_array[cursor+1:cursor+input_array[cursor]+1])
        cursor = cursor+input_array[cursor]+1
        # print(cursor)
    return segs

def readvtp(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()

    return reader.GetOutput()

def cyl_coor_pos(points):
    r = np.linalg.norm(points[:,1:],axis = 1)
    theta = np.arctan(points[:,1], points[:,2]) + np.pi*(points[:,1]<0)
    z = points[:,2]
    def normalize1d(data):
        return (data-np.min(data))/(np.max(data)-np.min(data))*2-1
    cyl_pos = np.array([normalize1d(r),normalize1d(theta),normalize1d(z)]).T
    return cyl_pos 


def vtk2GraphVertex(input_file, with_data = False, p_arrayid = 0, wss_arrayid = 1):
    """
    input: mesh file 
    output: graph 
    process: compute vertex normals, create edge_index, add pressure and wss data, add stmdist 
    """
    
    if type(input_file) == str:
        # aorta_vtk = readvtp(input_file)
        rotorblade = pv.read(input_file)
    else:
        rotorblade = input_file
    segs = np.array(cellArrayDivider(rotorblade.faces))
    points = np.array(rotorblade.points,dtype = np.float32)
    nodal_normals = np.array(rotorblade.point_normals,dtype = np.float32)
    if with_data == True:
        nodal_pressure = rotorblade.point_data['MappedAbsoluteTotalPressure']
        nodal_wss = rotorblade.point_data['MappedTau']

    nodal_stmdist = cyl_coor_pos(rotorblade.points)  # (r,theta,z) normalized 
    # nodal_stmdist = rotorblade.point_data['MappedDerivedVelocityMagnitude']  
    nodal_features = np.array(nodal_normals, dtype = np.float32)
    transform = torch_geometric.transforms.FaceToEdge(remove_faces=False) ### undirected graph, meaning eij exists for every eji
    if with_data == True:
        nodal_labels = np.hstack((np.array(nodal_pressure,dtype = np.float32)[:,np.newaxis], np.array(nodal_wss,dtype = np.float32)[:,np.newaxis])) # four dimensions
        readme = {'nodal_features':'normalx, normaly, normalz',
                # 'edge_features':'d_coordinatex, d_coordinatey, d_coordinatez, d_coordinatenorm',
                'nodal_pos':'corrdinatex, corrdinatey, corrdinatez',
                'nodal_labels':'pressure, wss',
                'nodal_norms':'normalx, normaly, normalz',
                'nodal_stmdist': 'stmdist', 'face':'triangles'}
        mesh_graph = Data(x = torch.tensor(nodal_features), #edge_index = torch.tensor(edge_connectivity.T),
                        # edge_attr = torch.tensor(edge_features), 
                        y = torch.tensor(nodal_labels), 
                        pos = torch.tensor(points), norm = torch.tensor(nodal_normals), stmdist = torch.tensor(np.array(nodal_stmdist,dtype = np.float32)),
                        face = torch.tensor(np.array(segs.T)))
    else:
        readme = {'nodal_features':'normalx, normaly, normalz',
                # 'edge_features':'d_coordinatex, d_coordinatey, d_coordinatez, d_coordinatenorm',
                'nodal_pos':'corrdinatex, corrdinatey, corrdinatez',
                'nodal_norms':'normalx, normaly, normalz',
                'nodal_stmdist': 'stmdist', 'face':'triangles'}
        mesh_graph = Data(x = torch.tensor(nodal_features),#edge_index = torch.tensor(edge_connectivity.T),
                        # edge_attr = torch.tensor(edge_features),
                        pos = torch.tensor(points), norm = torch.tensor(nodal_normals),stmdist = torch.tensor(np.array(nodal_stmdist,dtype = np.float32)[:,np.newaxis]),
                        face = torch.tensor(np.array(segs.T)))

    mesh_graph_transformed = transform(mesh_graph)
    return mesh_graph_transformed, readme


class MyCustomDatasetVertex(Dataset):
    def __init__(self,root,maxedge = 1):
        self.filename = ['mesh{:d}.vtp'.format(i) for i in range(89)]
        self.maxedge = maxedge
        super(MyCustomDatasetVertex, self).__init__(root)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):

        return ['data_{:d}.pt'.format(i) for i in range(89)]
    def download(self):
        pass

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # converting vtp to graph
            data,readme = vtk2GraphVertex(raw_path,with_data = True)
            # adding features 
            fd = FeatureDescriptors(self.maxedge)
            data = fd(data)
            # print('saving pt file for ',idx)
            torch.save(data, os.path.join(self.processed_dir, 'data_{:d}.pt'.format(idx)))
            idx += 1

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data

# calculated the edge length of th meshes
max_edge_list = []
min_edge_list = []
# get the max and min of the label data for nomalization 
mymax,mymin = [],[]

for i in range(89):
    mesh = pv.read('../../data/vertex/raw/mesh{:d}.vtp'.format(i))
    data,_ = vtk2GraphVertex(mesh,with_data = True)
    max_edge = torch.max(torch.norm(data.pos[data.edge_index[0]]-data.pos[data.edge_index[1]],dim=1))
    min_edge = torch.min(torch.norm(data.pos[data.edge_index[0]]-data.pos[data.edge_index[1]],dim=1))
    # print('max and min edges:',max_edge, min_edge) #tensor(0.0014) tensor(9.0898e-06)
    max_edge_list.append(max_edge)
    min_edge_list.append(min_edge)
    
    # print(data.y)
    mymax.append(torch.max(data.y,dim=0)[0])
    mymin.append(torch.min(data.y,dim=0)[0])

all_max_edge = 0.00169
# all_max_edge = torch.max(torch.stack(max_edge_list));print('max edge:',all_max_edge)
# all_min_edge = torch.max(torch.stack(min_edge_list));print('min_edge:',all_min_edge)
# mymax = torch.stack(mymax)
# mymin = torch.stack(mymin)
# allmax = torch.max(mymax,dim=0)[0]
# allmin = torch.min(mymin,dim=0)[0]

# with open("../../data/vertex/summary/datasetinfo.txt", 'w') as f:
#     f.write("all max edges: {:.5f} \n".format(all_max_edge))
#     f.write("all min edges: {:.5f} \n".format(all_min_edge))
#     f.write("y [pressure, wss] max: {:.5f},{:.5f} \n".format(allmax[0], allmax[1]))
#     f.write("y [pressure, wss] min: {:.5f},{:.5f} \n".format(allmin[0], allmin[1]))

# create dataset
test_dataset = MyCustomDatasetVertex('../../data/vertex/', maxedge = all_max_edge)

# test_dataset = MyCustomDatasetVertex('../../data/vertex/', maxedge = 1)
print(test_dataset[0])

# you have to delete the old 'processed' folder to generate new data/ 
