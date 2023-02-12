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


# path of the dataset 
path = "../../data/cases/dataset1/"
file_name_list = os.listdir(path)
file_name_list = sorted(file_name_list, key=lambda x:float(re.findall("(\d+)",x)[0]))
N = len(file_name_list); print('total num of data:',N)

def vtk2vtp(invtkfile, outvtpfile, binary=False):
    """What it says on the label"""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(invtkfile)
    reader.Update()

    reader2 = vtk.vtkGeometryFilter()
    reader2.SetInputData(reader.GetOutput())
    reader2.Update
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outvtpfile)
    if binary:
        writer.SetFileTypeToBinary()
    writer.SetInputConnection(reader2.GetOutputPort())
    writer.Update()
 
# collect the file name list 
files = []
mesh_list = []
for i in range(N):
    invtkfile = '../../data/cases/dataset1/'+file_name_list[i]+'/Mapped_Blade_Surface.vtk'
    outvtpfile = '../../data/cases/dataset1/'+file_name_list[i]+'/Mapped_Blade_Surface.vtp'
    # vtk2vtp(invtkfile, outvtpfile,binary=False)
    mesh_list.append(pv.read(outvtpfile))
    files.append(outvtpfile)

import vtk
from torch_geometric.utils import to_undirected


# functions for the vertexgraph transfer 
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

def vtk2GraphVertex(input_file, with_data = False, p_arrayid = 0, wss_arrayid = 1):
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

    nodal_stmdist = rotorblade.point_data['MappedDerivedVelocityMagnitude']  
    nodal_features = np.array(nodal_normals, dtype = np.float32)
    transform = torch_geometric.transforms.FaceToEdge(remove_faces=False)
    if with_data == True:
        nodal_labels = np.hstack((np.array(nodal_pressure,dtype = np.float32)[:,np.newaxis], np.array(nodal_wss,dtype = np.float32)[:,np.newaxis])) # four dimensions
        readme = {'nodal_features':'normalx, normaly, normalz',
                # 'edge_features':'d_coordinatex, d_coordinatey, d_coordinatez, d_coordinatenorm',
                'nodal_pos':'corrdinatex, corrdinatey, corrdinatez',
                'nodal_lables':'pressure, wss',
                'nodal_norms':'normalx, normaly, normalz',
                'nodal_stmdist': 'stmdist', 'face':'triangles'}
        mesh_graph = Data(x = torch.tensor(nodal_features), #edge_index = torch.tensor(edge_connectivity.T),
                        # edge_attr = torch.tensor(edge_features), 
                        y = torch.tensor(nodal_labels), 
                        pos = torch.tensor(points), norm = torch.tensor(nodal_normals), stmdist = torch.tensor(np.array(nodal_stmdist,dtype = np.float32)[:,np.newaxis]),
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



# pick one case for the transformation 
pick_id = 0
demo,_ = vtk2GraphVertex(mesh_list[pick_id], with_data = True) # x is pos, y is pressure and tau
fd = FeatureDescriptors(r = ) # fd adds features in to the graph data 
demo = fd(demo)
# print(demo)

# tmp = [demo.featr3[:, 0][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
#         demo.featr3[:, 1][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
#         demo.featr3[:, 2].reshape(-1, 9)]
# demo.x = torch.hstack((torch.hstack(tmp), demo.stmdist))
# print(demo)

# #  calculate the max and min edge of the cases
# max_edges = []
# min_edges = []
# for i, ele in enumerate(mesh_list): 
#     if i == 0: 
#         graph,_ = vtk2GraphVertex(ele)
#         min_edge = torch.argmin(torch.norm(graph.pos[graph.edge_index[0]]-graph.pos[graph.edge_index[1]],dim=1)); print(min_edge) #5310
#         print(graph.edge_index[:,min_edge]) #[896, 954]
# #     max_edge = torch.max(torch.norm(graph.pos[graph.edge_index[0]]-graph.pos[graph.edge_index[1]],dim=1))
# #     min_edge = torch.min(torch.norm(graph.pos[graph.edge_index[0]]-graph.pos[graph.edge_index[1]],dim=1))
# #     max_edges.append(max_edge)
# #     min_edges.append(min_edge)

# # all_max_edge = torch.max(torch.stack(max_edges)); print(all_max_edge)
# # all_min_edge = torch.min(torch.stack(min_edges)); print(all_min_edge)  # max = 0.0017; min = 4.66e-06

def normalize(data):
    max, min  = torch.max(data ,dim = 0)[0], torch.min(data ,dim = 0)[0]
    print('max, min', max,min)
    norm_data = (data-min)/(max-min)*2-1
    return norm_data, [max,min]
def denormalize(data, maxmin = None):
    max, min  = maxmin[0], maxmin[1]
    denorm_data = (data+1)/2*(max-min)+min
    return denorm_data


# normalize the datalabel 
demo.ydenorm = demo.y

demo.y, maxmin = normalize(demo.y) ;print('normalized data',demo.y)
# # demo = my_dataset[0]
# demo.edge_index.size() # torch.Size([2, 23878])
# calculate the norm of edges 
xij = torch.stack([demo.pos[demo.edge_index[1,i]]-demo.pos[demo.edge_index[0,i]] for i in range(demo.edge_index.size()[1])])
# calculate the norm of edges 
xijabs = torch.stack([torch.norm(demo.pos[demo.edge_index[1,i]]-demo.pos[demo.edge_index[0,i]] ) for i in range(demo.edge_index.size()[1])])
demo.edge_attr = torch.cat((xij, xijabs.unsqueeze(1)),dim = 1); print('edge attr size',demo.edge_attr.size())
# print('edge attr:',demo.edge_attr)

class My_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, improved=False,
                 cached=False, bias=True, fine_marker_dict=None, GCNtype = 'MGN'):
        super().__init__()
        self.GCNtype = GCNtype
        channels = [in_channels]+hidden_channels+[out_channels]


        convs = []
        for i in range(len(channels)-1):
            if GCNtype == 'MGN':
                convs.append(MGNGraphNet(3, in_channels=channels[i], in_edge_channels=4, out_channels=channels[i+1],hidFeature=128,aggr='mean'))
            if GCNtype == 'GCN':
                convs.append(GCNConv(in_channels=channels[i], out_channels=channels[i+1]))
        self.convs = torch.nn.ModuleList(convs)

    def forward(self, data):
        # if read in data, the inputs features will be a falttened feature vector and also the stmdist
        tmp = [data.featr2[:, 0][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
        data.featr2[:, 1][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
        data.featr2[:, 2].reshape(-1, 9)]
        out = torch.hstack((torch.hstack(tmp), data.stmdist))
        
        for i, conv in enumerate(self.convs[:-1]):
            if self.GCNtype == 'MGN':
                out = conv(out, data.edge_index,data.edge_attr) # .unsqueeze
            if self.GCNtype == 'GCN':
                # print('input size:',out.size())
                # print('edge index size:',data.edge_index.size())
                out = conv(out, data.edge_index)
            out = F.relu(out)
        # print('debug8',out.size())
        
        if self.GCNtype == 'MGN':
            out = self.convs[-1](out, data.edge_index,data.edge_attr) # .unsqueeze
        if self.GCNtype == 'GCN':
            out = self.convs[-1](out, data.edge_index) # .unsqueeze
        return out




# model = My_model(22,[256,512,512],2,GCNtype = 'MGN')  # exp1
# model = My_model(22,[256],2,GCNtype = 'MGN')  # exp0
# model = My_model(22,[256,256],2,GCNtype = 'MGN')  # exp2
# model = My_model(22,[256,256,256],2,GCNtype = 'MGN')  # exp3

# model = My_model(22,[512],2,GCNtype = 'MGN')  # exp4
# model = My_model(22,[512,512],2,GCNtype = 'MGN')  # exp5
# model = My_model(22,[512,512,512],2,GCNtype = 'MGN')  # exp6


# # model = My_model(22,[256,4],GCNtype = 'GCN')  # output 4 cuz shearstress xyz and p
# model = My_model(22,[256],2,GCNtype = 'GCN')  # exp1
# model = My_model(22,[256,256],2,GCNtype = 'GCN')  # exp2
# model = My_model(22,[256,256,256],2,GCNtype = 'GCN')  # exp3
model = My_model(22,[256,256,256,256],2,GCNtype = 'GCN')  # exp4

model.to('cuda')
epochs = 2001
lr = 0.01

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.linspace(0,3000,120,dtype = int), gamma=0.926)
error = []

# output_dir = 'MGN_exp5'
output_dir = 'GCN_exp4'
if osp.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir+'/figs')


demo = demo.to('cuda') 
for i in tqdm(range(epochs)):
    optimizer.zero_grad()
    out = model(demo)
    loss = criterion(out, demo.y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    error.append(loss)
    if i%1000==0:
        error1 = torch.stack(error)
        fig, ax = plt.subplots(figsize=(10,8))
        lw=3
        ax.plot(error1.detach().cpu().numpy(),color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='')
        ax.set_yscale('log')
        fig.savefig(output_dir+'/figs/train_error_epoch{:d}.png'.format(i),bbox_inches='tight' )
        
# print(error)
error1 = torch.stack(error)

torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': error1.detach().cpu(),
            }, output_dir+"/checkpoint.pt")

checkpoint = torch.load(output_dir+"/checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])

demo.to('cpu')
model.to('cpu')
out = denormalize(model(demo), maxmin)

prediction = out.detach().cpu().numpy()
label = demo.ydenorm.detach().cpu().numpy()

relativeerror = np.linalg.norm(prediction-label)/np.linalg.norm(label)

prediction_norm = model(demo).detach().cpu().numpy()
label_norm = demo.y.detach().cpu().numpy()
norm_relativeerror = np.linalg.norm(prediction_norm-label_norm)/np.linalg.norm(label_norm)
constnorm_relativeerror = np.linalg.norm(prediction_norm-label_norm)/np.linalg.norm(label_norm)

np.save(output_dir+"/relativeerror",relativeerror); print('relativeerror', relativeerror); print(norm_relativeerror)
with open(output_dir+"/relativeerror.txt", 'w') as f:
    f.write("relative error: {:.3f}".format(relativeerror))
    
    
mesh = pv.read(files[pick_id])
mesh.point_data['l_p'] = label[:,0]
mesh.point_data['p_p'] = prediction[:,0]
mesh.point_data['d_p'] = prediction[:,0]-label[:,0]
mesh.point_data['l_tau'] = label[:,1]
mesh.point_data['p_tau'] = prediction[:,1]
mesh.point_data['d_tau'] = prediction[:,1]-label[:,1]
mesh.save(output_dir+'/mesh.vtk')



# # # CUDA_VISIBLE_DEVICES=2 python test_fit.py



