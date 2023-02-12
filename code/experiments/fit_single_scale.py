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
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader

params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)

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

## dataset preparation 
# prepare the data 
snum=89
my_dataset = []
# label_collect = []
# predict pressure only 
for i in tqdm(range(snum)):
    temp_data = torch.load('../../data/vertex/processed/data_{:d}.pt'.format(i))

    my_data = Data(pos=temp_data.pos,
                    y=normalize(temp_data.y, maxmin = [allmax, allmin])[:,0],  # for pressure only
                    #  y=temp_data.y[:,1:],  # for wss
                    edge_index=temp_data.edge_index,
                    face = temp_data.face,
                    stmdist = temp_data.stmdist/torch.max(temp_data.stmdist),
                    featr = temp_data.featr,
                    featr2 = temp_data.featr2,
                    featr3 = temp_data.featr3,
                    featr4 = temp_data.featr4,
                    featr5 = temp_data.featr5,
                    featr6 = temp_data.featr6,
                    )
    my_dataset.append(my_data)

cut = int(0.9*snum)
data_size = snum
# batch_size = int(snum/3)
batch_size = 1
# print('batch size is:', batch_size)
device = 'cuda'
train_dataset = my_dataset[:cut]
valid_dataset = my_dataset[cut:]
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size = 1, shuffle=False)
train_valid_loader = DataLoader(train_dataset, batch_size = 1, shuffle=False)

# create model 

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
    
model = My_model(24,[256],1,GCNtype = 'GCN')  # exp3

model.to('cuda')

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

epochs = 3001
lr = 0.01

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.linspace(0,3000,120,dtype = int), gamma=0.926)
error = []

output_dir = 'batch_GCN_P_exp1'
if osp.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir+'/figs')

#start training 
train_ins_error = []
for epoch in tqdm(range(epochs)):
    temp_error= train(model, device, train_loader, optimizer,criterion, scheduler = scheduler)
    train_ins_error.append(temp_error)
    if epoch%1000==0:
        error1 = torch.stack(train_ins_error)
        fig, ax = plt.subplots(figsize=(10,8))
        lw=3
        ax.plot(error1.detach().cpu().numpy(),color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='')
        ax.set_yscale('log')
        fig.savefig(output_dir+'/figs/train_error_epoch{:d}.png'.format(epoch),bbox_inches='tight' )


error1 = torch.stack(train_ins_error)
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': error1.detach().cpu(),
            }, output_dir+"/checkpoint.pt")
print(error)

#start testing 
pick_ids = [i for i in range(89-cut)]
test_norm_error = []
test_denorm_error = []

model.to('cpu')
for i,batch in enumerate(train_valid_loader): 
    label  = batch.y
    output = model(batch)
    # print(label.detach().numpy(), output.detach().numpy())
    temp_norm_error = torch.norm(label-output)/torch.norm(label)
    label_denorm = denormalize(label, [allmax, allmin])
    output_denorm = denormalize(output, [allmax, allmin])
    temp_denorm_error = torch.norm(label_denorm-output_denorm)/torch.norm(label_denorm)
    test_norm_error.append(temp_norm_error)
    test_denorm_error.append(temp_denorm_error)
    
    # if i==pick_id:
    if i in pick_ids:
        mesh = pv.read('../../data/vertex/raw/mesh{:d}.vtp'.format(i))
        # mesh = pv.read('../../data/vertex/raw/mesh{:d}.vtp'.format(i+cut))
        mesh.point_data['l_p'] = label_denorm[:,0]
        mesh.point_data['p_p'] = output_denorm[:,0].detach()
        mesh.point_data['d_p'] = output_denorm[:,0].detach()-label_denorm[:,0]
        mesh.point_data['l_tau'] = label_denorm[:,1]
        mesh.point_data['p_tau'] = output_denorm[:,1].detach()
        mesh.point_data['d_tau'] = output_denorm[:,1].detach()-label_denorm[:,1]
        mesh.save(output_dir+'/train_mesh_{:d}.vtk'.format(i))
        # mesh.save(output_dir+'/mesh_{:d}.vtk'.format(i))

test_norm_error = torch.stack(test_norm_error)
test_denorm_error = torch.stack(test_denorm_error)

np.savez(output_dir+"/relativeerror",test_norm_error.detach().numpy(),test_denorm_error.detach().numpy())

with open(output_dir+"/relativeerror.txt", 'w') as f:
    f.write("relative norm error: {:.3f} /n".format(torch.mean(test_norm_error)))
    f.write("relative denorm error: {:.3f} /n".format(torch.mean(test_denorm_error)))