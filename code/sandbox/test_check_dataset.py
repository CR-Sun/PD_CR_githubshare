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


i=0
data = torch.load('../../data/vertex/processed/data_{:d}.pt'.format(i))
print(data)