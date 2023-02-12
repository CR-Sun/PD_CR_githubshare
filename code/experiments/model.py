from torch.nn.functional import layer_norm
from torch_geometric.nn.conv import MessagePassing
import torch
import pdb
import torch_geometric
from torch_geometric.nn import NNConv

class MLP(torch.nn.Module):
	def __init__(self,nIn,nOut,Hidlayer, withReLU):
		# nIn : input channels 
		# nOut : output channels
		# Hidlayer: a list of hidlayer dims 
		# withReLU : True or False in terms of adding ReLU or not 
  
		super(MLP, self).__init__()
		numHidlayer=len(Hidlayer)  # number of Hidlayers
		net=[]
		net.append(torch.nn.Linear(nIn,Hidlayer[0])) # append first layer
		if withReLU:
			net.append(torch.nn.ReLU())
		for i in range(0,numHidlayer-1):
			net.append(torch.nn.Linear(Hidlayer[i],Hidlayer[i+1]))
			if withReLU:
				net.append(torch.nn.ReLU())
		net.append(torch.nn.Linear(Hidlayer[-1],nOut))# append last layer
		self.mlp=torch.nn.Sequential(*net)
	def forward(self,x):
		return self.mlp(x)

class MGNLayer(MessagePassing):
    # MGN layers
    # the in_edge_channels should be equal to out_edge_channels!!!
	def __init__(self, in_channels,
					   out_channels,
					   in_edge_channels,
					   out_edge_channels,
				 	   **kwargs):
		super(MGNLayer, self).__init__(aggr='mean', **kwargs)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.in_edge_channels = in_edge_channels
		self.out_edge_channels = out_edge_channels
		self.nnEdgeHidNodeNum = [128, 128]
		self.nnNodeHidNodeNum = [128, 128]
		
		self.normEdge=torch.nn.LayerNorm(self.out_edge_channels)
		self.nnEdge=MLP(2*self.in_channels+self.in_edge_channels,
						self.out_edge_channels, self.nnEdgeHidNodeNum,withReLU=True)
		self.normNode=torch.nn.LayerNorm(self.out_channels)
		self.nnNode=MLP(self.in_channels+self.out_edge_channels,
						self.out_channels, self.nnNodeHidNodeNum,withReLU=True)
		
	def forward(self, inputList):
		x0, edge_index, edge_attr = inputList # here x0 is the input nodal feature, edge_index is the edge connection, edge_attr is the edge features 
		out=self.propagate(edge_index, x=x0, edge_attr=edge_attr) # do one time message passing 
		x=x0 + self.normNode(self.nnNode((torch.cat([x0, out], dim=1)))) # this is the resnet?
		outputList = [x, edge_index, self.edge_attr]
		return outputList

	def message(self, x_i, x_j, edge_attr):
		tmp_edge = self.nnEdge((torch.cat([x_i, x_j, edge_attr], dim=1)))
		tmp_edge = self.normEdge(tmp_edge)
		self.edge_attr=tmp_edge + edge_attr
		return tmp_edge


class NNLayer(torch.nn.Module):
	def __init__(self,in_edge_channels,hidFeature):
		super(NNLayer, self).__init__()
		self.relu=torch.nn.ReLU() 
		#NN = torch.nn.Linear(in_edge_channels, hidFeature*hidFeature)
		NN = MLP(in_edge_channels, hidFeature*hidFeature,[128,128],withReLU=True)
		layer_norm = torch.nn.LayerNorm(hidFeature*hidFeature)
		LIST = [NN,layer_norm] 
		self.NN = torch.nn.Sequential(*LIST)
		self.conv=NNConv(in_channels=hidFeature,
						  out_channels=hidFeature,
					      nn = self.NN,
						  aggr='mean').to('cuda').float()
		self.convNorm=torch_geometric.nn.LayerNorm(hidFeature)
	def forward(self,inputList):  # here the input feature dim should be equal to the hid channels!!!
		x, edge_index, edge_attr = inputList
		x0 = x
		x = self.conv(x, edge_index, edge_attr)
		x = self.relu(x)
		x =self.convNorm(x)
		x = x0 + x
		outputList = [x, edge_index, edge_attr]
		return outputList

class NNGraphNet(torch.nn.Module):
    
	def __init__(self,n_hidlayer,
					  in_channels=5,
					  in_edge_channels=3,
					  out_channels=3,
					  hidFeature=128,
					  aggr='mean',):
		super(NNGraphNet, self).__init__()
		self.relu=torch.nn.ReLU()
		hidNodeNums = [hidFeature,
					   hidFeature,
					   hidFeature]
		self.in_channels=in_channels
		self.out_channels=out_channels
		# LocalEncoder
		self.encoder = MLP(in_channels, hidFeature, hidNodeNums,True)
		self.encoderNorm=torch_geometric.nn.LayerNorm(hidFeature)

		hidLayerList = []
		for i in range(n_hidlayer):
			hidLayerList.append(NNLayer(in_edge_channels, hidFeature))
		self.hiddenlayers = torch.nn.Sequential(*hidLayerList)
		self.decoder = MLP(hidFeature, out_channels, hidNodeNums,True)


	def forward(self, x, edge_index, edge_attr):
		
		x = self.encoder(x)
		x =self.encoderNorm(x)
		inputList = [x, edge_index, edge_attr]
		x,_,_ = self.hiddenlayers(inputList)
		x = self.decoder(x)
		# layer norm
		return x
		
class MGNGraphNet(torch.nn.Module):
	def __init__(self,n_hidlayer,
					  in_channels=5,
					  in_edge_channels=3,
					  out_channels=3,
					  hidFeature=128,
					  aggr='mean',):
		super(MGNGraphNet, self).__init__()
		self.relu=torch.nn.ReLU()
		hidNodeNums = [hidFeature,
					   hidFeature,
					   hidFeature]
		self.in_channels=in_channels
		self.out_channels=out_channels
		# LocalEncoder
		self.encoderNode = MLP(in_channels, hidFeature, hidNodeNums,True)
		self.encoderNodeNorm=torch_geometric.nn.LayerNorm(hidFeature)

		self.encoderEdge = MLP(in_edge_channels, hidFeature, hidNodeNums,True)
		self.encoderEdgeNorm=torch_geometric.nn.LayerNorm(hidFeature)

		hidLayerList = []
		for i in range(n_hidlayer):
			hidLayerList.append(MGNLayer(hidFeature, hidFeature,hidFeature,hidFeature))
		self.hiddenlayers = torch.nn.Sequential(*hidLayerList)
		self.decoder = MLP(hidFeature, out_channels, hidNodeNums,True)


	def forward(self, x, edge_index, edge_attr):
		
		# print('debug1',x.size())
		x = self.encoderNode(x)
		# print('debug2',x.size())
		x =self.encoderNodeNorm(x)
		# print('debug3',x.size())
		# print('debug3 edge attr size:',edge_attr.size())
		edge_attr =self.encoderEdge(edge_attr)
		# print('debug4',x.size())
		edge_attr =self.encoderEdgeNorm(edge_attr)
		# print('debug5',x.size())

		inputList = [x, edge_index, edge_attr]
		x,_,_ = self.hiddenlayers(inputList)
		# print('debug6',x.size())
		x = self.decoder(x)
		# print('debug7',x.size())

		return x
