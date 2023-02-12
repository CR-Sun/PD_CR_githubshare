from gettext import dpgettext
import pyvista as pv
import numpy as np
import os
import vtk
import torch
from torch_geometric.data import Data

def readvtp(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()

    return reader.GetOutput()


def vtk2graph(input_file, with_data = False, p_arrayid = 0, wss_arrayid = 1):
    if type(input_file) == str:
        # aorta_vtk = readvtp(input_file)
        aorta_vtk = pv.read(input_file) # pv can readin vtk file as well which is better. 
    else:
        aorta_vtk = input_file
    aorta_vtk.BuildLinks()
    nc = aorta_vtk.GetNumberOfCells()
    # print(nc) #8100
    cell_normals = np.zeros((nc,3),dtype = np.float32)
    cell_centroids = np.zeros((nc,3),dtype = np.float32)
    for ci in range(nc):
        cell = aorta_vtk.GetCell(ci)
        pts = cell.GetPoints()
        cell.ComputeNormal(pts,3,(0,1,2),cell_normals[ci])
        cell.ComputeCentroid(pts,(0,1,2),cell_centroids[ci])

    edge_connectivity = [] # angle on edge 
    edge_angle = [] # angle on edge 
    edge_d_coordinate = [] # difference of coordinates on the dege 
    edge_length = [] # edge length 
    nodal_angle = []
    nodal_normal = []
    nodal_pos = []
    
    if with_data == True:
        nodal_pressure = []
        nodal_wss = []
    
    
    for ci in range(nc): #nc
        # print('cellid:'+str(ci))
        cell = aorta_vtk.GetCell(ci)
        angle = np.array([0.0,0.0,0.0])
        for ei in range(3): #looping through edges of one element , 3 cuz triangle element
            # print('edge',ei)
            edge = cell.GetEdge(ei)
            neighbour_cell_ids = vtk.vtkIdList()
            aorta_vtk.GetCellNeighbors(ci,edge.GetPointIds(),neighbour_cell_ids)
            nnci = neighbour_cell_ids.GetNumberOfIds()
            # print(nnci)
            if nnci == 0: # found boundary 
                angle[ei] = 0
            elif nnci == 1: # found neighbor cell, compute angle between two faces.
                nci = neighbour_cell_ids.GetId(0)

                # print(cell_normals[ci],cell_normals[nci])
                sign = np.sign(np.cross(cell_normals[ci], cell_normals[nci]))*np.sign(np.array(edge.GetPoints().GetPoint(1))-np.array(edge.GetPoints().GetPoint(0)))
                dot_product =(cell_normals[ci]*cell_normals[nci]).sum()
                # numerical_tolerance = 0.0001
                if dot_product<0:
                    print('Warning the angle is larger than 90 degrees.')
                elif dot_product>1:
                    print('found cross product={}, setting it to 1'.format(dot_product))
                    dot_product =1
                temp_angle = sign[0]*np.arccos(dot_product)
                angle[ei] = temp_angle
                if nci>ci:  # if the neighboring id is bigger than current id, record the connection and the angle
                    edge_connectivity.append((ci,nci))
                    edge_angle.append(temp_angle)
                    edge_d_coordinate.append(cell_centroids[nci]-cell_centroids[ci])
                    edge_length.append(np.linalg.norm(cell_centroids[nci]-cell_centroids[ci]))
            else:
                # print(neighbour_cell_ids.GetId(0),neighbour_cell_ids.GetId(1))
                print('Error:found more than one neighbouring elements on one edge')
                break
        nodal_angle.append(angle)
        nodal_normal.append(cell_normals[ci])
        nodal_pos.append(cell_centroids[ci])
        if with_data == True:
            nodal_pressure.append(aorta_vtk.GetCellData().GetArray(p_arrayid).GetValue(ci)) # get the pressure
            nodal_wss.append(aorta_vtk.GetCellData().GetArray(wss_arrayid).GetTuple(ci)) # 3 dimensional
    
    nodal_pos = np.array(nodal_pos,dtype = np.float64)
    nodal_features = np.hstack((np.array(nodal_angle),np.array(nodal_normal))) # six dimension
    edge_connectivity = np.array(edge_connectivity)
    # print(nodal_pos.shape)
    # print(np.array(edge_d_coordinate).shape , np.array(edge_length)[:,np.newaxis].shape, np.array(edge_angle).shape)
    edge_features = np.hstack((np.array(edge_d_coordinate),np.array(edge_length)[:,np.newaxis],np.array(edge_angle)[:,np.newaxis])) # five dimensions
    if with_data == True:
        nodal_labels = np.hstack((np.array(nodal_pressure)[:,np.newaxis], np.array(nodal_wss))) # four dimensions
        readme = {'nodal_features':'angle1, angle2, angle3, normalx, normaly, normalz',
                'edge_features':'d_coordinatex, d_coordinatey, d_coordinatez, d_coordinatenorm, edge_angle',
                'nodal_pos':'corrdinatex, corrdinatey, corrdinatez',
                'nodal_lables':'pressure, wssx, wssy, wssz'}
        mesh_graph = Data(x = torch.tensor(nodal_features),edge_index = torch.tensor(edge_connectivity),
                        edge_attr = torch.tensor(edge_features), y = torch.tensor(nodal_labels), pos = torch.tensor(nodal_pos))
    else:
        readme = {'nodal_features':'angle1, angle2, angle3, normalx, normaly, normalz',
                'edge_features':'d_coordinatex, d_coordinatey, d_coordinatez, d_coordinatenorm, edge_angle',
                'nodal_pos':'corrdinatex, corrdinatey, corrdinatez'}
        mesh_graph = Data(x = torch.tensor(nodal_features),edge_index = torch.tensor(edge_connectivity),
                        edge_attr = torch.tensor(edge_features), pos = torch.tensor(nodal_pos))
    return mesh_graph, readme

    