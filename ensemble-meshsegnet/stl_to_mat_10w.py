# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:38:26 2021

@author: lkh
"""
import os
import numpy as np
import torch
import torch.nn as nn
import vedo
import scipy.io as scio


mesh = vedo.load(r'F:\biyesheji\MeshSNet\dentalmesh(1)\surface_stl_10w\Sample_06.stl')
#print(mesh_path)
#print(i_sample)
# pre-processing: downsampling
mesh_d = mesh.clone()
predicted_labels_d = np.zeros([mesh_d.NCells(), 1], dtype=np.int32) # 生成一个9999行1列的矩阵

            # move mesh to origin
#print('\\tPredicting...')
cells = np.zeros([mesh_d.NCells(), 9], dtype='float32') # 生成一个9999行9列的矩阵
print(mesh_d.NCells())
for i in range(len(cells)): # len(cells)==9999
    cells[i][0], cells[i][1], cells[i][2] = mesh_d._polydata.GetPoint(mesh_d._polydata.GetCell(i).GetPointId(0)) # don't need to copy
    cells[i][3], cells[i][4], cells[i][5] = mesh_d._polydata.GetPoint(mesh_d._polydata.GetCell(i).GetPointId(1)) # don't need to copy
    cells[i][6], cells[i][7], cells[i][8] = mesh_d._polydata.GetPoint(mesh_d._polydata.GetCell(i).GetPointId(2)) # don't need to copy
                # 猜测：将mesh上3个点的XYZ坐标分别存入
            
original_cells_d = cells.copy()

mean_cell_centers = mesh_d.centerOfMass() # 取质心
cells[:, 0:3] -= mean_cell_centers[0:3]
cells[:, 3:6] -= mean_cell_centers[0:3]
cells[:, 6:9] -= mean_cell_centers[0:3]
            # 猜测：每个xyz坐标减去质心坐标

            # customized normal calculation; the vtk/vedo build-in function will change number of points
v1 = np.zeros([mesh_d.NCells(), 3], dtype='float32') # 9999*3的矩阵
v2 = np.zeros([mesh_d.NCells(), 3], dtype='float32')
v1[:, 0] = cells[:, 0] - cells[:, 3]
v1[:, 1] = cells[:, 1] - cells[:, 4]
v1[:, 2] = cells[:, 2] - cells[:, 5]
v2[:, 0] = cells[:, 3] - cells[:, 6]
v2[:, 1] = cells[:, 4] - cells[:, 7]
v2[:, 2] = cells[:, 5] - cells[:, 8]
mesh_normals = np.cross(v1, v2)
mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
mesh_normals[:, 0] /= mesh_normal_length[:]
mesh_normals[:, 1] /= mesh_normal_length[:]
mesh_normals[:, 2] /= mesh_normal_length[:]
mesh_d.addCellArray(mesh_normals, 'Normal') # mesh_normals法向量一个label，Normal

            # preprae input
points = mesh_d.points().copy()
            #for a in range(len(points)):
                #print(a)
points[:, 0:3] -= mean_cell_centers[0:3]
normals = mesh_d.getCellArray('Normal').copy() # need to copy, they use the same memory address
barycenters = mesh_d.cellCenters() # don't need to copy
barycenters -= mean_cell_centers[0:3]

            #normalized data
maxs = points.max(axis=0)
mins = points.min(axis=0)
means = points.mean(axis=0)
stds = points.std(axis=0)
nmeans = normals.mean(axis=0)
nstds = normals.std(axis=0)

for i in range(3):
    cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
    cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
    cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
    barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
    normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

X = np.column_stack((cells, barycenters, normals)) # 将cells,barycenters, normals矩阵拼接起来
scio.savemat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\stl_to_mat_10w\Sample_06_10w.mat', {'X': X})