# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:45:06 2021

@author: lkh
"""
import vedo
import scipy.io as scio
import numpy as np
import os
#mesh = vedo.load(r'F:\biyesheji\MeshSNet\dentalmesh(1)\dentalmesh1wstl\Sample_02.stl')
a=scio.loadmat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\zuijinlin_mat_1w\Sample_06_1w.mat')
X=a['X']
#meshpoint=mesh.getPointArray()

num_row=9999
num_line=9
point_row=0
point_line=0
point_mat=np.zeros([3*9999, 3], dtype=np.float64)
for i_row in range(num_row):
    for i_line in range(num_line):
        point_mat[point_row][point_line]=X[i_row][i_line]
        point_line+=1
        if point_line==3:
            point_line=0
            point_row+=1
b=scio.loadmat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\zuijinlin_predict_labels\Sample_06_predicted_labels.mat')   
labels=b['labels']
labels_mat=np.zeros([3*9999, 1], dtype=np.int32)
num_labels=0
labels_row=3*9999
count=0
for i_labels in range(labels_row):
    labels_mat[i_labels][0]=labels[num_labels][0]
    count+=1
    if count==3:
        num_labels+=1
        count=0
test_mat=np.zeros([5111, 3], dtype=np.float64)
test_mat[0][:]=point_mat[0][:]
k=0
j=0
idx_mat=[]
idx_mat.append(j)
for i in point_mat:
    i_reshape=i.reshape((1,3))
    for i_k in range(0,k+1):
        if all(test_mat[i_k][:]==i):
            break
        elif i_k==k:
            idx_mat.append(j)
            k+=1
            test_mat[k][:]=i
    j+=1

labels_of_point=np.zeros([5111, 1], dtype=np.int32)
k_labels=0
for i_idx_mat in idx_mat:
    labels_of_point[k_labels]=labels_mat[i_idx_mat]
    k_labels+=1
scio.savemat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\zuijinlin\Sample_06_1w_point.mat', {'X': test_mat}) 
scio.savemat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\zuijinlin\Sample_06_1w_labels.mat', {'X': labels_of_point})          

c=scio.loadmat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\zuijinlin\Sample_06_1w_point.mat')
d=scio.loadmat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\zuijinlin\Sample_06_1w_labels.mat')
e=scio.loadmat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\stl_to_mat_10w\Sample_06_10w.mat')
point_1w=c['X']
labels_1w=d['X']
mesh_10w_mat=e['X']
mesh_10w = vedo.load(r'F:\biyesheji\MeshSNet\dentalmesh(1)\surface_stl_10w\Sample_06.stl')
predicted_labels_10w = np.zeros([mesh_10w.NCells(), 1], dtype=np.int32)
point_mat_10w=np.zeros([mesh_10w.NCells(), 3], dtype=np.float64)
num_row_10w=mesh_10w.NCells()
num_line_10w=3
point_row_10w=0
point_line_10w=0
#point_mat=np.zeros([3*9999, 3], dtype=np.float64)
for i_row_10w in range(num_row_10w):
    for i_line_10w in range(num_line_10w):
        point_mat_10w[point_row_10w][point_line_10w]=mesh_10w_mat[i_row_10w][i_line_10w]
        point_line_10w+=1
        if point_line_10w==3:
            point_line_10w=0
            point_row_10w+=1
X_aix=np.zeros([5111, 1], dtype=np.float64)
Y_aix=np.zeros([5111, 1], dtype=np.float64)
Z_aix=np.zeros([5111, 1], dtype=np.float64)
k_line=0
for i_point_1w in point_1w:
    i_point_1w_reshape=i_point_1w.reshape((1,3))
    X_aix[k_line][0]=i_point_1w_reshape[0][0]
    Y_aix[k_line][0]=i_point_1w_reshape[0][1]
    Z_aix[k_line][0]=i_point_1w_reshape[0][2]
    k_line+=1
X_aix_10w=np.zeros([1, 1], dtype=np.float64)
Y_aix_10w=np.zeros([1, 1], dtype=np.float64)
Z_aix_10w=np.zeros([1, 1], dtype=np.float64)
juli_10w=np.zeros([5111, 1], dtype=np.float64)
juli_min_idx=[]
for i_point_mat_10w in point_mat_10w:
    i_point_mat_10w_reshape=i_point_mat_10w.reshape((1,3))
    X_aix_10w[0][0]=i_point_mat_10w_reshape[0][0]
    X_yunsuan=np.repeat(X_aix_10w,5111,axis=0)
    Y_aix_10w[0][0]=i_point_mat_10w_reshape[0][1]
    Y_yunsuan=np.repeat(Y_aix_10w,5111,axis=0)
    Z_aix_10w[0][0]=i_point_mat_10w_reshape[0][2]
    Z_yunsuan=np.repeat(Z_aix_10w,5111,axis=0)
    juli_10w=(X_aix_10w-X_aix)**2+(Y_aix_10w-Y_aix)**2+(Z_aix_10w-Z_aix)**2
    juli_min_idx.append(juli_10w.argmin())
print(mesh_10w.NCells())
for i_line_label in  range(mesh_10w.NCells())  :
    predicted_labels_10w[i_line_label]=labels_1w[juli_min_idx[i_line_label]]
mesh2 = mesh_10w.clone()
mesh2.addCellArray(predicted_labels_10w, 'Label')
output_path=r'F:\biyesheji\MeshSNet\dentalmesh(1)\mesh_predictied_vtp\\'
vedo.write(mesh2, os.path.join(output_path, 'Sample_06_zuijinlin_10w.vtp'))
#for i in point_mat:
#    np.unique