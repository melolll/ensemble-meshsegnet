# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:49:32 2021

@author: lkh
"""
import numpy as np
import scipy.io as scio
import vedo
import os


num_classes=15
mesh = vedo.load(r'F:\biyesheji\MeshSNet\dentalmesh(1)\surface_stl_10w\Sample_06.stl') #需定义
mesh_d=mesh.clone()
cell=np.zeros([mesh_d.NCells(), 15], dtype='float32')
cell_index_num=np.zeros([mesh_d.NCells(), 1], dtype='int32')
prob_mat_path=r'F:\biyesheji\MeshSNet\dentalmesh(1)\prob_save_path\\'
idx_mat_path=r'F:\biyesheji\MeshSNet\dentalmesh(1)\data_index_save_path\\'
#prob_mat=scio.loadmat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\prob_save_path\S1_Sample_01_prob.mat')
#idx_mat=scio.loadmat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\data_index_save_path\S1_Sample_01_index.mat')
num_mat=21
for i_num_mat in range(1,num_mat+1):
    input_file_name_prob='S{0}_Sample_06_prob.mat'.format(i_num_mat) #需定义
    input_file_name_idx='S{0}_Sample_06_index.mat'.format(i_num_mat) #需定义
    prob_mat=scio.loadmat(prob_mat_path+input_file_name_prob)
    idx_mat=scio.loadmat(idx_mat_path+input_file_name_idx)
    idx_list=idx_mat['idx']
    index=0
    prob_array=prob_mat['prob']
    test1=prob_array[:][:][0]
    #test2=test1[1].reshape((1,15))
    for i in test1:
        i_reshape=i.reshape((1,15))
        #print(i_reshape)
        row_num=idx_list[0][index]
        print(row_num)
        cell[[row_num],:]+=i_reshape
        cell_index_num[[row_num],:]+=1
        index+=1
        print(index)
    print(i_num_mat)
#test=idx_list[0][2]
test_num_1=0
for i_cell_index_num in cell_index_num:
    if i_cell_index_num == 0:
        test_num_1+=1
print(test_num_1)
cell_index_num_1=np.where(cell_index_num>0,cell_index_num,1)
cell_prob=cell/cell_index_num_1
predicted_labels = np.zeros([mesh_d.NCells(), 1], dtype=np.int32)
k=0
for i_prob in cell_prob:
    i_prob_reshape=i_prob.reshape((1,15))
    maxtest=np.argmax(i_prob_reshape)
    #print(maxtest)
    predicted_labels[[k],:]=maxtest
    k+=1
    #print(i_prob)
mesh2 = mesh_d.clone()
mesh2.addCellArray(predicted_labels, 'Label')
output_path=r'F:\biyesheji\MeshSNet\dentalmesh(1)\mesh_predictied_vtp\\'
vedo.write(mesh2, os.path.join(output_path, 'Sample_06_predicted_ensemble.vtp'))
print(predicted_labels.shape)
print(mesh.NCells())