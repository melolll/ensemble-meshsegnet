# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:00:32 2021

@author: lkh
"""
import scipy.io as scio
import random
import vedo
import numpy as np
#data_load_path=
data_save_path=r'F:\biyesheji\MeshSNet\dentalmesh(1)\mat10w_to_mat1w\\'
data_index_save_path=r'F:\biyesheji\MeshSNet\dentalmesh(1)\data_index_save_path\\'
num_sample=20 #需定义

#dataNew = r'F:\biyesheji\MeshSNet\dentalmesh(1)\mat10w_to_mat1w'

a=scio.loadmat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\stl_to_mat_10w\Sample_06_10w.mat') #需定义
b=a['X']
for i_num_sample in range(1,num_sample+1):
    print(i_num_sample)
    data=[]
    index=[]
    k=0
    j=0
    for i in b:
        r=random.randint(0,9)
        if r==0:
            print(i)
            k+=1
            if(k==10000):
                break
            data.append(i)
            index.append(j)
        j+=1
        
           
    output_file_name='S{0}_Sample_06.mat'.format(i_num_sample) #需定义
    scio.savemat(data_save_path+output_file_name, {'X':data})
    output_file_name_idx='S{0}_Sample_06_index.mat'.format(i_num_sample) #需定义
    scio.savemat(data_index_save_path+output_file_name_idx, {'idx':index})
  
mesh = vedo.load(r'F:\biyesheji\MeshSNet\dentalmesh(1)\surface_stl_10w\Sample_06.stl')# 需定义
cell = np.zeros([mesh.NCells(), 15], dtype='float32')
cell_index_num=np.zeros([mesh.NCells(), 1], dtype='int32')
data_save_path=r'F:\biyesheji\MeshSNet\dentalmesh(1)\mat10w_to_mat1w\\'
data_index_save_path=r'F:\biyesheji\MeshSNet\dentalmesh(1)\data_index_save_path\\'
num_sample_new=num_sample+1
for i_num_sample in range(1,num_sample+1):
    input_file_name='S{0}_Sample_06.mat'.format(i_num_sample) #需定义
    input_file_name_idx='S{0}_Sample_06_index.mat'.format(i_num_sample)#需定义
    prob_mat=scio.loadmat(data_save_path+input_file_name)
    idx_mat=scio.loadmat(data_index_save_path+input_file_name_idx)
    idx_list=idx_mat['idx']
    index=0
    prob_array=prob_mat['X']
    test1=prob_array[:][0]
    #test2=test1[1].reshape((1,15))
    for i in prob_array:
        i_reshape=i.reshape((1,15))
        #print(i_reshape)
        row_num=idx_list[0][index]
        print(row_num)
        cell[[row_num],:]+=i_reshape
        cell_index_num[[row_num],:]+=1
        index+=1
        print(index)
    print(i_num_sample)
#test=idx_list[0][2]
test_num_1=0
overplus=[]
for i_cell_index_num in cell_index_num:
    if i_cell_index_num == 0:
        overplus.append(test_num_1)
    test_num_1+=1
output_file_name_idx='S{0}_Sample_06_index.mat'.format(num_sample_new) #需定义
scio.savemat(data_index_save_path+output_file_name_idx, {'idx':overplus})
print(len(overplus))
print(test_num_1)


data_overplus=[]
num_overplus=0
line_num=0
for i_b in b:
    if line_num==overplus[num_overplus]:
        data_overplus.append(i_b)
        num_overplus+=1
        if num_overplus==len(overplus):
            break
    line_num+=1


output_file_name_overplus='S{0}_Sample_06.mat'.format(num_sample_new) #需定义
scio.savemat(data_save_path+output_file_name_overplus, {'X':data_overplus})
