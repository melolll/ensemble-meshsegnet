# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:13:37 2021

@author: lkh
"""
import os
import numpy as np
import torch
import torch.nn as nn
from meshsegnet import *
import utils
import vedo
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix
import scipy.io as scio


if __name__ == '__main__':

    #torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #print(utils.get_avail_gpu())
    model_path = r'F:\biyesheji\MeshSNet\MeshSegNet-master (1)\MeshSegNet-master\models'
    model_name = 'MeshSegNet_Max_15_classes_72samples_lr1e-2_best.tar'
    prob_save_path=r'F:\biyesheji\MeshSNet\dentalmesh(1)\zuijinlin_predict_labels\\'
    mesh_path = 'F:\\biyesheji\MeshSNet\dentalmesh(1)\dentalmesh'  # need to define
    #sample_filenames = ['S1_Sample_01.mat'] # need to define
    num_sample=1
    output_path = r'F:\biyesheji\MeshSNet\MeshSegNet-master (1)\MeshSegNet-master\outputs'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    num_classes = 15
    num_channels = 15

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

    # load trained model
    checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu') # cpu是备选项
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device, dtype=torch.float)

    #cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True # 注：在backends文件夹中无cudnn.py文件
    
    
    
    mesh = vedo.load(r'F:\biyesheji\MeshSNet\dentalmesh(1)\dentalmesh1wstl\Sample_06.stl') #需定义
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
    scio.savemat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\zuijinlin_mat_1w\Sample_06_1w.mat', {'X': X}) #需定义
    #mesh = vedo.load(r'F:\biyesheji\MeshSNet\dentalmesh(1)\dentalmesh\Sample_01_1w.stl')
    #mesh_d = mesh.clone()
    #predicted_labels_d = np.zeros([mesh_d.NCells(), 1], dtype=np.int32)
    # Predicting
    model.eval() #测试样本不改变权值
    with torch.no_grad():
        for i_sample in range(1,num_sample+1):

            print('Predicting Sample filename: S{0}_Sample_06.mat'.format(i_sample))
            a=scio.loadmat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\zuijinlin_mat_1w\Sample_06_1w.mat') #需定义
            #b=scio.loadmat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\stl_to_mat_10w\Sample_01_1wnormallkh.mat')
            X=a['X']
            A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            D = distance_matrix(X[:, 9:12], X[:, 9:12])
            A_S[D<0.1] = 1.0
            A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            A_L[D<0.2] = 1.0
            A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            # numpy -> torch.tensor
            X = X.transpose(1, 0)
            X = X.reshape([1, X.shape[0], X.shape[1]])
            X = torch.from_numpy(X).to(device, dtype=torch.float)
            A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
            A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
            A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
            A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)

            tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
            patch_prob_output = tensor_prob_output.cpu().numpy()
            for i_label in range(num_classes):
                predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label
            print(predicted_labels_d.shape)
            output_file_name='Sample_06_predicted_labels.mat'#需定义
            scio.savemat(prob_save_path + output_file_name, {'labels': predicted_labels_d})
            print('Sample filename: S{0}_Sample_06.mat completed'.format(i_sample))