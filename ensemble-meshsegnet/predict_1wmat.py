# -*- coding: utf-8 -*-
"""
Created on Sun May  9 17:35:48 2021

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
    model_path = './models'
    model_name = 'MeshSegNet_Max_15_classes_72samples_lr1e-2_best.tar'
    prob_save_path=r'F:\biyesheji\MeshSNet\dentalmesh(1)\prob_save_path\\'
    mesh_path = 'F:\\biyesheji\MeshSNet\dentalmesh(1)\dentalmesh'  # need to define
    #sample_filenames = ['S1_Sample_01.mat'] # need to define
    num_sample=30
    output_path = './outputs'
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

    # Predicting
    model.eval() #测试样本不改变权值
    with torch.no_grad():
        for i_sample in range(1,num_sample+1):

            print('Predicting Sample filename: S{0}_Sample_01.mat'.format(i_sample))
            a=scio.loadmat(r'F:\biyesheji\MeshSNet\dentalmesh(1)\mat10w_to_mat1w\S{0}_Sample_01.mat'.format(i_sample))
            X=a['X']
            # computing A_S and A_L
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
            output_file_name='S{0}_Sample_01_prob.mat'.format(i_sample)
            scio.savemat(prob_save_path + output_file_name, {'prob': patch_prob_output})
            print('Sample filename: S{0}_Sample_01.mat completed'.format(i_sample))