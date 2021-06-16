# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:25:39 2018

@author: chlian
"""

import vtk
import numpy as np
import os
from vtk.util.numpy_support import vtk_to_numpy

import scipy.io as scio
from stl import mesh
import h5py

def data_preprocess(stl_path, subj_name, subj_lbl_name, save_path, save_idx):
    
    ''' Read .stl file from path '''
    subj_lbl_name=[]
    StlReader = vtk.vtkSTLReader()
    StlReader.SetFileName(stl_path+subj_name)

    StlReader.Update()
    pcloud = StlReader.GetOutput().GetPoints().GetData()
    pcloud = vtk_to_numpy(pcloud)

    LblReader = [vtk.vtkSTLReader() for _ in range(len(subj_lbl_name))]
    for i in range(len(subj_lbl_name)):
        LblReader[i].SetFileName(stl_path+subj_lbl_name[i])

    ''' Mesh decimation '''
    #Deci = vtk.vtkDecimatePro()
    #Deci.SetInputConnection(StlReader.GetOutputPort())
    #Deci.SetTargetReduction(0.9)
    #Deci.PreserveTopologyOn() # preserve original topology
    #
    #Deci.Update()
    #deci_pcloud = Deci.GetOutput().GetPoints().GetData()
    #deci_pcloud = vtk_to_numpy(deci_pcloud)

    ''' Smooth '''
    #Smooth = vtk.vtkSmoothPolyDataFilter()
    #Smooth.SetInputConnection(Deci.GetOutputPort())
    #Smooth.SetNumberOfIterations(50)
    #Normals = vtk.vtkPolyDataNormals()
    #Normals.SetInputConnection(Smooth.GetOutputPort())
    #Normals.FlipNormalsOn()
    #
    #Smooth.Update()
    #smth_pcloud = Smooth.GetOutput().GetPoints().GetData()
    #smth_pcloud = vtk_to_numpy(smth_pcloud)
    Trans = vtk.vtkTransform()
    ''' Random rotation, translate, & scale '''
    ''' 
    Trans = vtk.vtkTransform()

    ry_flag = np.random.randint(0,2)
    rx_flag = np.random.randint(0,2)
    rz_flag = np.random.randint(0,2)
    if ry_flag == 1:
        # rotate along Yth axis
        Trans.RotateY(np.random.uniform(-180, 180))
    if rx_flag == 1:
        # rotate along Xth axis
        Trans.RotateX(np.random.uniform(-45, 45))
    if rz_flag == 1:
        # rotate along Zth axis
        Trans.RotateZ(np.random.uniform(-45, 45))

    trans_flag = np.random.randint(0,2)
    if trans_flag == 1:
        Trans.Translate([np.random.uniform(-20, 20),
                         0, 
                         np.random.uniform(-20, 20)])
    
    scale_flag = np.random.randint(0,2)
    if scale_flag == 1:
        Trans.Scale([np.random.uniform(0.7, 1.3),
                     np.random.uniform(0.7, 1.3),
                     np.random.uniform(0.7, 1.3)])
'''
    TransFilter = vtk.vtkTransformPolyDataFilter()
    TransFilter.SetTransform(Trans)#盲猜随机旋转、平移和缩放
    TransFilter.SetInputConnection(StlReader.GetOutputPort())

    TransFilter.Update()
    tpcloud = TransFilter.GetOutput().GetPoints().GetData()
    tpcloud = vtk_to_numpy(tpcloud)

    ''' Write stl to disk '''
    StlWriter = vtk.vtkSTLWriter()
    StlWriter.SetFileName(save_path+'A{0}_'.format(save_idx)+subj_name)
    StlWriter.SetInputConnection(TransFilter.GetOutputPort())
    StlWriter.Write()
    
    for i in range(len(subj_lbl_name)):
        TransFilter.SetInputConnection(LblReader[i].GetOutputPort())
        StlWriter = vtk.vtkSTLWriter()
        StlWriter.SetFileName(save_path+'A{0}_'.format(save_idx)+subj_lbl_name[i])
        StlWriter.SetInputConnection(TransFilter.GetOutputPort())
        StlWriter.Write()
    
    return tpcloud


def stl_to_h5(stl_path, h5_path, pset, subj_name, 
              subj_lbl_name, save_idx, h5_name, num_faces=5000):
    ''' 
    Sample 5000 faces from each stl file, and save them as h5
    
    Face features (dim=15) include:
    1. z-score normalized face coordinates, i.e., (cor-mean)/std
    2. relative location of each face's barycenter with respect to the object,
       i.e., (barycenter-min)/(max-min)
    3. normals for faces, also z-score normalized
    '''
    subj_lbl_name=[]
    mset = mesh.Mesh.from_file(stl_path+'A{0}_'.format(save_idx)+subj_name) #
    #mset_lbl = [mesh.Mesh.from_file(stl_path+'A{0}_'.format(save_idx)+subj_lbl_name[i])
               # for i in range(len(subj_lbl_name))] #
    
    faces = mset.points
    normals = mset.normals
    barycenters = (faces[:,0:3]+faces[:,3:6]+faces[:,6:9]) / 3
    
    #lbl_faces = [mset_lbl[i].points for i in range(len(subj_lbl_name))]
    """
    faces = np.round(faces)
    lbl_faces = [np.round(lbl_faces[i]) for i in range(len(subj_lbl_name))]
    """
    
    #Y = np.zeros((faces.shape[0],len(subj_lbl_name)+1), dtype='float32')
    for i_cls in range(len(subj_lbl_name)):
        lbl_idxs = [np.where(np.all(faces==lbl_faces[i_cls][i,:],axis=1))[0][0] 
                    for i in range(lbl_faces[i_cls].shape[0])]
        Y[lbl_idxs,i_cls] = 1
#    Y[:,-1] = 1 - np.sum(Y[:,:-1],axis=1)

                        
    maxs = pset.max(axis=0)
    mins = pset.min(axis=0)
    means = pset.mean(axis=0)
    stds = pset.std(axis=0)
    nmeans = normals.mean(axis=0)
    nstds = normals.std(axis=0)
    
    for i in range(3):
        barycenters[:,i] = (barycenters[:,i]-mins[i])/(maxs[i]-mins[i])
        faces[:,i] = (faces[:,i]-means[i])/stds[i]
        faces[:,i+3] = (faces[:,i+3]-means[i])/stds[i]
        faces[:,i+6] = (faces[:,i+6]-means[i])/stds[i]
        normals[:,i] = (normals[:,i]-nmeans[i])/nstds[i]
        
    X = np.column_stack((faces, barycenters,))
    
    '''
    # sample 5000 faces: positive faces + (5000-num_positive) uniform faces
    num_positives = sum(Y==1)[0]
    num_negatives = num_faces - num_positives
    positive_idxs = np.where(Y==1)[0].tolist()
    negative_idxs = np.where(Y==0)[0]
    negative_idxs = np.random.choice(negative_idxs,size=num_negatives,replace=False)
    sample_idxs = positive_idxs + negative_idxs.tolist()
    sample_idxs.sort()
    
    X = X[sample_idxs,:]
    Y = Y[sample_idxs,:]
    '''
    '''
    h5_file = h5py.File(h5_path+'A{0}_'.format(save_idx)+h5_name, 'w')
    h5_file.create_dataset('faces', data=X)
    h5_file.create_dataset('labels', data=Y)
    h5_file.close()
    '''
    #scio.savemat(dataNew, {h5_path+'A{0}_'.format(save_idx)+h5_name: data['X','Y']})
    print(h5_path+'A{0}_'.format(save_idx)+h5_name)

    scio.savemat(h5_path+'A{0}_'.format(save_idx)+h5_name, {'X':X})
   #scio.savemat(h5_path+'center{0}_'.format(save_idx)+h5_name, {'barycenters':barycenters})
    print(h5_path+'A{0}_'.format(save_idx)+h5_name)
    #return X,Y
    return faces,barycenters
        
    
if __name__ == '__main__':

    #stl_path = 'F:\\biyesheji\MeshSNet\dentalmesh(1)\surface_stl_10w\\'
    stl_path=r'F:\biyesheji\MeshSNet\dentalmesh(1)\dentalmesh\\'
    stl_save_path = r'F:\biyesheji\MeshSNet\dentalmesh(1)\aug_stl_10w\\'
    if not os.path.exists(stl_save_path):
        os.mkdir(stl_save_path)
    #h5_save_path = '/home/cflian/Data/maxillary_surfaces_original_h5/'
    h5_save_path = r'F:\\biyesheji\MeshSNet\dentalmesh(1)\stl_to_mat_10w\\'
    if not os.path.exists(h5_save_path):
        os.mkdir(h5_save_path)
    
    import glob
    data_list = glob.glob(h5_save_path+'*.h5')

    #subj_idxs = [3]
    for i_subj in range(1,2):
        subj_name = 'Sample_0{0}.stl'.format(i_subj)
        subj_lbl_name = ['Sample_0{0}_.stl'.format(i_subj,i_tooth)
                            for i_tooth in ['2','3','4','5','6',
                                            '7','8','9','10','11',
                                            '12','13','14','15']]#
        h5_name = 'Sample_0{0}.mat'.format(i_subj)

        '''
        StlReader = vtk.vtkSTLReader()
        StlReader.SetFileName(stl_path + subj_name)
        StlReader.Update()
        pcloud = StlReader.GetOutput().GetPoints().GetData()
        pcloud = vtk_to_numpy(pcloud)

        X, Y = stl_to_h5(stl_path, h5_save_path, pcloud, subj_name,
                         subj_lbl_name, 0, h5_name)
        print (X.shape,Y.shape)
        '''

        for i_aug in range(1,2):
            
            pset = data_preprocess(stl_path, subj_name, subj_lbl_name,
                                   stl_save_path, i_aug)
                                 
            StlReader = vtk.vtkSTLReader()
            StlReader.SetFileName(stl_save_path+'A{0}_'.format(i_aug)+subj_name)

            StlReader.Update()
            pset = StlReader.GetOutput().GetPoints().GetData()
            pset = vtk_to_numpy(pset)
            
            X = stl_to_h5(stl_save_path, h5_save_path, pset, subj_name,
                            subj_lbl_name, i_aug, h5_name)    

        print(i_subj)
