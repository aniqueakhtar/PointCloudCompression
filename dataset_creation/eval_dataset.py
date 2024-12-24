#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:00:53 2021

@author: anique

Filename should have been eval_dataset
We are creating evaluation dataset during training.
Unfortunately, it is called test_dataset throughout the code.
"""
import numpy as np
import open3d as o3d
import glob
import os


test_1 = ['soldier', 'redandblack']
test_2 = ['basketball', 'dancer', 'exercise', 'model']

no_of_frames = 3    # number of frames to process.
start_test_files = 14

dir_test = 'test/dataset_v5'  # Should be called Eval dataset

# Test Files Should Look Like This:
# [F1_GT, F2_GT, F3_GT, F1_Reconstructed, F3_Reconstructed]
# But it looks like in this particular file at the moment:
# [F1_GT, F2_GT, F3_GT, F1_GT, F3_GT]


if not os.path.exists(dir_test):
        os.makedirs(dir_test)
## Creating Test files.
print('\n\n !!! Creating Test Data !!!')
for i,seq_name in enumerate(test_1):
    print('\n!!!!!=====================!!!!')
    print(seq_name)

    if seq_name == 'queen':
        files = sorted(glob.glob('/home/anique/Anique/Coding/Dataset_2/cat2_Dynamic_Objects/Technicolor/'+seq_name+'/**.ply'))
        files = files[3:]
    else:
        files = sorted(glob.glob('/home/anique/Anique/Coding/Dataset_2/cat2_Dynamic_Objects/8isequences/'+seq_name+'/Ply/**.ply'))

    f_c = files[start_test_files:start_test_files+no_of_frames]
    f_c = f_c + f_c[:1] + f_c[-1:]
    print(f_c)

    xyz_l = []
    check = False
    for f in f_c:
        pcd = o3d.io.read_point_cloud(f)
        xyz = np.asarray(pcd.points)
        if xyz.shape[0] == 0:
            print("Couldn't read file !!")
            print(f)
            check = True
            break
        xyz_l.append(xyz)
    if check:
        continue

    xyz_np = np.array(xyz_l, dtype=object)

    # print(xyz_np.shape)
    # for j in xyz_np:
    #     print(j.shape)
    np.savez(os.path.join(dir_test, seq_name+'.npz'), xyz=xyz_np)



dir_name = {
    'basketball' : 'basketball_player_vox11',
    'dancer': 'dancer_vox11',
    'exercise': 'exercise_vox11',
    'model' : 'model_vox11'
    }


'''
['/home/anique/Anique/Coding/Dataset_2/cat2_Dynamic_Objects/Owlii_2/model_vox11/model_vox11_00000015.ply',
 '/home/anique/Anique/Coding/Dataset_2/cat2_Dynamic_Objects/Owlii_2/model_vox11/model_vox11_00000016.ply',
 '/home/anique/Anique/Coding/Dataset_2/cat2_Dynamic_Objects/Owlii_2/model_vox11/model_vox11_00000017.ply',]
'''

for i,seq_name in enumerate(test_2):
    print('\n!!!!!=====================!!!!')
    print(seq_name)

    files = sorted(glob.glob('/home/anique/Anique/Coding/Dataset_2/cat2_Dynamic_Objects/Owlii_2/'+dir_name[seq_name]+'/**.ply'))

    f_c = files[start_test_files:start_test_files+no_of_frames]
    f_c = f_c + f_c[:1] + f_c[-1:]
    print(f_c)


    xyz_l = []
    check = False
    for f in f_c:
        pcd = o3d.io.read_point_cloud(f)
        xyz = np.asarray(pcd.points)
        xyz = np.unique(np.round(xyz/2), axis=0)
        if xyz.shape[0] == 0:
            print("Couldn't read file !!")
            print(f)
            check = True
            break
        xyz_l.append(xyz)
    if check:
        continue

    xyz_np = np.array(xyz_l, dtype=object)

    # print(xyz_np.shape)
    # for j in xyz_np:
    #     print(j.shape)
    np.savez(os.path.join(dir_test, seq_name+'.npz'), xyz=xyz_np)
