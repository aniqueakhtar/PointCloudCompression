#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:00:53 2021

@author: anique
"""
import numpy as np
# import pptk
import open3d as o3d
import glob
import os
from tqdm import tqdm

train = ['loot', 'longdress', 'queen']
test = ['redandblack', 'soldier']

no_of_frames = 2    # number of frames to process.
kd_depth = 2
jump = no_of_frames
start_test_files = 14

dir_train = 'train/dataset_v1'
dir_test = 'test/dataset_v1'    # Should be called Eval dataset


'''
#########
'''

# KD-tree Implementation
# Use variable pc to divide a list of pc in variable l_pc to depth
def kdtree_partition4(pc, pc_l, depth=kd_depth):
    parts = []
    parts_l = []

    class KD_node:
        def __init__(self, LL = None, RR = None):
            self.left = LL
            self.right = RR

    def createKDTree(root, d, data, li):
        if d >= depth:
            parts.append(data)
            parts_l.append(li)
            return

        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]
        l_li = []
        r_li = []
        for l in li:
            l_sorted = l[np.lexsort(l.T[dim_index, None])]
            l_li.append(l_sorted[: int((len(l) / 2))])
            r_li.append(l_sorted[int((len(l) / 2)):])

        root = KD_node(None)
        root.left = createKDTree(root.left, d+1, data_sorted[: int((len(data) / 2))], l_li)
        root.right = createKDTree(root.right, d+1, data_sorted[int((len(data) / 2)):], r_li)

        return root

    init_root = KD_node(None)
    root = createKDTree(init_root, 0, pc, pc_l)

    return parts, parts_l


'''
#########
'''


if not os.path.exists(dir_train):
        os.makedirs(dir_train)

print('\n\n !!! Creating Training Data !!!')

for seq_name in train:
    print('\n!!!!!=====================!!!!')
    print(seq_name)

    if seq_name == 'queen':
        files = sorted(glob.glob('/home/anique/Anique/Coding/Dataset_2/cat2_Dynamic_Objects/Technicolor/'+seq_name+'/**.ply'))
        files = files[3:]
    else:
        files = sorted(glob.glob('/home/anique/Anique/Coding/Dataset_2/cat2_Dynamic_Objects/8isequences/'+seq_name+'/Ply/**.ply'))


    start = int(files[0].rsplit('_',1)[1].split('.')[0])
    end = int(files[-1].rsplit('_',1)[1].split('.')[0])
    diff = (end-start+1)//no_of_frames * no_of_frames

    ## Going through no_of_frames at a time
    print('Number of files = ', len(files))
    print('Reading the files')
    for i in tqdm(range(0,diff-no_of_frames,jump)):
        f_c = files[i:i+no_of_frames]

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

        xyz_c = np.concatenate(xyz_l)

        _, parts_l = kdtree_partition4(xyz_c, xyz_l, depth=kd_depth)

        if i == 0:
            xyz_np = np.array(parts_l, dtype=object, ndmin=2)
        else:
            xyz_np = np.concatenate((xyz_np, np.array(parts_l, dtype=object, ndmin=2)), axis=0)


    print(xyz_np.shape)
    np.savez(os.path.join(dir_train, seq_name+'_'+'kd'+str(kd_depth)+'.npz'), xyz=xyz_np)


'''

## Plotting
import pptk
v_list = []
for i, p in enumerate(parts):
    print(i)
    v = pptk.viewer(p)
    v_list.append(v)
    for p2 in parts_l[i]:
        v = pptk.viewer(p2)
        v_list.append(v)

    input("Press Enter to continue...")

    for v in v_list:
        v.close()

'''
