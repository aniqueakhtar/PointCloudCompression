import numpy as np
import torch
import MinkowskiEngine as ME
import os
import time
from torch.utils.data.sampler import Sampler


class Dataset_Test(torch.utils.data.Dataset):

    def __init__(self, path, files):
        # self.coords_G1 = []
        self.coords_G2 = []
        # self.coords_G3 = []
        self.coords_R1 = []
        # self.coords_R3 = []
        
        for i,f in enumerate(files):
            data_path = os.path.join(path, f)
            data = np.load(data_path, mmap_mode='r', allow_pickle=True)
            xyz = data['xyz']
            # self.coords_G1.append(xyz[0])
            self.coords_G2.append(xyz[1])
            # self.coords_G3.append(xyz[2])
            self.coords_R1.append(xyz[3])
            # self.coords_R3.append(xyz[4])
        
    def __len__(self):
        return len(self.coords_G2)

    def __getitem__(self, idx):
        return (self.coords_G2[idx], self.coords_R1[idx])
    
def collate_pointcloud_fn_test(list_data):
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    
    coords_G2, coords_R1 = list(zip(*list_data))
    
    coords_G2_batch = ME.utils.batched_coordinates(coords_G2)
    coords_R1_batch = ME.utils.batched_coordinates(coords_R1)
    
    return coords_G2_batch, coords_R1_batch


class Dataset_Train(torch.utils.data.Dataset):

    def __init__(self, path, files):
        self.coords_1 = []
        self.coords_2 = []
        # self.coords_3 = []
        
        for i,f in enumerate(files):
            data_path = os.path.join(path, f)
            data = np.load(data_path, mmap_mode='r', allow_pickle=True)
            xyz = data['xyz']
            
            for c in xyz:
                self.coords_1.append(c[0])
                self.coords_2.append(c[1])
                # self.coords_3.append(c[1])
            
    def __len__(self):
        return len(self.coords_1)

    def __getitem__(self, idx):
        return (self.coords_1[idx], self.coords_2[idx])
        
    
def collate_pointcloud_fn_train(list_data):
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    
    coords_1, coords_2 = list(zip(*list_data))
    
    coords_1_batch = ME.utils.batched_coordinates(coords_1)
    coords_2_batch = ME.utils.batched_coordinates(coords_2)
    
    return coords_1_batch, coords_2_batch

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)



def make_data_loader(path, files, batch_size, train, shuffle, num_workers, repeat):
    
    if train:
        args = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'collate_fn': collate_pointcloud_fn_train, 
            'pin_memory': True,
            'drop_last': False
        }
    
        start_time = time.time()
        print("Going to load the whole dataset in the memory")
        dataset = Dataset_Train(path, files)
        print("Time taken to load the dataset: ", round(time.time() - start_time, 4))
        
        if repeat:
            args['sampler'] = InfSampler(dataset, shuffle)
        else:
            args['shuffle'] = shuffle
        
        loader = torch.utils.data.DataLoader(dataset, **args)
        
    else:
        args = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'collate_fn': collate_pointcloud_fn_test, 
            'pin_memory': True,
            'drop_last': False
        }
        
        start_time = time.time()
        print("Going to load the whole dataset in the memory")
        dataset = Dataset_Test(path, files)
        print("Time taken to load the dataset: ", round(time.time() - start_time, 4))
        
        if repeat:
            args['sampler'] = InfSampler(dataset, shuffle)
        else:
            args['shuffle'] = shuffle
        
        loader = torch.utils.data.DataLoader(dataset, **args)
    
    return loader
