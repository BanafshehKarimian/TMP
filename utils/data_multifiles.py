import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from Data.dataset_dict import datasets_dict

class TeacherEmbeddingDataset(Dataset):
    def __init__(self, teacher_paths, teacher_name, datasets, length, indices):
        self.teacher_paths = teacher_paths
        self.teacher_name = teacher_name
        self.length = length
        index = 0
        self.embeding_index_to_file = {}
        for ds in datasets:
            data = np.load(self.teacher_paths + "/"+teacher_name+"_"+ds+".npy", mmap_mode="r")
            self.embeding_index_to_file[index] = self.teacher_paths + "/"+teacher_name+"_"+ds+".npy"
            index = index + np.shape(data)[0]
        self.indices = indices

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        index = np.argmax(self.indices > idx) - 1
        ds_index = self.indices[index]
        local_index = idx - ds_index
        data = np.load(self.embeding_index_to_file[ds_index], mmap_mode="r")
        #print(self.embeding_index_to_file[ds_index])
        return data[local_index]
        

        
class MultiTeacherAlignedEmbeddingDataset:

    def __init__(  self,  teachers_path = "./Embeddings", list_teachers = None, transform = None):
        self.teacher_paths = teachers_path
        index = 0
        teacher_name = list_teachers[0]
        self.embeding_index_to_ds = {}
        self.datasets = []
        for ds in datasets_dict.keys():
            self.datasets.append(ds)
            self.embeding_index_to_ds[index] = datasets_dict[ds](root = "./Data",download = True, transform=transform)
            data = np.load(self.teacher_paths + "/"+teacher_name+"_"+ds+".npy", mmap_mode="r")
            index = index + np.shape(data)[0]
        self.length = index
        self.teachers_ds = []
        self.indices = np.array(list(self.embeding_index_to_ds.keys()))
        for teacher_name in list_teachers:
            self.teachers_ds.append(TeacherEmbeddingDataset(self.teacher_paths, teacher_name, self.datasets, self.length, self.indices))
        
    
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        index = np.argmax(self.indices > idx) - 1
        ds_index = self.indices[index]
        local_index = idx - ds_index
        img, lable = self.embeding_index_to_ds[ds_index][local_index]
        if not torch.is_tensor(img):
            img = torch.stack(img['pixel_values'], dim=0)
            if img.shape[0]==1:
                img = img.squeeze(0)
            print(img.shape)
        #print(self.datasets[index], idx, ds_index, local_index)
        emb = []#torch.tensor([])
        for teacher in self.teachers_ds:
            emb.append(teacher[idx])# = torch.cat((emb, teacher[idx]))
        return np.array(img), emb#, lable, .flatten()

        

