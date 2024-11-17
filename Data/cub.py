import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torch.utils.data import Subset
def get_cub_train(root, train = True, download = False, transform = None):
    shuffle = False
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             transforms.RandomErasing(p=0.25, value='random')
        ])
        all_data = datasets.ImageFolder(root+'/CUB_200_2011/images', transform=transform)
        train_indices = []
        test_indices = []
        
        with open('./Data/train_test_split.txt', 'r') as f:
            for line in f:
                index, is_train = map(int, line.strip().split())
                if is_train == 0:
                    test_indices.append(index)
                elif is_train == 1:
                    train_indices.append(index)
        
        # Create subsets for train and test
        train_data = Subset(all_data, train_indices)
        test_data = Subset(all_data, test_indices)
        
        return train_data
    
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        all_data = datasets.ImageFolder(root+'/CUB_200_2011/images', transform=transform)
        train_indices = []
        test_indices = []
        
        with open('./Data/train_test_split.txt', 'r') as f:
            for line in f:
                index, is_train = map(int, line.strip().split())
                if is_train == 0:
                    test_indices.append(index)
                elif is_train == 1:
                    train_indices.append(index)
        
        # Create subsets for train and test
        train_data = Subset(all_data, train_indices)
        test_data = Subset(all_data, test_indices)
        return test_data
def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes
def get_data_loaders(data_dir, batch_size, train = False, shuffle = False):
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             transforms.RandomErasing(p=0.25, value='random')
        ])
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data)*0.75)
        valid_data_len = int((len(all_data) - train_data_len)/2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        return train_loader, train_data_len
    
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data)*0.70)
        valid_data_len = int((len(all_data) - train_data_len)/2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        return (val_loader, test_loader, valid_data_len, test_data_len)