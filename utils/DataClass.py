from Data.dataset_dict import datasets_dict
import pytorch_lightning as pl
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
import random
import numpy as np
import open_clip
import torchvision
from torchvision import datasets
from medmnist import *
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g )

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g )

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g )

class CIFAR100DataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["CIFAR100"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["CIFAR100"](root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        


class CIFAR10DataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["CIFAR10"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["CIFAR10"](root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class CIFAR10DataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform =  v2.Compose([
        	v2.RandomHorizontalFlip(p=0.5),
                #v2.RandomVerticalFlip(p=0.5),
                #v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.05),
                v2.Resize((225,225)),
                #v2.RandomCrop((200,200)),
                v2.Resize((225,225)),
                v2.RandomResizedCrop((225,225), scale=(0.9, 1.0)),
                v2.Resize((225,225)),
                v2.ToTensor(),])
        self.transform_test = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["CIFAR10"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["CIFAR10"](root = "./Data", train = False, download = download, transform=self.transform_test)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        self.valid_data.transform = self.transform_test
              

class FashionMNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["FMNIST"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["FMNIST"](root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class MNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform =  v2.Compose([
        	#v2.RandomHorizontalFlip(p=0.5),
                #v2.RandomVerticalFlip(p=0.5),
                #v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.05),
                v2.RandomResizedCrop((225,225), scale=(0.3, 1.0)),
                v2.ToTensor(),])
        self.transform_test = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["MNIST"](root = "./Data", train = True, download = download, transform=self.transform_test)
        self.test_data = datasets_dict["MNIST"](root = "./Data", train = False, download = download, transform=self.transform_test)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        self.valid_data.transform = self.transform_test
        

class STL10DataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["STL10"](root = "./Data", split = 'train', download = download, transform=self.transform)
        self.test_data = datasets_dict["STL10"](root = "./Data", split = 'test', download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        
class CustomCelebA(torchvision.datasets.CelebA):
    def __init__(self, root, split="train", target_type="attr", transform=None, target_transform=None, download=False, specific_index=None):
        # Initialize the parent CelebA class with the standard arguments
        super().__init__(root=root, split=split, target_type=target_type, transform=transform, target_transform=target_transform, download=download)
        
        self.index = specific_index
        
    def __getitem__(self, index):
        # Retrieve image and label
        image, labels = super().__getitem__(index)
        # Return image, index, and label corresponding to that index
        return image, labels
        
class CelebADataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, index = 0):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([
        	v2.RandomHorizontalFlip(p=0.5),
                #v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.05),
                v2.RandomResizedCrop((225,225), scale=(0.3, 1.0)),
                v2.ToTensor(),])
        self.transform_test = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = CustomCelebA(root = "./Data", split = 'train', download = download, transform=self.transform, specific_index = index)
        self.test_data = CustomCelebA(root = "./Data", split = 'test', download = download, transform=self.transform_test, specific_index = index)
        self.valid_data = CustomCelebA(root = "./Data", split = 'valid', download = download, transform=self.transform_test, specific_index = index)
        

class SVHNDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["SVHN"](root = "./Data", split = 'train', download = download, transform=self.transform)
        self.test_data = datasets_dict["SVHN"](root = "./Data", split = 'test', download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class QMNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["QMNIST"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["QMNIST"](root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class KMNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["KMNIST"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["KMNIST"](root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        
med_datasets = {"PathMNIST": PathMNIST, "DermaMNIST": DermaMNIST, "OCTMNIST": OCTMNIST, "PneumoniaMNIST": PneumoniaMNIST, "RetinaMNIST": RetinaMNIST, "ChestMNIST": ChestMNIST, 
                    "BreastMNIST": BreastMNIST, "BloodMNIST": BloodMNIST, "TissueMNIST": TissueMNIST ,"OrganAMNIST": OrganAMNIST, "OrganCMNIST": OrganCMNIST ,"OrganSMNIST": OrganSMNIST,
                    "OrganMNIST3D": OrganMNIST3D, "NoduleMNIST3D": NoduleMNIST3D, "FractureMNIST3D": FractureMNIST3D, "AdrenalMNIST3D": AdrenalMNIST3D, "VesselMNIST3D": VesselMNIST3D, "SynapseMNIST3D": SynapseMNIST3D}
        
class MedMNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, data_name, download = True):
        super().__init__(batch_size, num_workers)
        info = INFO[data_name.lower()]
        self.num_classes = len(info['label'])
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = med_datasets[data_name](root = "./Data", split="train", download = download, transform=self.transform)
        self.test_data = med_datasets[data_name](root = "./Data", split="test", download = download, transform=self.transform)
        self.valid_data = med_datasets[data_name](root = "./Data", split="val", download = download, transform=self.transform)



class PCAMDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = False):
        super().__init__(batch_size, num_workers)
        _,_, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        self.transform = v2.Compose([
                self.preprocess_val.transforms[0],
                self.preprocess_val.transforms[1],
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.05),
                self.preprocess_val.transforms[2],
                self.preprocess_val.transforms[3],
                self.preprocess_val.transforms[4],])
        self.training_data = datasets.PCAM(root = "./Data", split = 'train', download = download, transform=self.transform)
        self.valid_data = datasets.PCAM(root = "./Data", split = "val", download = download, transform=self.preprocess_val)
        self.test_data = datasets.PCAM(root = "./Data", split = "test", download = download, transform=self.preprocess_val)

data_module_dict = {#"CIFAR10", "FMNIST", "MNIST", "STL10", "SVHN", "QMNIST", "KMNIST"
            "CIFAR10" : CIFAR10DataModule,
            "CIFAR100": CIFAR100DataModule,
            "FMNIST": FashionMNISTDataModule,
            "MNIST": MNISTDataModule,
            "STL10": STL10DataModule,
            "CalebA": CelebADataModule,
            "SVHN": SVHNDataModule,
            "QMNIST": QMNISTDataModule,
            "KMNIST": KMNISTDataModule,
            }
data_class_dict = {
            "CIFAR10" : 10,
            "CIFAR100": 100,
            "FMNIST": 10,
            "MNIST": 10,
            "STL10": 10,
            "CalebA": 40,
            "SVHN": 10,
            "QMNIST": 10,
            "KMNIST": 10,
            "PCAM": 2,
            }
