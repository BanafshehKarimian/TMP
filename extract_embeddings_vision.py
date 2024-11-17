import argparse
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import v2
import numpy as np
from utils.teachers_dict import teachers_dict, EmbedderFromTorchvision, embedder_size
from datasets import load_dataset
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"


datasets = {
            #"DTD": torchvision.datasets.DTD,
            #"ImageNet": torchvision.datasets.ImageNet,
            #"FGVCAircraft": torchvision.datasets.FGVCAircraft,
            "CIFAR10" : torchvision.datasets.CIFAR10,
            #"CIFAR100": torchvision.datasets.CIFAR100,
            #"FakeData": torchvision.datasets.FakeData,#a random data
            "FMNIST": torchvision.datasets.FashionMNIST,
            #"Flickr8": torchvision.datasets.Flickr8k,#needs to be downloaded manually
            #"Flickr30": torchvision.datasets.Flickr30k,#needs to be downloaded manually
            #"ImageNet": torchvision.datasets.ImageNet,#RuntimeError: The archive ILSVRC2012_devkit_t12.tar.gz is not present in the root directory or is corrupted. You need to download it externally and place it in ./Data.
            #"LSUN": torchvision.datasets.LSUN,#needs to be downloaded manually
            "MNIST": torchvision.datasets.MNIST,
            #"Places365": torchvision.datasets.Places365,too big
            "STL10": torchvision.datasets.STL10,#(image, target) where target is index of the target class.
            "CelebA": torchvision.datasets.CelebA,
            "SVHN": torchvision.datasets.SVHN,#	(image, target) where target is index of the target class.
            "QMNIST": torchvision.datasets.QMNIST,#?CvtModel
            #"EMNIST": torchvision.datasets.EMNIST,#?TypeError: EMNIST.__init__() missing 1 required positional argument: 'split'
            "KMNIST": torchvision.datasets.KMNIST,#?
            #"Omniglot": torchvision.datasets.Omniglot,#?
            #"CityScapes": torchvision.datasets.Cityscapes,
            #"COCO": torchvision.datasets.CocoCaptions,#Tuple (image, target). target is a list of captions for the image.
            #"SBU": torchvision.datasets.SBU, #image, caption
            #"Detection": torchvision.datasets.CocoDetection,
            #"PhotoTour": torchvision.datasets.PhotoTour,
            #"SBD": torchvision.datasets.SBDataset,
            #"USPS": torchvision.datasets.USPS,
            #"VOC": torchvision.datasets.VOCDetection, #(image, target) where target is a dictionary of the XML tree.
            }
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--teachers", nargs="+", type=str, default=list(teachers_dict.keys()))
    parser.add_argument("--datasets", nargs="+", type=str, default=list(datasets.keys()))

    args = parser.parse_args()    
    batch_size = 64
    num_workers = 10
    for dataset_name in args.datasets:
        for teacher_name in args.teachers:
            output = "Embeddings/"+teacher_name + "_" + dataset_name
            teacher = EmbedderFromTorchvision(teacher_name)
            transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
            data = datasets[dataset_name](root = "./Data",download = True, transform=transform)
            dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            teacher.to(device)
            dim = embedder_size[teacher_name]
            emb = torch.empty((0, dim), dtype=torch.float32).to("cpu")
            for batch in tqdm(dataloader):
                img, label = batch
                if img.shape[1] < 3:
                    img = torch.stack((img,img,img), dim = 1).squeeze(2)
                out = teacher(img.to(device))
                emb = torch.cat((emb, out.cpu().detach()), 0)
            print("saving" + output)
            print(emb.shape)
            np.save(output+'.npy', emb.cpu().detach().numpy()) 
