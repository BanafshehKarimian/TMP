import argparse
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import v2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from Data.cub import get_cub_train
from utils.teachers_dict import teachers_dict_vit, teachers_dict, EmbedderFromTorchvision, embedder_size
from transformers import DPTModel, PvtV2ForImageClassification, ViTHybridModel, CvtModel, LevitConfig, LevitModel, AutoImageProcessor, AutoModel, AutoFeatureExtractor, SwinForImageClassification, MobileViTFeatureExtractor, MobileViTForImageClassification, ViTImageProcessor, ViTForImageClassification, AutoFeatureExtractor, DeiTForImageClassificationWithTeacher, BeitImageProcessor, BeitForImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
root_imgnet = "/export/datasets/public/image_classification/imagenet/images"
datasets = {
            "cub": get_cub_train,
            #"Food101": torchvision.datasets.Food101,
            "DTD": torchvision.datasets.DTD,
            #"ImageNet": torchvision.datasets.ImageNet,
            "FGVCAircraft": torchvision.datasets.FGVCAircraft,
            "CIFAR10" : torchvision.datasets.CIFAR10,
            "CIFAR100": torchvision.datasets.CIFAR100,
            #"FakeData": torchvision.datasets.FakeData,#a random data
            #"FMNIST": torchvision.datasets.FashionMNIST,
            #"Flickr8": torchvision.datasets.Flickr8k,#needs to be downloaded manually
            #"Flickr30": torchvision.datasets.Flickr30k,#needs to be downloaded manually
            #"ImageNet": torchvision.datasets.ImageNet,#RuntimeError: The archive ILSVRC2012_devkit_t12.tar.gz is not present in the root directory or is corrupted. You need to download it externally and place it in ./Data.
            #"LSUN": torchvision.datasets.LSUN,#needs to be downloaded manually
            #"MNIST": torchvision.datasets.MNIST,
            #"Places365": torchvision.datasets.Places365,too big
            "STL10": torchvision.datasets.STL10,#(image, target) where target is index of the target class.
            #"CalebA": torchvision.datasets.CelebA,
            "SVHN": torchvision.datasets.SVHN,#	(image, target) where target is index of the target class.
            #"QMNIST": torchvision.datasets.QMNIST,#?CvtModel
            #"EMNIST": torchvision.datasets.EMNIST,#?TypeError: EMNIST.__init__() missing 1 required positional argument: 'split'
            #"KMNIST": torchvision.datasets.KMNIST,#?
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
vision_models = {
    "Swin": SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224"),
    #"DeiT": DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-base-distilled-patch16-224'),
    "DINOv2": AutoModel.from_pretrained('facebook/dinov2-base'),
    #"ViTH":  ViTHybridModel.from_pretrained("google/vit-hybrid-base-bit-384"),
    #"DPT": DPTModel.from_pretrained("Intel/dpt-large"),
    "ViT": ViTForImageClassification.from_pretrained('google/vit-base-patch16-224'),
    "BEiT": BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224'),
    #"CVT": CvtModel.from_pretrained("microsoft/cvt-13"),
    #"LeViT": LevitModel.from_pretrained("facebook/levit-128S"),
    #"ConvNeXT": ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224-22k"),
    #"MobileViT": MobileViTForImageClassification.from_pretrained("apple/mobilevit-small"),
    "PVTv2": PvtV2ForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0"),
}
transforms = {
    "ViT":  ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'),
    "Swin": AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224"),
    "DINOv2": AutoImageProcessor.from_pretrained('facebook/dinov2-base'),
    "BEiT": BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224'),
    "PVTv2": AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0"),
    
}
for name, model in vision_models.items():
    if name in ["Swin", "ViT", "BEiT"]:
        # Remove the classifier layer in Swin
        model.classifier = torch.nn.Identity()
    elif name in ["PVTv2"]:
        # Remove the classifier layer in Swin
        model.classifier = torch.nn.Identity()
    elif name == "DINOv2":
        # DINOv2 models often do not have a classifier attached, so nothing to remove here
        pass

parser = argparse.ArgumentParser()
parser.add_argument("--teachers", nargs="+", type=str, default=list(vision_models.keys()))
parser.add_argument("--datasets", nargs="+", type=str, default=list(datasets.keys()))
args = parser.parse_args()    
batch_size = 64
num_workers = 4
for dataset_name in args.datasets:
    for teacher_name in args.teachers:
        output = "EmbeddingsVit/"+teacher_name + "_" + dataset_name
        teacher = vision_models[teacher_name]
        processor = transforms[teacher_name]
        data = datasets[dataset_name](root = "./Data",download = True, transform=transforms[teacher_name])
        dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        teacher.eval()
        teacher.to(device)
        dim = embedder_size[teacher_name]
        emb = torch.empty((0, dim), dtype=torch.float32).to("cpu")
        print(0)
        for batch in tqdm(dataloader):
            img, label = batch
            if dataset_name!= "cub":
                img = torch.stack(img['pixel_values'], dim=0)
                if img.shape[0]==1:
                    img = img.squeeze(0)
            out = teacher(img.to(device))
            if teacher_name == "DINOv2":
                out = out.pooler_output
            else:
                out = out.logits
            emb = torch.cat((emb, out.cpu().detach()), 0)
        print("saving" + output)
        np.save(output+'.npy', emb.cpu().detach().numpy()) 
        