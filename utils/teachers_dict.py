import torchvision.models as models
from torch import nn
import torch
from transformers import DPTModel, PvtV2ForImageClassification, ViTHybridModel, CvtModel, LevitConfig, LevitModel, AutoImageProcessor, AutoModel, AutoFeatureExtractor, SwinForImageClassification, MobileViTFeatureExtractor, MobileViTForImageClassification, ViTImageProcessor, ViTForImageClassification, AutoFeatureExtractor, DeiTForImageClassificationWithTeacher, BeitImageProcessor, BeitForImageClassification

teachers_dict = {
    "resnet18": models.resnet18(pretrained = True),
    "squeezenet": models.squeezenet1_0(pretrained = True),
    "densenet": models.densenet161(pretrained = True),#
    "googlenet": models.googlenet(pretrained = True),
    "shufflenet": models.shufflenet_v2_x1_0(pretrained = True),#
    "mobilenet": models.mobilenet_v2(pretrained = True),#
    "resnext50_32x4d": models.resnext50_32x4d(pretrained = True),
    "wide_resnet50_2": models.wide_resnet50_2(pretrained = True),
    "mnasnet": models.mnasnet1_0(pretrained = True),#
}
teachers_dict_vit = {
    "Swin": SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224", return_dict=False),
    "ViT": ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', return_dict=False),
    "DINOv2": AutoModel.from_pretrained('facebook/dinov2-base', return_dict=False),
    "BEiT": BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224', return_dict=False),
    "PVTv2": PvtV2ForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0", return_dict=False),
}
embedder_size = {
    "resnet18": 512,
    "squeezenet": 512,
    "densenet": 2208,
    "googlenet": 1024,
    "shufflenet": 1024,
    "mobilenet": 1280,
    "resnext50_32x4d": 2048,
    "wide_resnet50_2": 2048,
    "mnasnet": 1280,
    "DINOv2": 768,
    "ViT": 768,
    "Swin": 1024,
    "BEiT": 768,
    "MobileViT": 640,
    "ConvNeXT": 1024,
    "PVTv2": 256
}
class ModelImageTransform:
    def __init__(self, model_name):

        self.transforms = {
            "ViT": ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', return_dict=False),
            "Swin": AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224", return_dict=False),
            "DINOv2": AutoImageProcessor.from_pretrained('facebook/dinov2-base', return_dict=False),
            "BEiT": BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224', return_dict=False),
            "PVTv2": AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0", return_dict=False),
        }
        
        self.processor = self.transforms[model_name]
    
    def __call__(self, image):

        processed_image = self.processor(images=image, return_tensors="pt")
        return processed_image["pixel_values"].squeeze(0)
    
    
class EmbedderFromTorchvision:
    def __new__(cls, name, *args, **kwargs):
        model = teachers_dict[name]
        model.eval()
        if name in [
            "resnet18",
            "shufflenet",
            "resnext50_32x4d",
            "wide_resnet50_2",
            "googlenet",
        ]:
            model.fc = nn.Identity()
        elif name in ["squeezenet"]:
            model.classifier = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif hasattr(model, "classifier"):
            model.classifier = nn.Identity()
        return model

class EmbedderFromViT:
    def __new__(cls, name, *args, **kwargs):
        model = teachers_dict_vit[name]
        if name in ["Swin", "ViT", "BEiT"]:
            # Remove the classifier layer in Swin
            model.classifier = torch.nn.Identity()
        elif name in ["PVTv2"]:
            # Remove the classifier layer in Swin
            model.classifier = torch.nn.Identity()
        return model

if __name__ == "__main__":
    import torch

    image = torch.randn(1, 3, 225, 225)
    out_sizes = {}
    for name in teachers_dict.keys():
        model = EmbedderFromTorchvision(name)
        print(model(image).shape)
        out_sizes[name] = model(image).shape[1]

    print(out_sizes)