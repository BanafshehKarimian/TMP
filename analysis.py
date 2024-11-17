import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import top_k_accuracy_score
from torchmetrics.functional import auroc
import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall, CohenKappa
from utils.teachers_dict import teachers_dict
from Data.DataClass import *
model = 'resnext50_32x4d'
print("student:")
for task in ["CIFAR10", "FMNIST", "MNIST", "STL10", "SVHN", "QMNIST", "KMNIST"]:
    print(task)
    data = pd.read_csv('./output_student/' + model + '_student_'+ task +'_finetuning/predictions.csv')
    l = []
    num_classes = 10
    for i in range(num_classes):
        l.append(data['class_'+ str(i)])
    preds = torch.tensor(np.stack(l).transpose())
    targets = torch.tensor(np.array(data['target']))
    acc = Accuracy(task="multiclass", num_classes=num_classes)
    auc = auroc(preds, targets, num_classes=num_classes, average='macro', task='multiclass')
    print("Accuracy:{}, AUC: {}".format(acc(preds, targets), auc))
'''
for model in teachers_dict.keys():
    print(model)
    data = pd.read_csv('./output_teacher/'+model+'_'+ task +'_teacher_finetuning/predictions.csv')
    l = []
    num_classes = data_class_dict[task]
    for i in range(num_classes):
        l.append(data['class_'+ str(i)])
    preds = torch.tensor(np.stack(l).transpose())
    targets = torch.tensor(np.array(data['target']))
    acc = Accuracy(task="multiclass", num_classes=num_classes)
    auc = auroc(preds, targets, num_classes=num_classes, average='macro', task='multiclass')
    print("Accuracy:{}, AUC: {}".format(acc(preds, targets), auc))
'''

