import numpy as np
import pandas as pd 
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

def highlight_top_1(s, props=""):
    max_value = s.max()
    is_max = s == max_value
    return [props if v else '' for v in is_max]

def highlight_top_2(s, props=""):
    max_values = s.nlargest(2)
    is_max = s.isin(max_values)
    return [props if v else '' for v in is_max]


column_name = ["Method", "Model", "CIFAR10", "FMNIST", "MNIST", "STL10", "SVHN", "QMNIST", "KMNIST"]
datsets = ["CIFAR10", "FMNIST", "MNIST", "STL10", "SVHN", "QMNIST", "KMNIST"]

students = ["resnet18" , "squeezenet" , "densenet", "googlenet", "shufflenet", "mobilenet", "resnext50_32x4d", "wide_resnet50_2", "mnasnet"]
rows_list = []

for model in students[:1]:
    for task in datsets:
        data = pd.read_csv('./output_student/' + model + '_student_'+ task +'_finetuning_/predictions.csv')
        l = []
        num_classes = 10
        for i in range(num_classes):
            l.append(data['class_'+ str(i)])
        preds = torch.tensor(np.stack(l).transpose())
        targets = torch.tensor(np.array(data['target']))
        acc = Accuracy(task="multiclass", num_classes=num_classes)
        rows_list.append({'Teacher' : "Multi-teacher",'Task' : task, 'ACC' : acc(preds, targets).item()*100})


teacher_model = ["squeezenet" , "densenet", "googlenet", "shufflenet", "mobilenet", "resnext50_32x4d", "wide_resnet50_2", "mnasnet"]
for teacher in teacher_model:
    for task in datsets:
        data = pd.read_csv('./output_student/' + "distill-one-"+teacher+"-to-resnet18_" + task + '_finetuning/predictions.csv')
        l = []
        num_classes = 10
        for i in range(num_classes):
            l.append(data['class_'+ str(i)])
        preds = torch.tensor(np.stack(l).transpose())
        targets = torch.tensor(np.array(data['target']))
        acc = Accuracy(task="multiclass", num_classes=num_classes)
        rows_list.append({'Teacher' : teacher,'Task' : task, 'ACC' : acc(preds, targets).item()*100})

df_merged = pd.DataFrame(rows_list)   
df_merged.to_csv('vision_fig1.csv', index=False) 
import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6))
cmap = sns.color_palette("tab20")
ax = sns.barplot(x='Task', y="ACC", data=df_merged, hue = "Teacher", estimator="median", errorbar=None, fill=True, alpha=.7, palette=cmap)
plt.xticks(rotation=30)
plt.ylim(62,100)
plt.tight_layout()
fig.savefig('Fig1.png', dpi=fig.dpi)