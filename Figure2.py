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
results_student = []
results_student_c = []
results_student_MSE = []
results_teacher = []

students = ["resnet18" , "squeezenet" , "densenet", "googlenet", "shufflenet", "mobilenet", "resnext50_32x4d", "wide_resnet50_2", "mnasnet"]
rows_list = []

for model in students:
    r = ["NLL"]
    for task in datsets:
        data = pd.read_csv('./output_student/' + model + '_student_'+ task +'_finetuning_/predictions.csv')
        l = []
        num_classes = 10
        for i in range(num_classes):
            l.append(data['class_'+ str(i)])
        preds = torch.tensor(np.stack(l).transpose())
        targets = torch.tensor(np.array(data['target']))
        acc = Accuracy(task="multiclass", num_classes=num_classes)
        rows_list.append({'Method' : "NLL",'Model' : model, "Task": task, 'ACC' : acc(preds, targets).item()*100})
        r.append(acc(preds, targets).item()*100)

for model in ["resnet18" , "squeezenet" , "googlenet", "shufflenet", "mnasnet"]:
    r = ["Cosine"]
    for task in datsets:
        data = pd.read_csv('./output_student/' + model + '_student_'+ task +'_finetuning_Cosine/predictions.csv')
        l = []
        num_classes = 10
        for i in range(num_classes):
            l.append(data['class_'+ str(i)])
        preds = torch.tensor(np.stack(l).transpose())
        targets = torch.tensor(np.array(data['target']))
        acc = Accuracy(task="multiclass", num_classes=num_classes)
        rows_list.append({'Method' : "Cosine",'Model' : model, "Task": task, 'ACC' : acc(preds, targets).item()*100})
        r.append(acc(preds, targets).item()*100)

students_df_c = pd.DataFrame(results_student_c, columns = np.array(column_name).flatten())

for model in ["resnet18" , "squeezenet" , "googlenet", "shufflenet", "mobilenet"]:
    for task in datsets:
        data = pd.read_csv('./output_student/' + model + '_student_'+ task +'_finetuning_MSE/predictions.csv')
        l = []
        num_classes = 10
        for i in range(num_classes):
            l.append(data['class_'+ str(i)])
        preds = torch.tensor(np.stack(l).transpose())
        targets = torch.tensor(np.array(data['target']))
        acc = Accuracy(task="multiclass", num_classes=num_classes)
        rows_list.append({'Method' : "L2",'Model' : model, "Task": task, 'ACC' : acc(preds, targets).item()*100})

students_df_MSE = pd.DataFrame(results_student_MSE, columns = np.array(column_name).flatten())

for model in students:
    for task in datsets:
        data = pd.read_csv('./output_teacher/' + model + '_'+ task +'_teacher_finetuning/predictions.csv')
        l = []
        num_classes = 10
        for i in range(num_classes):
            l.append(data['class_'+ str(i)])
        preds = torch.tensor(np.stack(l).transpose())
        targets = torch.tensor(np.array(data['target']))
        acc = Accuracy(task="multiclass", num_classes=num_classes)
        rows_list.append({'Method' : "No KD",'Model' : model, "Task": task, 'ACC' : acc(preds, targets).item()*100})

df_merged = pd.DataFrame(rows_list)    

import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6))
cmap = sns.color_palette("husl", 4, desat=0.5)
ax = sns.catplot(x="Model", y="ACC", data=df_merged, hue = "Method", estimator="median")
plt.xticks(rotation=30)
plt.tight_layout()
fig.savefig('Fig2.png', dpi=fig.dpi)
'''df_merged = pd.concat([students_df, teachers_df])#, students_df_c, students_df_MSE
del df_merged['Model']
df_mean = df_merged.groupby('Method').mean()
df_std = df_merged.groupby('Method').std()
df_std = df_std.add_suffix('_95CI')
df_std = 1.96*(df_std/np.sqrt(len(students)))
print(pd.concat([df_mean, df_std], axis=1))'''