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


column_name = ["Method", "Model", "PCAM"]
datsets = ["PCAM"]
results_student = []
results_student_c = []
results_student_MSE = []
results_teacher = []

students = ["resnet18" , "squeezenet" , "densenet", "googlenet", "shufflenet", "mobilenet", "resnext50_32x4d", "wide_resnet50_2", "mnasnet"]

for model in students[:1]:
    results = ["NLL", model, ]
    for task in datsets:
        data = pd.read_csv('./output_student/' + model + '_student_'+ task +'_finetuning_/predictions.csv')
        l = []
        num_classes = 2
        for i in range(num_classes):
            l.append(data['class_'+ str(i)])
        preds = torch.tensor(np.stack(l).transpose())
        targets = torch.tensor(np.array(data['target']))
        acc = Accuracy(task="multiclass", num_classes=num_classes)
        results.append(balanced_accuracy_score(preds.argmax(1), targets).item()*100)
    results_student.append(results)

students_df = pd.DataFrame(results_student, columns = np.array(column_name).flatten())


for model in students:
    results = ["NoKD", model, ]
    for task in datsets:
        data = pd.read_csv('./output_teacher/' + model + '_'+ task +'_teacher_finetuning/predictions.csv')
        l = []
        num_classes = 2
        for i in range(num_classes):
            l.append(data['class_'+ str(i)])
        preds = torch.tensor(np.stack(l).transpose())
        targets = torch.tensor(np.array(data['target']))
        acc = Accuracy(task="multiclass", num_classes=num_classes)
        results.append(balanced_accuracy_score(preds.argmax(1), targets).item()*100)
    results_teacher.append(results)
teachers_df = pd.DataFrame(results_teacher, columns = np.array(column_name).flatten())
df_merged = pd.concat([students_df, teachers_df], ignore_index=True)#students_df_c, students_df_MSE, 
#for col in datsets:
#    df_merged[col] = df_merged[col].apply(lambda x: float("{:.2f}".format(x)))
df_merged = df_merged.set_index(["Method", "Model"])
style = df_merged.style.format("{:.2f}")#.hide(axis="index") #.format('${:,.2f}')
style = style.apply(lambda x: highlight_top_1(x, "bfseries:"), axis=0, subset=style.columns)
style = style.apply(lambda x: highlight_top_2(x, "underline:--rwrap"), axis=0, subset=style.columns)
col_format = "r|"
over_cols = None
for col in style.columns:
    if col == "Avg":
        col_format += "|"
    col_format += "c"


style.format('${:,.2f}')
latex = style.to_latex(
    column_format=col_format,
    multicol_align="|c|",
    siunitx=True,
    caption="caption",
)

latex = latex.replace("Cosine", "\multirow[c]{1}{*}{Cosine}")
latex = latex.replace("MSE", "\multirow[c]{1}{*}{MSE}")
latex = latex.replace("\multirow", "\midrule  \multirow")
latex = latex.replace("_", "-")
latex = latex.replace("$", "")
latex = latex.replace("ccccccc", "ccccccccc")

'''latex = latex.replace(r"\begin{tabular}", r"\resizebox{\textwidth}{!}{\begin{tabular}")
latex = latex.replace(r"\end{tabular}", r"\end{tabular}}")'''
print(latex)
table_path = "./table3.tex"
with open(table_path, "w") as f:
    f.write(latex)