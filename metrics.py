from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file

ignore_header = True

y_truth = []
y_pred_score = []
y_pred = []
fn1 = 'fl_prediction.txt'
fn2 = 'prediction'
for line in open(fn1, 'r', encoding='utf-8').readlines():
    if ignore_header:
        ignore_header = False
        continue
    truth, pred, _, score = line.strip().split(' ')
    y_truth.append(float(truth))
    y_pred.append(float(pred))
    y_pred_score.append(float(score))

fpr, tpr, thresholds = roc_curve(y_truth, y_pred_score)
f1 = f1_score(y_truth, y_pred)
a = auc(fpr, tpr)
print(f1, a)
#print(y_pred_score)