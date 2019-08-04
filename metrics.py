from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file


X, y = load_svmlight_file('/Users/johnny/Code/liblinear-master/news20.binary')

print(y)
lr = LogisticRegression(solver = 'lbfgs')
model = lr.fit(X, y)
1/0
ignore_header = True

y_truth = []
y_pred_score = []
y_pred = []
fn1 = 'fl_prediction.txt'
fn2 = 'prediction'
for line in open(fn1, 'r').readlines():
    if ignore_header:
        ignore_header = False
        continue
    truth, pred, _ = line.strip().split(' ')
    y_truth.append(float(truth))
    if pred > _:
        y_pred.append(1)
    else:
        y_pred.append(-1)
    y_pred_score.append(float(pred))

#fpr, tpr, thresholds = roc_curve(y_truth, y_pred_score, pos_label=2)
a = f1_score(y_truth, y_pred)
#a = auc(fpr, tpr)
print(a)
#print(y_pred_score)