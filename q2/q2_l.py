# Sahil Singh Rathore
# B20227
# Mob No: 9559176048

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Used sklearn only for train test split and accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def likelihood(x, u, v):
    det = np.sqrt(np.linalg.det(v))
    t1 = np.matmul(-0.5*(x-u).T, np.linalg.inv(v))
    t2 = np.matmul(t1, (x-u))
    e = np.exp(t2)
    p = e/(2*np.pi*det)

    return p


def post_prob(x, u, v):

    lh = likelihood(x, u, v)
    ans = lh/3

    return ans


def predict(arr, cov1, cov2, cov3, c):
    y_pred = []
    class1 = []
    class2 = []
    class3 = []
    for i in arr:
        p1 = post_prob(i, df1_mean, cov1)
        p2 = post_prob(i, df2_mean, cov2)
        p3 = post_prob(i, df3_mean, cov3)
        if p1 > p2 and p1 > p3:
            y_pred.append(1)
            class1.append(i)
        elif p2 > p1 and p2 > p3:
            y_pred.append(2)
            class2.append(i)
        else:
            y_pred.append(3)
            class3.append(i)

    y_true = np.array(y_test)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    fsc = f1_score(y_true, y_pred, average='weighted')

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F-score:", fsc)

    class1 = np.array(class1).T
    class2 = np.array(class2).T
    class3 = np.array(class3).T

    class1_x = class1[0]
    class2_x = class2[0]
    class3_x = class3[0]
    class1_y = class1[1]
    class2_y = class2[1]
    class3_y = class3[1]

    plt.scatter(class1_x, class1_y, color='red', label='class1')
    plt.scatter(class2_x, class2_y, color='blue', label='class2')
    plt.scatter(class3_x, class3_y, color='green', label='class3')
    plt.legend()
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.title(f"C{c} classifier on linear data!")

    plt.show()


df1 = pd.read_csv("q2/data/l_class1.txt", sep=" ", header=None)
df1.columns = ['A', 'B']
df1_cov = df1.cov().to_numpy()
df1_mean = df1.mean().to_numpy()
df1['Y'] = 1

df2 = pd.read_csv("q2/data/l_class2.txt", sep=" ", header=None)
df2.columns = ['A', 'B']
df2_cov = df2.cov().to_numpy()
df2_mean = df2.mean().to_numpy()
df2['Y'] = 2

df3 = pd.read_csv("q2/data/l_class3.txt", sep=" ", header=None)
df3.columns = ['A', 'B']
df3_cov = df3.cov().to_numpy()
df3_mean = df3.mean().to_numpy()
df3['Y'] = 3

df = pd.concat([df1, df2, df3])
df = df.sample(frac=1).reset_index().drop('index', axis=1)

X = df.drop('Y', axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=29)

arr = X_test.to_numpy()


# C1 Classifier
var = (df1_cov[0][0]+df1_cov[1][1]+df2_cov[0][0] +
       df2_cov[1][1]+df3_cov[0][0]+df3_cov[1][1])/6

cov = np.array([[var, 0], [0, var]])
print("\nC1 Classifier")
predict(arr, cov, cov, cov, 1)


# C2 Classifier
cov = (df1_cov+df2_cov+df3_cov)/3
print("\nC2 Classifier")
predict(arr, cov, cov, cov, 2)


# C3 Classifier
cov1 = df1_cov.copy()
cov2 = df2_cov.copy()
cov3 = df3_cov.copy()
cov1[0][1] = 0
cov1[1][0] = 0
cov2[0][1] = 0
cov2[1][0] = 0
cov3[0][1] = 0
cov3[1][0] = 0
print("\nC3 Classifier")
predict(arr, cov1, cov2, cov3, 3)


# C4 Classifier
print("\nC4 Classifier")
predict(arr, df1_cov, df2_cov, df3_cov, 4)
