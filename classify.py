import numpy as np

matrix =  np.loadtxt(open("Dataset_train_X_vector.csv","rb"),delimiter=",")
X = matrix

#print(X)

Y = [0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0]

Y_num = [int(numeric_string) for numeric_string in Y]

Y = np.asarray(Y_num)
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf = clf.fit(X, Y)

GaussianNB(priors=None)

P = np.loadtxt(open("Test.csv","rb"),delimiter=",")

A = clf.predict(P)
#print(A)

T = [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1,1,1]

T_num = [int(numeric_string) for numeric_string in T]

T = np.asarray(T_num)

#print(T)
from sklearn.metrics import accuracy_score
#y_pred = [0, 2, 1, 3]
#y_true = [0, 1, 2, 3]
score = accuracy_score(T, A)
print(score)
