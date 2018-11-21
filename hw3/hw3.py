"""
hw3.py
Artur Rodrigues Rocha Neto
artur.rodrigues26@gmail.com
NOV/2018

Código-fonte do homework #03 de Inteligência Computacional Aplicada 2018.2
Requisitos: Python 3.5+, numpy, pandas, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import NearestCentroid as DMC
from sklearn.naive_bayes import GaussianNB as CQG
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC as SVM
from sklearn.neural_network import MLPClassifier as MLP

def reduction_pca(X, n):
	pca = PCA(n_components=n)
	pca.fit(X)
	X = pca.transform(X)
	
	return X

def reduction_lda(X, y, n):
	pca = LDA(n_components=n)
	pca.fit(X, y)
	X = pca.transform(X)
	
	return X

def sensitivity(tp, fn):
	return tp / (tp + fn)

def specificity(tn, fp):
	return tn / (tn + fp)

def do_normalize(X_train, X_test):
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	
	return X_train, X_test

def classify(classifiers, X_train, y_train, X_test, y_test, normalize=False):
	ans = {key: {"score" : None, "sens" : None, "spec" : None}
	       for key, value in classifiers.items()}
	
	for name, classifier in classifiers.items():
		if normalize:
			X_train, X_test = do_normalize(X_train, X_test)
		
		classifier.fit(X_train, y_train)
		score = classifier.score(X_test, y_test)
		y_pred = classifier.predict(X_test)
		tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
		sens = sensitivity(tp, fn)
		spec = specificity(tn, fp)
		
		ans[name]["score"] = score
		ans[name]["sens"] = sens
		ans[name]["spec"] = spec
		
		print("{} DONE!".format(name))

	return ans
	
def sumary(ans):
	print("CLASSIF\t\tSCORE\tSENSI\tSPECIF")
	for n in ans:
		name = n
		score = round(ans[n]["score"]*100, 2)
		sens = round(ans[n]["sens"]*100, 2)
		spec = round(ans[n]["spec"]*100, 2)
		print("{}\t\t{}\t{}%\t{}%".format(name, score, sens, spec))

def set_datasets():
	training = pd.read_csv("data/training.csv")
	testing = pd.read_csv("data/testing.csv")
	reduced = pd.read_csv("data/reducedSet.csv")
	
	df = training[list(reduced["x"]) + ["Class"]]
	df.to_csv("data/data_training.csv", index=None)
	
	df = testing[list(reduced["x"]) + ["Class"]]
	df.to_csv("data/data_testing.csv", index=None)

if __name__ == "__main__":
	classifiers = {"KNN1"           : KNN(n_neighbors=1),
				   "KNN2"           : KNN(n_neighbors=2),
				   "KNN3"           : KNN(n_neighbors=3),
				   "KNN4"           : KNN(n_neighbors=4),
				   "KNN5"           : KNN(n_neighbors=5),
				   "LDA"            : LDA(n_components=1),
				   "SVMlinearAuto"  : SVM(kernel="linear", gamma="auto"),
				   "SVMradialAuto"  : SVM(kernel="rbf", gamma="auto"),
				   "SVMpolyAuto"    : SVM(kernel="poly", gamma="auto"),
				   "SVMlinearScale" : SVM(kernel="linear", gamma="scale"),
				   "SVMradialScale" : SVM(kernel="rbf", gamma="scale"),
				   "SVMpolyScale"   : SVM(kernel="poly", gamma="scale"),
	               "DMC"            : DMC(),
	               "CQG"            : CQG(),
	               "MLP"            : MLP()}
	
	training = pd.read_csv("data/data_training.csv")
	testing = pd.read_csv("data/data_testing.csv")
	
	X_train = training.drop(["Class"], axis=1)
	y_train = training["Class"]
	X_test = testing.drop(["Class"], axis=1)
	y_test = testing["Class"]
	
	ans = classify(classifiers, X_train, y_train, X_test, y_test)
	sumary(ans)
	
