"""
hw3.py
Artur Rodrigues Rocha Neto
artur.rodrigues26@gmail.com
NOV/2018

Código-fonte do Homework #03 de Inteligência Computacional Aplicada 2018.2
Requisitos: Python 3.5+, numpy, pandas, matplotlib, seaborn, scikit-learn
"""

import warnings
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LogisticClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

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

def do_skewremoval(X_train, X_test):
	power = PowerTransformer()
	power.fit(X_train)
	X_train = power.transform(X_train)
	X_test = power.transform(X_test)
	
	return X_train, X_test

def classify(classifiers, X_train, y_train, X_test, y_test, nm=True, sk=True):
	ans = {key: {"score" : None, "sens" : None, "spec" : None, "confmat" : None}
	       for key, value in classifiers.items()}
	
	if nm:
		X_train, X_test = do_normalize(X_train, X_test)
	
	if sk:
		X_train, X_test = do_skewremoval(X_train, X_test)
	
	for name, classifier in classifiers.items():
		classifier.fit(X_train, y_train)
		score = classifier.score(X_test, y_test)
		y_pred = classifier.predict(X_test)
		confmat = confusion_matrix(y_test, y_pred)
		tn, fp, fn, tp = confmat.ravel()
		sens = sensitivity(tp, fn)
		spec = specificity(tn, fp)
		
		ans[name]["score"] = score
		ans[name]["sens"] = sens
		ans[name]["spec"] = spec
		ans[name]["confmat"] = confmat
		
		score = round(score*100, 2)
		sens = round(sens*100, 2)
		spec = round(spec*100, 2)
		
		print("{:<16}{:<8}{:<8}{:<8}".format(name, score, sens, spec))

	return ans
	
def sumary(ans, msg="Sumary"):
	row_size = 38
	print("="*row_size)
	print(msg)
	print("-"*row_size)
	print("{:<16}{:<8}{:<8}{:<8}".format("CLASSIFIER", "SCORE", "SENS", "SPEC"))
	print("-"*row_size)
	for n in ans:
		name = n
		score = round(ans[n]["score"]*100, 2)
		sens = round(ans[n]["sens"]*100, 2)
		spec = round(ans[n]["spec"]*100, 2)
		print("{:<16}{:<8}{:<8}{:<8}".format(name, score, sens, spec))
	print("-"*row_size)
	print()

def confmatrix(ans, name):
	classes = ["unsuccessful", "successful"]
	df = pd.DataFrame(ans[name]["confmat"], index=classes, columns=classes)
	fig = plt.figure()
	
	try:
		heatmap = sb.heatmap(df, annot=True, fmt="d")
	except ValueError:
		raise ValueError("Valores na matrix de confusão devem ser inteiros!")
	
	heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0)
	heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45)
	plt.ylabel('Classe correta')
	plt.xlabel('Classe classificada')
	
	return fig

def set_datasets():
	training = pd.read_csv("data/training.csv")
	testing = pd.read_csv("data/testing.csv")
	reduced = pd.read_csv("data/reducedSet.csv")
	
	training = training[list(reduced["x"]) + ["Class"]]
	training.to_csv("data/data_training.csv", index=None)
	
	testing = testing[list(reduced["x"]) + ["Class"]]
	testing.to_csv("data/data_testing.csv", index=None)

if __name__ == "__main__":
	# desabilita mensagens de warning
	warnings.filterwarnings("ignore")
	
	"""
	# classificadores lineares
	linear = {"LDA" : LDA(solver="svd", n_components=1),
	          "logit" : LogisticClassifier(solver="liblinear")}
	
	# classificadores não-lineares
	nonlinear = {"MLP" : MLP(),
	             "QDA" : QDA(reg_param=1),
	             "SVM_radial" : SVM(kernel="rbf", C=1.41),
	             "KNN_manhattam" : KNN(metric="manhattan", n_neighbors=24)}
	"""
	
	"""
	# config massa antes do gridsearch
	nonlinear = {"MLP_{}".format(round(a, 2)) : MLP(hidden_layer_sizes=(200,), solver="sgd", learning_rate="constant", max_iter=300, power_t=0.4, )
	             for a in np.linspace(0.1, 1, 10)}
	"""
	
	# carregando dados
	training = pd.read_csv("data/training.csv")
	testing = pd.read_csv("data/testing.csv")
	X_train = training.drop(["Class"], axis=1)
	y_train = training["Class"]
	X_test = testing.drop(["Class"], axis=1)
	y_test = testing["Class"]
	
	# seleção de atributos baseados em variância (baixa variância = descarte)
	toel = 0.06
	train_var = X_train.var()
	train_var = train_var.where(train_var > toel).dropna()
	predictors = list(train_var.index)
	print("Preditores com variância >{}: {}".format(toel, len(predictors)))
	X_train = X_train[predictors]
	X_test = X_test[predictors]
	
	param_grid = [
		{"hidden_layer_sizes" : [(h,) for h in range(100,310,10)],
		 "activation" : ["identity", "logistic", "tanh", "relu"],
		 "solver" : ["sgd"],
		 "alpha" : [a for a in np.linspace(0.0001, 0.001, 10)],
		 "learning_rate" : ["constant"],
		 "learning_rate_init" : [l for l in np.linspace(0.001, 0.1, 10)],
		 "power_t" : [p for p in np.linspace(0.1, 0.5, 5)],
		 "max_iter" : [1000],
		 "momentum" : [m for m in np.linspace(0, 1, 10)]
		}
	]
	scores = ['precision', 'recall']
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()
		
		clf = GridSearchCV(MLP(), param_grid, cv=5, scoring='%s_macro' % score)
		clf.fit(X_train, y_train)
		
		print("Best parameters set found on development set:")
		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
		print()
		
		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		y_true, y_pred = y_test, clf.predict(X_test)
		print(classification_report(y_true, y_pred))
		print()
	
	# rodando classificadores
	#ans = classify(linear, X_train, y_train, X_test, y_test)
	#ans = classify(nonlinear, X_train, y_train, X_test, y_test)
	
	# plotando alguns resultados dumb
	#fig = confmatrix(ans, "SVM_radial")
	#plt.show()
	
