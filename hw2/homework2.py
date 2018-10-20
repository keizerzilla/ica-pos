"""
REFERENCIAS
-----------

01- http://bagrow.info/dsv/LEC10_notes_2014-02-13.html

"""

import os
import re
import math
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, train_test_split

def clean_data():
	try:
		os.mkdir("data")
	except:
		print("OPS! Diretório já existe!")
	
	inputs = ["solTestX.txt",
		      "solTestXtrans.txt",
		      "solTrainX.txt",
		      "solTrainXtrans.txt"]
	
	for input in inputs:
		output = "data/"+input
		input = "original/"+input
		with open(input, "r") as infile:
			lines = infile.readlines()
			with open(output, "w") as outfile:
				header = lines[0].replace("\t", ",")
				header = header.replace("\"", "")
				outfile.write(header)
				for i in range(1, len(lines)):
					newline = lines[i].replace("\t", ",")
					newline = re.sub("\"\d+\",", "", newline)
					outfile.write(newline)
	
	inputs = ["solTestY.txt",
		      "solTrainY.txt"]

	for input in inputs:
		output = "data/"+input
		input = "original/"+input
		with open(input, "r") as infile:
			lines = infile.readlines()
			with open(output, "w") as outfile:
				header = "y\n"
				outfile.write(header)
				for i in range(1, len(lines)):
					newline = lines[i].replace("\t", ",")
					newline = re.sub("\"\d+\",", "", newline)
					outfile.write(newline)
	
	df1 = pd.read_csv("data/solTestX.txt")
	df2 = pd.read_csv("data/solTrainX.txt")
	frames = [df1, df2]
	result = pd.concat(frames)
	result.to_csv("data/solX.txt", index=False)

	df3 = pd.read_csv("data/solTestY.txt")
	df4 = pd.read_csv("data/solTrainY.txt")
	frames = [df3, df4]
	result = pd.concat(frames)
	result.to_csv("data/solY.txt", index=False)

	df5 = pd.read_csv("data/solX.txt")
	df6 = pd.read_csv("data/solY.txt")
	df5["y"] = df6
	df5.to_csv("data/sol.txt", index=False)

def transf_yeojohnson(df, output):
	yeojohnson = df.copy()
	predictors = list(df.filter(regex="^(?!FP\d+)", axis=1))
	for p in predictors:
		pt = PowerTransformer(method="yeo-johnson", standardize=False)
		data = np.array(df[p]).reshape(-1, 1)
		pt.fit(data)
		ans = np.ravel(pt.transform(data))
		yeojohnson[p] = pd.Series(ans)

	yeojohnson.to_csv(output, index=False)
	return yeojohnson

def transf_boxcox(df, bonus, output):
	boxcox = df.copy()
	predictors = list(df.filter(regex="^(?!FP\d+)", axis=1))
	for p in predictors:
		pt = PowerTransformer(method="box-cox", standardize=False)
		data = np.array(df[p] + bonus).reshape(-1, 1)
		pt.fit(data)
		ans = np.ravel(pt.transform(data))
		boxcox[p] = pd.Series(ans)

	boxcox.to_csv(output, index=False)
	return boxcox

def statistics(df, output):
	predictors = list(df.filter(regex="^(?!FP\d+)", axis=1))
	cols = ["predictor", "mean", "std", "var", "skewness"]
	monovariate = pd.DataFrame(columns=cols)
	for p in predictors:
		mean = round(df[p].mean(), 2)
		std = round(df[p].std(), 2)
		var = round(df[p].var(), 2)
		skewness = round(df[p].skew(), 2)
		new_entry = pd.DataFrame([[p, mean, std, var, skewness]], columns=cols)
		monovariate = monovariate.append(new_entry)
	
	monovariate.to_csv(output, index=False)

def histogram(df, title):
	predictors = list(df.filter(regex="^(?!FP\d+)", axis=1))
	fig, ax = plt.subplots(5, 4)
	row = 0
	col = 0
	for p in predictors:
		ax[row, col].hist(df[p], bins=5)
		ax[row, col].set_title(str(p))
		
		col = col + 1
		if col >= 4:
			col = 0
			row = row + 1
	
	fig.suptitle(title)
	plt.subplots_adjust(hspace=0.8, wspace=0.2)
	plt.show()

def heatmap(df, title):
	data = df.copy()
	data = data.filter(regex="^(?!FP\d+)", axis=1)
	
	corr = data.corr()
	annot = corr.round(decimals=2)
	upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
	to_drop = [col for col in upper.columns if any(upper[col] >= 0.9)]
	super_corr = [col for col in upper.columns if any(upper[col] >= 0.98)]
	print("drops sugeridos: " + str(to_drop))
	print("super correlacoes: " + str(super_corr))
	
	hm = seaborn.heatmap(corr, vmin=-1.0, vmax=1.0, cmap="Purples", annot=annot)
	
	for item in hm.get_xticklabels():
		item.set_rotation(45)
	
	for item in hm.get_yticklabels():
		item.set_rotation(360)
	
	plt.title(title)
	plt.show()

def islinear(df_in, df_out, title):
	predictors = list(df_in.filter(regex="^(?!FP\d+)", axis=1))
	fig, ax = plt.subplots(5, 4)
	row = 0
	col = 0
	y = np.ravel(df_out)
	for p in predictors:
		x = np.array(df[p])
		slope, intercept, r_value, p_value, std_err = linregress(x, y)
		line = intercept + slope*x
		
		ax[row, col].scatter(x, y, s=1)
		ax[row, col].plot(x, line, "r")
		ax[row, col].set_title("{} (r = {})".format(p, round(r_value, 2)))
		
		col = col + 1
		if col >= 4:
			col = 0
			row = row + 1
	
	seaborn.set()
	fig.suptitle(title)
	plt.subplots_adjust(hspace=0.8, wspace=0.2)
	plt.show()

def ordinary_linear_regression(X_tr, y_tr, X_tst, y_tst):
	lm = linear_model.LinearRegression()
	lm.fit(X_tr, y_tr)
	y_pred = lm.predict(X_tst)
	rmse = math.sqrt(metrics.mean_squared_error(y_tst, y_pred))
	r2 = lm.score(X_tst, y_tst)
	
	return (rmse, r2)

def ridge_regression(X_tr, y_tr, X_tst, y_tst, alpha):
	lm = linear_model.Ridge(alpha=alpha)
	lm.fit(X_tr, y_tr)
	y_pred = lm.predict(X_tst)
	rmse = math.sqrt(metrics.mean_squared_error(y_tst, y_pred))
	r2 = lm.score(X_tst, y_tst)
	coef = lm.coef_
	
	return (rmse, r2, coef)

def kfold_ordinary_linear_regression(X, y, k):
	rmse_l = []
	r2_l = []
	
	kf = KFold(n_splits=k, shuffle=True)
	i = 1
	for train_index, test_index in kf.split(X):
		X_tr, X_tst = X[train_index], X[test_index]
		y_tr, y_tst = y[train_index], y[test_index]
		
		rmse, r2 = ordinary_linear_regression(X_tr, y_tr, X_tst, y_tst)
		rmse_l.append(rmse)
		r2_l.append(r2)
		
		print("[{}] RMSE: {}, R2: {}".format(i, round(rmse, 2), round(r2, 2)))
		i = i + 1
	
	return (rmse_l, r2_l)

def tunned_ridge_regression(X, y, pred):
	alphas = []
	coefs = pd.DataFrame(columns=pred)
	acc_cols = ["RMSE", "R2"]
	acc = pd.DataFrame(columns=acc_cols)
	
	X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.3)
	
	for alpha in np.linspace(0.0000000000001, 1):
		rmse, r2, coef = ridge_regression(X_tr, y_tr, X_tst, y_tst, alpha)
		alphas.append(alpha)
		
		newcoef = pd.DataFrame(coef.reshape(1,-1), columns=pred)
		coefs = coefs.append(newcoef)
		newacc = pd.DataFrame([[rmse, r2]], columns=acc_cols)
		acc = acc.append(newacc)
	
	return (acc, alphas, coefs)

def pls_regression(X, y, k, components):
	pls = PLSRegression(n_components=components)
	rmse_l = []
	r2_l = []
	
	kf = KFold(n_splits=k, shuffle=True)
	for train_index, test_index in kf.split(X):
		X_tr, X_tst = X[train_index], X[test_index]
		y_tr, y_tst = y[train_index], y[test_index]
		
		pls.fit(X_tr, y_tr)
		r2 = pls.score(X_tst, y_tst)
		y_pred = pls.predict(X_tst)
		rmse = math.sqrt(metrics.mean_squared_error(y_tst, y_pred))
		rmse_l.append(rmse)
		r2_l.append(r2)
	
	return (rmse_l, r2_l)

def avg(l):
	return sum(l) / len(l)

if __name__ == "__main__":
	# dados da divisao original do R
	print("QUESTAO 00")
	X_tr = pd.read_csv("data/solTrainXtrans.txt")
	y_tr = pd.read_csv("data/solTrainY.txt")
	X_tst = pd.read_csv("data/solTestXtrans.txt")
	y_tst = pd.read_csv("data/solTestY.txt")
	X_tr = X_tr.filter(regex="^(?!FP\d+)", axis=1)
	X_tst = X_tst.filter(regex="^(?!FP\d+)", axis=1)
	rmse, r2 = ordinary_linear_regression(X_tr, y_tr, X_tst, y_tst)
	print("[-] RMSE: {}, R2: {}".format(round(rmse, 2), round(r2, 2)))
	
	# dados para questoes 1 a 3
	X = pd.read_csv("data/solBoxCox.txt").filter(regex="^(?!FP\d+)", axis=1)
	pred = list(X)
	y = pd.read_csv("data/solY.txt")
	X = np.array(X)
	y = np.ravel(y)
	
	# questao 1
	print("QUESTAO 01")
	rmse_l, r2_l = kfold_ordinary_linear_regression(X, y, 10)
	print("RMSE_mean: {}, R2_mean: {}".format(avg(rmse_l), avg(r2_l)))
	
	# questao 2
	print("QUESTAO 02")
	acc, alphas, coefs = tunned_ridge_regression(X, y, pred)
	print("RMSE_mean: {}, R2_mean: {}".format(acc["RMSE"].mean(), acc["R2"].mean()))
	print(acc)
	
	"""
	plt.plot(alphas, coefs)
	plt.title("Evolução dos coeficientes da regressão rígida")
	plt.xlabel("Lambda")
	plt.ylabel("Coeficiente")
	plt.legend(list(coefs), bbox_to_anchor=(1,0,0,1), loc="center left")
	plt.grid()
	plt.show()
	"""
	
	"""
	plt.scatter(alphas, acc["RMSE"])
	plt.scatter(alphas, acc["R2"])
	plt.legend(list(acc), loc="upper right")
	plt.title("Precisão dos modelos em termos de RMSE e R²")
	plt.xlabel("Lambda")
	plt.ylabel("Precisão")
	plt.grid()
	plt.show()
	"""
	
	# questao 3
	print("QUESTAO 03")
	rmse_l, r2_l = pls_regression(X, y, 10, 2)
	print("RMSE_mean: {}, R2_mean: {}".format(avg(rmse_l), avg(r2_l)))
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
