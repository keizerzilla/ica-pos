"""
REFERENCIAS
-----------

01- http://bagrow.info/dsv/LEC10_notes_2014-02-13.html

"""

import os
import re
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer

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

def transf_boxcox(df, output):
	boxcox = df.copy()
	predictors = list(df.filter(regex="^(?!FP\d+)", axis=1))
	for p in predictors:
		pt = PowerTransformer(method="box-cox", standardize=False)
		data = np.array(df[p] + 1).reshape(-1, 1)
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
	model = lm.fit(X_tr, y_tr)
	r2 = lm.score(X_tst, y_tst)
	
	return r2


if __name__ == "__main__":
	#df = pd.read_csv("data/solX.txt")
	#df_in = transf_boxcox(df, "data/solBoxCox.txt")
	#df_out = pd.read_csv("data/solY.txt")
	
	X_tr = pd.read_csv("data/solTrainXtrans.txt").filter(regex="^(?!FP\d+)", axis=1)
	y_tr = pd.read_csv("data/solTrainY.txt")
	X_tst = pd.read_csv("data/solTestXtrans.txt").filter(regex="^(?!FP\d+)", axis=1)
	y_tst = pd.read_csv("data/solTestY.txt")
	
	r2 = ordinary_linear_regression(X_tr, y_tr, X_tst, y_tst)
	print(round(r2, 2))
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	