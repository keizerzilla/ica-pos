import seaborn
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# setupzz
seaborn.set()

# carregando dadods
df = pd.read_csv("solX.txt").filter(regex="^(?!FP\d+)", axis=1)

# analise monovariada
# - histograma
# - media
# - variancia
# - assimetria
predictors = list(df)
columns = ["predictor", "mean", "std", "var", "skewness"]
monovariate = pd.DataFrame(columns=columns)
f, ax = plt.subplots(5, 4)
row = 0
col = 0
for p in predictors:
	#fig = plt.figure()
	ax[row, col].hist(df[p], bins=5)
	ax[row, col].set_title(str(p))
	
	col = col + 1
	if col >= 4:
		col = 0
		row = row + 1
	
	mean = round(df[p].mean(), 4)
	std = round(df[p].std(), 4)
	var = round(df[p].var(), 4)
	skewness = round(df[p].skew(), 4)
	new_entry = pd.DataFrame([[p, mean, std, var, skewness]], columns=columns)
	monovariate = monovariate.append(new_entry)
	
	print("Análise global do preditor {} OK".format(p))

plt.subplots_adjust(hspace=0.8, wspace=0.2)
plt.show()

# box-cox para remocao de assimetria dos dados


# salva resultados em arquivo
monovariate.to_csv("../analysis.txt", index=False)

