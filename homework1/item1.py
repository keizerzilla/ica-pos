import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

"""
Item #01: Analise monovariada global dos preditores
- Plotar histogramas
- Calcular media, desvio padrao e assimetria
"""

# paths e arquivos
figpath = "figures/item1/"
dataset = "datasets/glass.dat"
resfile = "item1.dat"

# carregando dados e limpando coluna id (dencessaria)
df = pd.read_csv(dataset)
df = df.drop(["id"], axis=1)

# lista de preditores (exclui-se a coluna da classe)
predictors = list(df.drop(["class"], axis=1))

# PCA
#X = np.array(df.drop(["class"], axis=1))
#X = scale(X, axis=1)
#pca = PCA()
#pca.fit(X)
#print(pca.explained_variance_ratio_)
#print(np.sum(pca.explained_variance_ratio_))

# analise monovariada
# - histograma
# - media
# - desvio padrao
# - assimetria
columns = ["predictor", "mean", "std", "skewness"]
monovariate = pd.DataFrame(columns=columns)
for p in predictors:
	fig = plt.figure()
	plt.hist(df[p], bins=5)
	plt.title("Preditor {}".format(p))
	plt.grid(b=True)
	fig.savefig("{}hist_p-{}.png".format(figpath, p))
	plt.close(fig)

	mean = df[p].mean()
	std = df[p].std()
	skewness = df[p].skew()

	new_entry = pd.DataFrame([[p, mean, std, skewness]], columns=columns)
	monovariate = monovariate.append(new_entry)

# salva resultados em arquivo
monovariate.to_csv(resfile, index=False)


