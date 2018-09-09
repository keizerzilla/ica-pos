import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
Item #04: Análise de componentes principais [[INCOMPLETO]]
- reter o 2 maiores componentes
- plotar os scatterplots das observações projetadas
"""

# funcao helper para desenho de vetores em grafico
def draw_vector(v0, v1, ax=None):
	ax = ax or plt.gca()
	arrowprops=dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
	ax.annotate('', v1, v0, arrowprops=arrowprops)

# setando estilo e outras configs
seaborn.set()

# paths e arquivos
dataset = "datasets/glass.dat"
pca_barplot = "figures/item4/pca_barplot.png"
figpath = "figures/item4/pca.png"

# carregando dados e limpando coluna id (dencessaria)
df = pd.read_csv(dataset)
df = df.drop(["id"], axis=1)
classes = df["class"].unique()
df = df.drop(["class"], axis=1)

# PCA
predictors = list(df)
num_pred = len(predictors)
f, axes = plt.subplots(num_pred, num_pred, sharex="col", sharey="row")
for i1, p1 in enumerate(predictors):
	for i2, p2 in enumerate(predictors, start=1):
		data = np.array(df[[p1, p2]])
		scaler = StandardScaler()
		scaler.fit(data)
		data = scaler.transform(data)
		
		pca = PCA(n_components=2)
		projection = pca.fit_transform(data)
		
		title = "PCA_{}-{}.png".format(p1, p2)
		
		plt.subplot(num_pred, num_pred, (i2 + i1*num_pred))
		
		plt.scatter(projection[:,0], projection[:,1])
		#plt.xlabel(str(p1))
		#plt.ylabel(str(p2))
		#plt.title(title)
		
		for length, vector in zip(pca.explained_variance_, pca.components_):
			v = vector * 3 * np.sqrt(length)
			draw_vector(pca.mean_, pca.mean_ + v)
			plt.axis('equal');


plt.savefig(figpath, bbox_inches="tight")

"""
predictors = list(df)
num_pred = len(predictors)
f, axes = plt.subplots(num_pred, num_pred, sharex=True, sharey=True)
for i1, p1 in enumerate(predictors):
	for i2, p2 in enumerate(predictors):
		data = np.array(df[[p1, p2]])
		#plt.subplot(num_pred, num_pred, (i2 + i1*num_pred), sharex="col")
		#plt.scatter(data[:,0], data[:,1])
		axes[i1, i2].scatter(data[:,0], data[:,1])
		#axes[i1, i2].set_xlim([np.min(data[:,0]), np.max(data[:,0])])
		#axes[i1, i2].set_ylim([np.min(data[:,1]), np.max(data[:,1])])

#plt.xlabel(" ".join(predictors))
#plt.ylabel(" ".join(predictors))
plt.show()
exit()
"""
