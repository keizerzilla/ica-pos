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

# setando estilo e outras configs
seaborn.set()

# paths e arquivos
dataset = "datasets/glass.dat"
pca_barplot = "figures/item4/pca_barplot.png"

# carregando dados e limpando coluna id (dencessaria)
df = pd.read_csv(dataset)
df = df.drop(["id"], axis=1)
classes = df["class"].unique()
df = df.drop(["class"], axis=1)

# normalizacao basica: centraliza na media e escala com desvio padrao
#data = np.array(df)
scaler = StandardScaler()
scaler.fit(df)
df = scaler.transform(df)

# PCA
pca = PCA(n_components=2)
projection = pca.fit_transform(df)
plt.scatter(projection[:,0], projection[:,1])
plt.show()

exit()

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


