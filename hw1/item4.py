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
figpath = "figures/item4/"

# carregando dados e limpando coluna id (dencessaria)
df = pd.read_csv(dataset)
df = df.drop(["id"], axis=1)
classes = df["class"].unique()
df = df.drop(["class"], axis=1)

# normalizacao
data = np.array(df)
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

# PCA
pca = PCA(n_components=2)
projection = pca.fit_transform(data)

# plot
plt.scatter(projection[:,0], projection[:,1])
plt.title("PCA do conjunto de dados")
