import seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
Item #04: Análise de componentes principais
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

# PCA
pca = PCA(n_components=2)
pca.fit(df)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)


