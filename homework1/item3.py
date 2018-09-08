import seaborn
import pandas as pd
import matplotlib.pyplot as plt

"""
Item #03: Análise bivariada dos preditores usando pairplot (a.k.a scatter matrix)
"""

# setando estilo e outras configs
seaborn.set()

# paths e arquivos
dataset = "datasets/glass.dat"
heatmap_file = "figures/item3/correlation_heatmap.png"
pairplot_file = "figures/item3/scatter_matrix.png"

# carregando dados e limpando coluna id (dencessaria)
df = pd.read_csv(dataset)
df = df.drop(["id"], axis=1)
classes = df["class"].unique()

# matriz de correlacao
corr = df.drop(["class"], axis=1).corr()
annot = corr.round(decimals=1)
seaborn.heatmap(corr, vmin=-1.0, vmax=1.0, linewidths=0.4, cmap="Purples", annot=annot)
plt.title("Correlação entre preditores")
plt.savefig(heatmap_file)

# pairplot
g = seaborn.PairGrid(df, hue="class")
g = g.map_diag(plt.hist, histtype="step", linewidth=4)
g = g.map_offdiag(plt.scatter, s=25)
handles = g._legend_data.values()
labels = g._legend_data.keys()
g.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=len(classes))
g.fig.subplots_adjust(top=0.98, bottom=0.02)
g.savefig(pairplot_file)


