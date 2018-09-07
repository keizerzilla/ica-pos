import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib.markers as mrk

"""
Item #03: Analise bivariada dos preditores usando pair plot
"""

# setando estilo e outras configs
seaborn.set()

# paths e arquivos
dataset = "datasets/glass.dat"
figpath = "figures/item3/scatter_matrix.png"

# carregando dados e limpando coluna id (dencessaria)
df = pd.read_csv(dataset)
df = df.drop(["id"], axis=1)
classes = df["class"].unique()

# plot

g = seaborn.PairGrid(df, hue="class")
g = g.map_diag(plt.hist, histtype="step", linewidth=4)
g = g.map_offdiag(plt.scatter, s=25)
handles = g._legend_data.values()
labels = g._legend_data.keys()
g.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=len(classes))
g.fig.subplots_adjust(top=0.98, bottom=0.02)
g.savefig(figpath)
plt.show()


