import pandas as pd
import seaborn
import matplotlib.pyplot as plt

"""
Item #02: Analise monovariada por classe dos preditores
- Plotar histogramas
- Calcular media, desvio padrao e assimetria
"""

# setando estilo e outras configs
seaborn.set()

# paths e arquivos
dataset = "datasets/glass.dat"
figpath = "figures/item2/"
result_file = "results/item2.dat"
samplecount_file = "results/samples_per_class.dat"

# carregando dados e limpando coluna id (dencessaria)
df = pd.read_csv(dataset)
df = df.drop(["id"], axis=1)

# lista de preditores (exclui-se a coluna da classe)
predictors = list(df.drop(["class"], axis=1))

# analise monovariada por classe
# - classe
# - histograma
# - media
# - desvio padrao
# - assimetria
monoclass_header = ["class", "predictor", "mean", "std", "var", "skewness"]
countsamples_header = ["class", "num_samples"]

monoclass = pd.DataFrame(columns=monoclass_header)
countsamples = pd.DataFrame(columns=countsamples_header)

classes = df["class"].unique()

for c in classes:
	col = df.loc[df["class"] == c]
	num_samples = len(col.index)
	new_entry = pd.DataFrame([[c, num_samples]], columns=countsamples_header)
	countsamples = countsamples.append(new_entry)
	for p in predictors:
		fig = plt.figure()
		plt.hist(col[p], bins=5)
		plt.title("classe {} - preditor {}".format(c, p))
		plt.grid(b=True)
		fig.savefig("{}hist_class-{}_p-{}.png".format(figpath, c, p))
		plt.close(fig)

		mean = col[p].mean()
		std = col[p].std()
		var = col[p].var()
		skewness = col[p].skew()

		new_entry = pd.DataFrame([[c, p, mean, std, var, skewness]], columns=monoclass_header)
		monoclass = monoclass.append(new_entry)
		
		print("Analise do preditor {} da classe {} OK".format(p, c))
	
# salva resultados em arquivo
monoclass.to_csv(result_file, index=False)
countsamples.to_csv(samplecount_file, index=False)


