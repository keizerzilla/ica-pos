import pandas as pd
import seaborn
import matplotlib.pyplot as plt

"""
Item #01: Analise monovariada global dos preditores
- Plotar histogramas
- Calcular media, desvio padrao e assimetria
"""

# setando estilo e outras configs
seaborn.set()

# paths e arquivos
dataset = "datasets/glass.dat"
figpath = "figures/item1/"
result_file = "results/item1.dat"

# carregando dados e limpando coluna id (dencessaria)
df = pd.read_csv(dataset)
df = df.drop(["id"], axis=1)

# lista de preditores (exclui-se a coluna da classe)
predictors = list(df.drop(["class"], axis=1))

# analise monovariada
# - histograma
# - media
# - desvio padrao
# - assimetria
columns = ["predictor", "mean", "std", "var", "skewness"]
monovariate = pd.DataFrame(columns=columns)
for p in predictors:
	fig = plt.figure()
	plt.hist(df[p], bins=5)
	plt.title("Analise monovariada: preditor {}".format(p))
	plt.grid(b=True)
	fig.savefig("{}hist_p-{}.png".format(figpath, p))
	plt.close(fig)

	mean = df[p].mean()
	std = df[p].std()
	var = df[p].var()
	skewness = df[p].skew()

	new_entry = pd.DataFrame([[p, mean, std, var, skewness]], columns=columns)
	monovariate = monovariate.append(new_entry)
	
	print("Analise global do preditor {} OK".format(p))

# salva resultados em arquivo
monovariate.to_csv(result_file, index=False)


