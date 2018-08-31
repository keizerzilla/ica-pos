import pandas as pd
import matplotlib.pyplot as plt

# paths e arquivos
figpath = "figures/"
dataset = "datasets/glass.dat"
resfile = "analyzers.dat"

# carregando dados e limpando coluna id (dencessaria)
df = pd.read_csv(dataset)
df = df.drop(["id"], axis=1)

# lista de preditores (exclui-se a coluna da classe)
predictors = list(df.drop(["class"], axis=1).columns.values)

# analise monovariada
# - histograma
# - media
# - desvio padrao
# - assimetria
columns = ["predictor", "mean", "std", "skewness"]
analyzers = pd.DataFrame(columns=columns)
for p in predictors:
	fig = plt.figure()
	plt.hist(df[p], bins=5)
	plt.title("Preditor {}".format(p))
	plt.grid(b=True)
	fig.savefig("{}hist_{}.png".format(figpath, p))
	plt.close(fig)

	mean = df[p].mean()
	std = df[p].std()
	skewness = df[p].skew()

	new_entry = pd.DataFrame([[p, mean, std, skewness]], columns=columns)
	analyzers = analyzers.append(new_entry)


# debuf
analyzers.to_csv(resfile, index=False)
