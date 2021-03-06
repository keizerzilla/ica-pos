import pandas as pd
import matplotlib.pyplot as plt
import seaborn

"""
Item #02: Análise monovariada por classe dos preditores
- plotar histogramas
- calcular média, desvio padrão e assimetria
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

f, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(10, 8))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for ix, c in enumerate(classes, start=1):
	# boxplot
	col = df.loc[df["class"] == c].drop(["class"], axis=1)
	plt.subplot(3, 2, ix)
	seaborn.boxplot(data=col)
	plt.title("Classe {}".format(c))


plt.savefig(figpath + "monovariada-por-classe.png")
plt.clf()

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
		fig.savefig("{}hist_class-{}_p-{}.png".format(figpath, c, p), bbox_inches="tight")
		plt.close(fig)
		
		
		mean = round(col[p].mean(), 4)
		std = round(col[p].std(), 4)
		var = round(col[p].var(), 4)
		skewness = round(col[p].skew(), 4)

		new_entry = pd.DataFrame([[c, p, mean, std, var, skewness]], columns=monoclass_header)
		monoclass = monoclass.append(new_entry)
		
		print("Análise do preditor {} da classe {} OK".format(p, c))

# coloca zero em class4 para nao deixar de fora
new_entry = pd.DataFrame([["class4", 0]], columns=countsamples_header)
countsamples = countsamples.append(new_entry)

# salva resultados em arquivo
monoclass.to_csv(result_file, index=False)
countsamples.to_csv(samplecount_file, index=False)

# cria barplot com numero de amostras
ax = countsamples.plot.bar(x="class", y="num_samples", legend=False)
ax.set_title("Número de amostras por classe de vidro")
ax.set_xlabel("Classes")
ax.set_ylabel("Amostras")
plt.savefig(figpath+"barplot_numsamples.png", bbox_inches="tight")


