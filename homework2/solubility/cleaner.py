import re
import os
import pandas as pd

try:
	os.mkdir("clean")
except:
	print("OPS! Diretorio jah existe!")

print("FORMATANDO DADOS VARIAVEIS INDEPENDETES")
inputs = ["solTestX.txt",
          "solTestXtrans.txt",
          "solTrainX.txt",
          "solTrainXtrans.txt"]

for input in inputs:
	output = "clean/"+input
	input = "original/"+input
	with open(input, "r") as infile:
		lines = infile.readlines()
		with open(output, "w") as outfile:
			header = lines[0].replace("\t", ",")
			header = header.replace("\"", "")
			outfile.write(header)
			for i in range(1, len(lines)):
				newline = lines[i].replace("\t", ",")
				newline = re.sub("\"\d+\",", "", newline)
				outfile.write(newline)
	print(output + " ok!")

print("FORMATANDO DADOS SAIDA")
inputs = ["solTestY.txt",
          "solTrainY.txt"]

for input in inputs:
	output = "clean/"+input
	input = "original/"+input
	with open(input, "r") as infile:
		lines = infile.readlines()
		with open(output, "w") as outfile:
			header = "y\n"
			outfile.write(header)
			for i in range(1, len(lines)):
				newline = lines[i].replace("\t", ",")
				newline = re.sub("\"\d+\",", "", newline)
				outfile.write(newline)
	print(output + " ok!")

print("CRIANDO MERGE DOS DATASETS")

df1 = pd.read_csv("clean/solTestX.txt")
df2 = pd.read_csv("clean/solTrainX.txt")
frames = [df1, df2]
result = pd.concat(frames)
result.to_csv("clean/solX.txt", index=False)

df3 = pd.read_csv("clean/solTestY.txt")
df4 = pd.read_csv("clean/solTrainY.txt")
frames = [df3, df4]
result = pd.concat(frames)
result.to_csv("clean/solY.txt", index=False)

df5 = pd.read_csv("clean/solX.txt")
df6 = pd.read_csv("clean/solY.txt")
df5["y"] = df6
df5.to_csv("clean/sol.txt", index=False)

print("Pronto!")

