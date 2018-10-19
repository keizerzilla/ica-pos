base=read.csv('homework_1_dados.csv')
base$ID= NULL
summary(base)
base[,1:9]=scale(base[,1:9])


hist(base$MG,xlab = "Mg",main="histograma Mg")
hist(base$K,xlab = "K",main="histograma K")