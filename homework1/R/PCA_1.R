base=read.csv('homework_1_dados.csv')
base$ID= NULL
summary(base)
base[,1:9]=scale(base[,1:9])


#aplicando PCA
base.pcal<- princomp(base[,1:9],scores=TRUE, cor=TRUE)
#cor:Se TRUE, os dados serão centralizados e redimensionados antes da análise
#scores:Se TRUE, as coordenadas em cada componente principal são calculadas


#observar desvio padrão e a propoção de variação de cada componente
summary(base.pcal)

plot(base.pcal,main = "Proportion of Variance",xlab="Componentes")

#a matriz de loadings variáveis
base.pcal$loadings

#preencher todos os valores da matriz loadings, mesmo os pequenos
load=with(base.pcal, unclass(loadings))
#contribuição proporcional de cada preditor para cada componente principal
aload=abs(load)
sweep(aload, 2, colSums(aload), "/")

