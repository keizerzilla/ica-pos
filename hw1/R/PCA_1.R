base=read.csv('homework_1_dados.csv')
base$ID= NULL
summary(base)
base[,1:9]=scale(base[,1:9])


#aplicando PCA
base.pcal<- princomp(base[,1:9],scores=TRUE, cor=TRUE)
#cor:Se TRUE, os dados ser�o centralizados e redimensionados antes da an�lise
#scores:Se TRUE, as coordenadas em cada componente principal s�o calculadas


#observar desvio padr�o e a propo��o de varia��o de cada componente
summary(base.pcal)

plot(base.pcal,main = "Proportion of Variance",xlab="Componentes")

#a matriz de loadings vari�veis
base.pcal$loadings

#preencher todos os valores da matriz loadings, mesmo os pequenos
load=with(base.pcal, unclass(loadings))
#contribui��o proporcional de cada preditor para cada componente principal
aload=abs(load)
sweep(aload, 2, colSums(aload), "/")

