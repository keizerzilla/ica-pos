install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
data(solubility)

#renomear os preditores

names(atrib)= c('1','2','3','4','5','6','7','8','9','10',
                '11','12','13','14','15','16','17','18','19','20')


# preparar para transformar as entradas nao binarias
atrib=solTrainX[ ,209:228]
summary(atrib)

#yeojohnson
install.packages("bestNormalize")
library("bestNormalize")


#calcular com yeojohnson
?library(bestNormalize)
?yeojohnson()
yj1=yeojohnson(atrib$'1', eps = 0.001, standardize = TRUE)
yj2=yeojohnson(atrib$'2', eps = 0.001, standardize = TRUE)
yj3=yeojohnson(atrib$'3', eps = 0.001, standardize = TRUE)
yj4=yeojohnson(atrib$'4', eps = 0.001, standardize = TRUE)
yj5=yeojohnson(atrib$'5', eps = 0.001, standardize = TRUE)
yj6=yeojohnson(atrib$'6', eps = 0.001, standardize = TRUE)
yj7=yeojohnson(atrib$'7', eps = 0.001, standardize = TRUE)
yj8=yeojohnson(atrib$'8', eps = 0.001, standardize = TRUE)
yj9=yeojohnson(atrib$'9', eps = 0.001, standardize = TRUE)
yj10=yeojohnson(atrib$'10', eps = 0.001, standardize = TRUE)
yj11=yeojohnson(atrib$'11', eps = 0.001, standardize = TRUE)
yj12=yeojohnson(atrib$'12', eps = 0.001, standardize = TRUE)
yj13=yeojohnson(atrib$'13', eps = 0.001, standardize = TRUE)
yj14=yeojohnson(atrib$'14', eps = 0.001, standardize = TRUE)
yj15=yeojohnson(atrib$'15', eps = 0.001, standardize = TRUE)
yj16=yeojohnson(atrib$'16', eps = 0.001, standardize = TRUE)
yj17=yeojohnson(atrib$'17', eps = 0.001, standardize = TRUE)
yj18=yeojohnson(atrib$'18', eps = 0.001, standardize = TRUE)
yj19=yeojohnson(atrib$'19', eps = 0.001, standardize = TRUE)
yj20=yeojohnson(atrib$'20', eps = 0.001, standardize = TRUE)


#Apos o yeo johnson os dados devem ter uma distribuicao mais proxima de uma normal
#jarque bera(JB) retorna 0 para distribicoes normais
#Em muitos casos o valor com boxcox e yeojohnson são proximos
#No caso do atribudo 18, boxcox realizar um distribuição mais proxima a uma normal
install.packages("tseries")
library(tseries)
jarque.bera.test (yj1$x.t)
jarque.bera.test (yj2$x.t)
jarque.bera.test (yj3$x.t)
jarque.bera.test (yj4$x.t)
jarque.bera.test (yj5$x.t)
jarque.bera.test (yj6$x.t)
jarque.bera.test (yj7$x.t)
jarque.bera.test (yj8$x.t)
jarque.bera.test (yj9$x.t)
jarque.bera.test (yj10$x.t)
jarque.bera.test (yj11$x.t)
jarque.bera.test (yj12$x.t)
jarque.bera.test (yj13$x.t)
jarque.bera.test (yj14$x.t)
jarque.bera.test (yj15$x.t)
jarque.bera.test (yj16$x.t)
jarque.bera.test (yj17$x.t)
jarque.bera.test (yj18$x.t)
jarque.bera.test (yj19$x.t)
jarque.bera.test (yj20$x.t)

#No caso do atribudo 18(hidrophilicFactor), boxcox realizar um distribuição mais proxima a uma normal
#comparar o o JB do atributo hidrophilicFactor
jarque.bera.test (yj18$x.t)$statistic #igual a 52.75
jarque.bera.test (bc18$x.t)$statistic #igual 3.19


#plotr o histograma pos yeojohnson

par(mfrow=c(4,5))
hist(yj1$x.t,main = "mol Weight")
hist(yj2$x.t,main ="NumAtoms")
hist(yj3$x.t,main = "NumNonAtoms")
hist(yj4$x.t,main = "NumBonds")
hist(yj5$x.t,main = "NumNonBonds")
hist(yj6$x.t,main = "NumMultBonds")
hist(yj7$x.t,main = "NumRotBonds")
hist(yj8$x.t,main = "NumDBLBonds")
hist(yj9$x.t,main = "NumAromaticBonds")
hist(yj10$x.t,main = "NumHydrogen")
hist(yj11$x.t,main = "NumCarbon")
hist(yj12$x.t,main = "NumNitrogen")
hist(yj13$x.t,main = "NumOxygen")
hist(yj14$x.t,main = "NumSulfer")
hist(yj15$x.t,main = "NumChorine")
hist(yj16$x.t,main = "NumHalogen")
hist(yj17$x.t,main = "NumRings")
hist(yj18$x.t,main = "HydrophilicFactor")
hist(yj19$x.t,main = "SurfaceArea1")
hist(yj20$x.t,main = "SurfaceArea2")




#plotar as matrizes de correlacao

#dataframe com o sdados transformados
fr_yj=data.frame(yj1$x.t,yj2$x.t,yj3$x.t,yj4$x.t,yj5$x.t,yj6$x.t,yj7$x.t,yj8$x.t,
                 yj9$x.t,yj10$x.t,yj11$x.t,yj12$x.t,yj13$x.t,yj14$x.t,yj15$x.t,yj16$x.t,
                 yj17$x.t,yj18$x.t,yj19$x.t,yj20$x.t)

#biblioteca para plotar a matriz de correlação
library(PerformanceAnalytics)
#matriz de correlação das entradas transformadas
chart.Correlation(cor(fr_yj),histogram = TRUE)

#dataframe com a saida e as entradas transformadas
yt=data.frame(fr_yj,solTrainY)
#matriz de correlação da saida com as entradas transformadas
chart.Correlation(cor(yt),histogram = TRUE)

#calcular os coeficientes da regressao

#função que retorna os coeficientes da regressão usando o yeo johnson
#solTrainY a variavel dependente das soma das entradas idenpedentes. 
reta_yj=lm(solTrainY~yj1$x.t+yj2$x.t+yj3$x.t+yj4$x.t+yj5$x.t+yj6$x.t+yj7$x.t+yj8$x.t+
             yj9$x.t+yj10$x.t+yj11$x.t+yj12$x.t+yj13$x.t+yj14$x.t+yj15$x.t+yj16$x.t+yj17$x.t+
             yj18$x.t+yj19$x.t+yj20$x.t)
#atributo com os valores dos coeficentes 
reta_yj$coefficients

View(reta_yj$coefficients)

#comparando coxbox com yeojohson
summary(fr_bc)
summary(fr_yj)
