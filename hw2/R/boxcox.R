
install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
data(solubility)

#renomear os preditores

names(atrib)= c('1','2','3','4','5','6','7','8','9','10',
                '11','12','13','14','15','16','17','18','19','20')


# preparar para transformar as entradas nao binarias
atrib=solTrainX[ ,209:228]
summary(atrib)

#boxcox
install.packages("bestNormalize")
library("bestNormalize")
#box cox so realizar transformacao de numeros positivo. 
#Somar mais 1 em todos os atrubuidos que contém entradas negativas.

atrib$`20`=atrib$`20`+1
atrib$`19`=atrib$`19`+1
atrib$`18`=atrib$`18`+1
atrib$`17`=atrib$`17`+1
atrib$`16`=atrib$`16`+1
atrib$`15`=atrib$`15`+1
atrib$`14`=atrib$`14`+1
atrib$`13`=atrib$`13`+1
atrib$`12`=atrib$`12`+1
atrib$`10`=atrib$`10`+1
atrib$`9`=atrib$`9`+1
atrib$`8`=atrib$`8`+1
atrib$`7`=atrib$`7`+1
atrib$`6`=atrib$`6`+1

#calcular com box cox

bc1=boxcox(atrib$'1', standardize = TRUE)
bc2=boxcox(atrib$'2', standardize = TRUE)
bc3=boxcox(atrib$'3', standardize = TRUE)
bc4=boxcox(atrib$'4', standardize = TRUE)
bc5=boxcox(atrib$'5', standardize = TRUE)
bc6=boxcox(atrib$'6', standardize = TRUE)
bc7=boxcox(atrib$'7', standardize = TRUE)
bc8=boxcox(atrib$'8', standardize = TRUE)
bc9=boxcox(atrib$'9', standardize = TRUE)
bc10=boxcox(atrib$'10', standardize = TRUE)
bc11=boxcox(atrib$'11', standardize = TRUE)
bc12=boxcox(atrib$'12', standardize = TRUE)
bc13=boxcox(atrib$'13', standardize = TRUE)
bc14=boxcox(atrib$'14', standardize = TRUE)
bc15=boxcox(atrib$'15', standardize = TRUE)
bc16=boxcox(atrib$'16', standardize = TRUE)
bc17=boxcox(atrib$'17', standardize = TRUE)
bc18=boxcox(atrib$'18', standardize = TRUE)
bc19=boxcox(atrib$'19', standardize = TRUE)
bc20=boxcox(atrib$'20', standardize = TRUE)

#Apos o box cox os dados devem ter uma distribuicao mais proxima de uma normal
#a comparacao entre os dados originais e apos tranformação
#comparando a normalidade de algumas entradas
#jarque bera retorna 0 para distribuicoes normais
#
install.packages("tseries")
library(tseries)
jarque.bera.test (bc1$x.t)
jarque.bera.test (atrib$'1')
jarque.bera.test (bc2$x.t)
jarque.bera.test (atrib$'2')
jarque.bera.test (bc3$x.t)
jarque.bera.test (atrib$'3')
jarque.bera.test (bc4$x.t)
jarque.bera.test (atrib$'4')
jarque.bera.test (bc5$x.t)
jarque.bera.test (atrib$'5')
jarque.bera.test (bc6$x.t)
jarque.bera.test (atrib$'6')
jarque.bera.test (bc7$x.t)
jarque.bera.test (atrib$'7')
jarque.bera.test (bc8$x.t)
jarque.bera.test (atrib$'8')
jarque.bera.test (bc9$x.t)
jarque.bera.test (atrib$'9')
jarque.bera.test (bc10$x.t)
jarque.bera.test (atrib$'10')
jarque.bera.test (bc11$x.t)
jarque.bera.test (atrib$'11')
jarque.bera.test (bc12$x.t)
jarque.bera.test (atrib$'12')
jarque.bera.test (bc13$x.t)
jarque.bera.test (atrib$'13')
jarque.bera.test (bc14$x.t)
jarque.bera.test (atrib$'14')
jarque.bera.test (bc15$x.t)
jarque.bera.test (atrib$'15')
jarque.bera.test (bc16$x.t)
jarque.bera.test (atrib$'16')
jarque.bera.test (bc17$x.t)
jarque.bera.test (atrib$'17')
jarque.bera.test (bc18$x.t)
jarque.bera.test (atrib$'18')
jarque.bera.test (bc19$x.t)
jarque.bera.test (atrib$'19')
jarque.bera.test (bc20$x.t)
jarque.bera.test (atrib$'20')
#todos as entradas tiveram o JB menor, isso pode ser visto o histograma

par(mfrow=c(1,2))
hist(bc18$x.t,xlab = "value",main="hidrophilic boxcox")
hist(atrib$'18',xlab = "value",main="hidrophilic")
?hist()
#plotar as matrizes de correlacao

#dataframe com os dados transformados
fr_bc=data.frame(bc1$x.t,bc2$x.t,bc3$x.t,bc4$x.t,bc5$x.t,bc6$x.t,bc7$x.t,bc8$x.t,
                 bc9$x.t,bc10$x.t,bc11$x.t,bc12$x.t,bc13$x.t,bc14$x.t,bc15$x.t,bc16$x.t,
                 bc17$x.t,bc18$x.t,bc19$x.t,bc20$x.t)

#biblioteca para plotar a matriz de correlação
library(PerformanceAnalytics)
#matriz de correlação das entradas transformadas
chart.Correlation(cor(fr_bc),histogram = TRUE)

#dataframe com a saida e as entradas transformadas
yt=data.frame(fr_bc,solTrainY)
#matriz de correlação da saida com as entradas transformadas
chart.Correlation(cor(yt),histogram = TRUE)


#calcular os coeficientes da regressao

#função que retorna os coeficientes da regressão usando o box cox
#solTrainY a variavel dependente das soma das entradas idenpedentes. 
reta_bc=lm(solTrainY~bc1$x.t+bc2$x.t+bc3$x.t+bc4$x.t+bc5$x.t+bc6$x.t+bc7$x.t+bc8$x.t+
             bc9$x.t+bc10$x.t+bc11$x.t+bc12$x.t+bc13$x.t+bc14$x.t+bc15$x.t+bc16$x.t+bc17$x.t+
             bc18$x.t+bc19$x.t+bc20$x.t)
#atributo com os valores dos coeficentes
reta_bc$coefficients
View(reta_bc$coefficients)
