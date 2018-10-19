library("e1071")
base=read.csv('homework_1_dados.csv')
base$ID= NULL
summary(base)
base[,1:9]=scale(base[,1:9])

#skewness(base$RI)
#skewness(base$NA.)
#skewness(base$MG)
#skewness(base$AL)
#skewness(base$SI)
#skewness(base$K)
#skewness(base$CA)
#skewness(base$BA)
#skewness(base$FE)

skew=c(skewness(base$RI),skewness(base$NA.),skewness(base$MG),
       skewness(base$AL),skewness(base$SI),skewness(base$K),
       skewness(base$CA),skewness(base$BA),skewness(base$FE))

plot(skew,ylab = "valor das skewness",main = "Skewness",
     xlab = "Ri=1  Na=2  Mg=3  Al=4  Si=5  K=6  Ca=7  Ba=8  Fe=9",type = "b")
