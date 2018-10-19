base=read.csv('homework_1_dados.csv')
base$ID= NULL
summary(base)
base[,1:9]=scale(base[,1:9])

#analisar a distribuição dos elementos
pairs(base[,1:9],col=base$CLASS)

#analisar a distribuição dos elementos por classe
#pairs(base_1)
#pairs(base_2)
#pairs(base_3)
#pairs(base_4)
#pairs(base_5)
#pairs(base_6)
#pairs(base_7)

#proximo é o código de correlação