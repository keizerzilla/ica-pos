base=read.csv('homework_1_dados.csv')
base$ID= NULL
summary(base)
base[,1:9]=scale(base[,1:9])

#calculando a covariaca dos elemntos com ri 1
cor(base$RI,base$NA.)
cor(base$RI,base$MG)
cor(base$RI,base$AL)
cor(base$RI,base$K)
cor(base$RI,base$BA)
cor(base$RI,base$FE)
cor(base$RI,base$CA)
cor(base$RI,base$SI)
#calculando a covariaca dos elemntos com Na

cor(base$NA.,base$MG)
cor(base$NA.,base$AL)
cor(base$NA.,base$K)
cor(base$NA.,base$BA)
cor(base$NA.,base$FE)
cor(base$NA.,base$CA)
cor(base$NA.,base$SI)

#calculando a covariaca dos elemntos com Mg

cor(base$MG,base$AL)
cor(base$MG,base$K)
cor(base$MG,base$BA)
cor(base$MG,base$FE)
cor(base$MG,base$CA)
cor(base$MG,base$SI)

#calculando a covariaca dos elemntos com Al
cor(base$AL,base$K)
cor(base$AL,base$BA)
cor(base$AL,base$FE)
cor(base$AL,base$CA)
cor(base$AL,base$SI)

#calculando a covariaca dos elemntos com k
cor(base$K,base$BA)
cor(base$K,base$FE)
cor(base$K,base$CA)
cor(base$K,base$SI)

#calculando a covariaca dos elemntos com Ba
cor(base$BA,base$FE)
cor(base$BA,base$CA)
cor(base$BA,base$SI)

#calculando a covariaca dos elemntos com Fe
cor(base$FE,base$CA)
cor(base$FE,base$SI)

#calculando a covariaca dos elemntos com Ca
cor(base$CA,base$SI)




cor(base$RI,base$CA)
plot(base$RI,base$CA,ylab = "Ca",xlab = "RI",main="Maior correlação positiva, 0.8104",col=base$CLASS)

cor(base$RI,base$SI)
plot(base$RI,base$SI,ylab = "Si",xlab = "RI",main="Maior correlação negativa, -0.5420",col=base$CLASS)

cor(base$K,base$MG)
plot(base$K,base$MG,ylab = "Mg",xlab = "K",main="Baixa correlação, -0.0054",col=base$CLASS)