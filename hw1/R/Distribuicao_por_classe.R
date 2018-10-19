base=read.csv('homework_1_dados.csv')
base$ID= NULL
summary(base)
base[,1:9]=scale(base[,1:9])

length(base_1$CLASS)
length(base_2$CLASS)
length(base_3$CLASS)
length(base_4$CLASS)
length(base_5$CLASS)
length(base_6$CLASS)
length(base_7$CLASS)

distribuicao=c(length(base_1$CLASS),length(base_2$CLASS),length(base_3$CLASS),
               length(base_4$CLASS),length(base_5$CLASS),length(base_6$CLASS),
               length(base_7$CLASS))
plot(distribuicao,ylab = "Número de intâncias",xlab = "Classe",
     main = "Distribuição por classe",type = "b")