base=read.csv('homework_1_dados.csv')
base$ID= NULL
summary(base)
base[,1:9]=scale(base[,1:9])
#separando por classe
base_1= base[base$CLASS==1,]
base_2= base[base$CLASS==2,]
base_3= base[base$CLASS==3,]
base_4= base[base$CLASS==4,]
base_5= base[base$CLASS==5,]
base_6= base[base$CLASS==6,]
base_7= base[base$CLASS==7,]
#analisar as classes
summary(base_1)
summary(base_2)
summary(base_3)
summary(base_4)
summary(base_5)
summary(base_6)
summary(base_7)
