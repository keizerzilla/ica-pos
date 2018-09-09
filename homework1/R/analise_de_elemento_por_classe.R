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
#separando por  elementos de classe 1
base_1_ri=base_1$RI
base_1_na=base_1$NA.
base_1_mg=base_1$MG
base_1_al=base_1$AL
base_1_si=base_1$SI
base_1_k=base_1$K
base_1_ca=base_1$CA
base_1_ba=base_1$BA
base_1_fe=base_1$FE
#ver quais elementos tem menor variacao
summary(base_1)
#histograma do elementos com meno variacao
hist(base_1_fe)
hist(base_1_ba)
#separando por  elementos de classe 2
base_2_ri=base_2$RI
base_2_na=base_2$NA.
base_2_mg=base_2$MG
base_2_al=base_2$AL
base_2_si=base_2$SI
base_2_k=base_2$K
base_2_ca=base_2$CA
base_2_ba=base_2$BA
base_2_fe=base_2$FE
#ver quais elementos tem menor variacao
summary(base_2)
#histograma do elementos com menor variacao
hist(base_2_fe)
hist(base_2_ba)
#separando por  elementos de classe 3
base_3_ri=base_3$RI
base_3_na=base_3$NA.
base_3_mg=base_3$MG
base_3_al=base_3$AL
base_3_si=base_3$SI
base_3_k=base_3$K
base_3_ca=base_3$CA
base_3_ba=base_3$BA
base_3_fe=base_3$FE
#ver quais elementos tem menor varia??o
summary(base_3)
#separando por elementos de classe 4
base_4_ri=base_4$RI
base_4_na=base_4$NA.
base_4_mg=base_4$MG
base_4_al=base_4$AL
base_4_si=base_4$SI
base_4_k=base_4$K
base_4_ca=base_4$CA
base_4_ba=base_4$BA
base_4_fe=base_4$FE
#ver quais elementos tem menor varia??o
summary(base_4)
#separando por elementos de classe 5
base_5_ri=base_5$RI
base_5_na=base_5$NA.
base_5_mg=base_5$MG
base_5_al=base_5$AL
base_5_si=base_5$SI
base_5_k=base_5$K
base_5_ca=base_5$CA
base_5_ba=base_5$BA
base_5_fe=base_5$FE
#ver quais elementos tem menor varia??o
summary(base_5)
#separando por elementos de classe 6
base_6_ri=base_6$RI
base_6_na=base_6$NA.
base_6_mg=base_6$MG
base_6_al=base_6$AL
base_6_si=base_6$SI
base_6_k=base_6$K
base_6_ca=base_6$CA
base_6_ba=base_6$BA
base_6_fe=base_6$FE
#ver quais elementos tem menor varia??o
summary(base_6)
#separando por elementos de classe 7
base_7_ri=base_7$RI
base_7_na=base_7$NA.
base_7_mg=base_7$MG
base_7_al=base_7$AL
base_7_si=base_7$SI
base_7_k=base_7$K
base_7_ca=base_7$CA
base_7_ba=base_7$BA
base_7_fe=base_7$FE
#ver quais elementos tem menor variancao
summary(base_7)


#comparar cada elemnto por diferentes classes
summary(base_1_ba)
summary(base_2_ba)
summary(base_3_ba)
#summary(base_4_ba)
summary(base_5_ba)
summary(base_6_ba)
summary(base_7_ba)

summary(base_1_fe)
summary(base_2_fe)
summary(base_3_fe)
#summary(base_4_fe)
summary(base_5_fe)
summary(base_6_fe)
summary(base_7_fe)

summary(base_1_al)
summary(base_2_al)
summary(base_3_al)
#summary(base_4_al)
summary(base_5_al)
summary(base_6_al)
summary(base_7_al)

summary(base_1_ca)
summary(base_2_ca)
summary(base_3_ca)
#summary(base_4_ca)
summary(base_5_ca)
summary(base_6_ca)
summary(base_7_ca)

summary(base_1_k)
summary(base_2_k)
summary(base_3_k)
#summary(base_4_k)
summary(base_5_k)
summary(base_6_k)
summary(base_7_k)

summary(base_1_mg)
summary(base_2_mg)
summary(base_3_mg)
summary(base_5_mg)
summary(base_6_mg)
summary(base_7_mg)

summary(base_1_na)
summary(base_2_na)
summary(base_3_na)
summary(base_5_na)
summary(base_6_na)
summary(base_7_na)

summary(base_1_si)
summary(base_2_si)
summary(base_3_si)
summary(base_5_si)
summary(base_6_si)
summary(base_7_si)

summary(base_1_ri)
summary(base_2_ri)
summary(base_3_ri)
summary(base_5_ri)
summary(base_6_ri)
summary(base_7_ri)
#fim da comparação de cada elemnto por diferentes classes