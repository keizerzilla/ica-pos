#codigo do classificicador de regressão linear
#treino_C e teste_C são as variaveis com os 66 preditores e a etiqueta Class
teste_C=read.csv("teste_newRed.csv")
treino_C=read.csv("treino_newRed.csv")

#comando para instalar o pacote
#install.packages('e1071')
#comando para usar a biblioteca
#library(e1071)
#help("glm")
classificador_rl = glm(formula = treino.training.Class ~ ., family = gaussian, data = treino_C)
probabilidades = predict(classificador_rl, type = 'response', newdata =teste_C[-67])
previsoes_rl = ifelse(probabilidades > 0.5, 1, 0)
matriz_confusao_rl= table(teste_C[, 67], previsoes_rl)

#comando para usar a biblioteca
#library(caret)
confusionMatrix(matriz_confusao_rl)
#Acuracia com 0,8398 familia binomial com 66 preditores
#Acuracia com 0,8475 familia gaussian  com 66 preditores