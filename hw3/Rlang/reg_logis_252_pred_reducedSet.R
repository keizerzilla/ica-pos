#codigo do classificicador de regressão linear

teste=read.csv("teste_red.csv")
treino=read.csv("treino_red.csv")

#comando para instalar o pacote
#install.packages('e1071')
#comando para usar a biblioteca
#library(e1071)

classificador_rl = glm(formula = training.Class ~ ., family = gaussian, data = treino)
probabilidades = predict(classificador_rl, type = 'response', newdata =teste[-253])
previsoes_rl = ifelse(probabilidades > 0.5, 1, 0)
matriz_confusao_rl= table(teste[, 253], previsoes_rl)

#comando para usar a biblioteca
#library(caret)
confusionMatrix(matriz_confusao_rl)

#Acuracia com 0,8398 familia binomial com 252 preditores
#Acuracia com 0,8494 familia gaussian com 252 preditores