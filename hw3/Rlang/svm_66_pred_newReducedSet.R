#codigo para rodar com os preditores que apresentam variancia acima de 0.06
#treino_C e teste_C são as variaveis com os 66 preditores e a etiqueta Class

teste_C=read.csv("teste_newRed.csv")
treino_C=read.csv("treino_newRed.csv")

library(e1071)
classificador_svm = svm(formula = treino.training.Class ~ ., data = treino_C, type = 'C-classification',
                        kernel = 'radial', cost = 1,scale = TRUE)

previsoes_svm = predict(classificador_svm, newdata = teste_C[-67])
matriz_confusao= table(teste_C[ ,67], previsoes_svm)

library(caret)
confusionMatrix(matriz_confusao)

# Acuracia 0.8668 com kernel = radial e cost= 2 e 66 preditores
#Acuracia 0.861 com kernel = radial e cost= 1 e 66 preditores
