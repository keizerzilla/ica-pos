#codigo do classificicador SVM
teste=read.csv("teste_red.csv")
treino=read.csv("treino_red.csv")

#comando para instalar o pacote
#install.packages('e1071')

#comando para usar a biblioteca
#library(e1071)

#help(svm)

classificador_svm = svm(formula = training.Class ~ ., data = treino, type = 'C-classification',
                        kernel = 'radial', cost = 2,scale = TRUE)

previsoes_svm = predict(classificador_svm, newdata = teste[-253])
matriz_confusao_svm= table(teste[ ,253], previsoes_svm)
#length(previsoes_svm)

#comando para usar a biblioteca
#library(caret)

confusionMatrix(matriz_confusao_svm)

# Acuracia 0.8127 com kernel = radial e cost= 0.8 e 1882 preditores
# Acuracia 0.8108 com kernel = radial e cost= 0.5 e 1882 preditores
# Acuracia 0.8514 com kernel = radial e cost= 0.8 e 252 preditores
# Acuracia 0.8533 com kernel = radial e cost= 1 e 252 preditores
# Acuracia 0.8533 com kernel = radial e cost= 2 e 252 preditores
# Acuracia 0.7703 com kernel = sigmoid e cost= 2 e 252 preditores
