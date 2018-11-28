#regresao logistica com pca
teste=read.csv("teste_red.csv")
treino=read.csv("treino_red.csv")
# calcular pca
teste_pca<- princomp(teste[,1:252],scores=TRUE, cor=FALSE)
treino_pca<- princomp(treino[,1:252],scores=TRUE, cor=FALSE)
#cortar para o numero de componetes principais que desejo
testex=teste_pca$scores[,1:66]
treinox=treino_pca$scores[,1:66]

#juntar com Class
treino_pca_class=data.frame(treinox,treino$training.Class)
teste_pca_class=data.frame(testex,teste$testing.Class)

classificador = glm(formula = treino.training.Class ~ ., family = gaussian, data =treino_pca_class )
probabilidades = predict(classificador, type = 'response', newdata = teste_pca_class[-67])
previsoes = ifelse(probabilidades > 0.5, 1, 0)
matriz_confusao = table(teste_pca_class[, 67], previsoes)
library(caret)
confusionMatrix(matriz_confusao)
#acuracia com pca utilizando 66 componetes e classificador logit family binomial 0.6583
#acuracia com pca utilizando 66 componetes e classificador logit family gaussian 0.6525
