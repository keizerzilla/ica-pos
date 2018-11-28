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

classificador_svm = svm(formula = treino.training.Class ~ ., data = treino_pca_class, type = 'C-classification',
                        kernel = 'radial', cost = 2,scale = TRUE)

previsoes_svm = predict(classificador_svm, newdata = teste_pca_class[-67])
matriz_confusao= table(teste_pca_class[ ,67], previsoes_svm)
confusionMatrix(matriz_confusao)
#acuracia com pca utilizando 66 componetes, cost = 2 e classificador svm 0.6718