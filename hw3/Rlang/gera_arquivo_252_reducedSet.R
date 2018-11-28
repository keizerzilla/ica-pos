# criando e salvando os df como treino_red.csv e treino_red.csv

# reduzir dimensionalidade 
#reduzindo a dimensão do treino para colunas iguais ao reducedSet
training_reduce=training[,reducedSet]
#adicionando a coluna Classe, pois o reducedSet não contém ela
treino1=data.frame(training_reduce,training$Class)

#treino[,1:252]=scale(treino[,1:252])

#reduzindo a dimensão do teste para colunas iguais ao reducedSet
testing_reduce=testing[,reducedSet]
#adicionando a coluna Classe, pois o reducedSet não contém ela
teste1=data.frame(testing_reduce,testing$Class)
#teste[,1:252]=scale(teste[,1:252])

#substitir por 0 e 1 os valores da coluna Class
treino1$training.Class = factor(treino1$training.Class, levels = c('unsuccessful', 'successful'), labels = c(0, 1))
teste1$testing.Class = factor(teste1$testing.Class, levels = c('unsuccessful', 'successful'), labels = c(0, 1))
#gerar arquivos com colunas iguais ao reducedSet
write.csv(teste1, "teste_red.csv", row.names = FALSE)
write.csv(treino1, "treino_red.csv", row.names = FALSE)