#antes de rodar esse aruivo deve ter sido f=gerado o arquivo que gera o teste_red.csv
# e o treino_red.csv
teste=read.csv("teste_red.csv")
treino=read.csv("treino_red.csv")

newReducedSet=nearZeroVar(treino,names = TRUE)

reduzir=setdiff(reducedSet,newReducedSet)

treino_min=treino[,reduzir]
teste_min=teste[,reduzir]
treino_C=data.frame(treino_min,treino$training.Class)
teste_C=data.frame(teste_min,teste$testing.Class)

#gerar arquivos com colunas iguais ao reducedSet
write.csv(teste_C, "teste_newRed.csv", row.names = FALSE)
write.csv(treino_C, "treino_newRed.csv", row.names = FALSE)