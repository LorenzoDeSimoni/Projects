library(corrplot)


red = read.csv('Red.csv', header = TRUE, sep = ',')
dim(red)
head(red)
white = read.csv('White.csv', header = TRUE, sep = ',')
dim(white)
head(red)
head(white)
pairs(X)
X = as.matrix(red[,-1])
head(X)
Y = as.matrix(white[,-1])
par(mfrow = c(1,2))
boxplot(red$quality,main = 'Red Wine Quality', ylim= c(3,9))
boxplot(white$quality,main = 'White Wine Quality' )

colMeans(X)
colMeans(Y)
#analisi esplorativa
plot(red$quality, red$alcohol)
pairs(Y)
head(Y)
#analisi correlazione variabili 
r = cor(X)
s = cor(Y)
corrplot(r, type = 'upper', method = 'ellipse') #forte correlazione positiva tra 
corrplot(r, type = 'upper', method = 'ellipse') #forte correlazione positiva tra 

#scelta del numero dei fattori 
fa5 = factanal(Y, factors = 5)
fa6 = factanal(Y, factors = 5, rotation = 'promax')
fa5$loadings
fa5$uniquenesses
fa6$loadings
fa4 = factanal(X, factors = 5, rotation = 'none')
xstd = scale(X)
fa3 = factanal(X, factors = 5)
fa3$loadings
fa4$loadings
#componenti principali 
pca = prcomp(X)
pca2 = prcomp(Y)
screeplot(pca2, type = 'l')
summary(pca)
summary(pca2)
pca
plot(pca,type = 'l', main = 'Scree-plot vino rosso')
summary(pca2)
plot(pca2,type = 'l', main = 'Scree-plot vino bianco')

fa5 = factanal(Y,factors = 5)
fa5$loadings

biplot(pca)
