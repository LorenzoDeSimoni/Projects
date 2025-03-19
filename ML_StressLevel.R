rm(list = ls()); graphics.off(); cat("\014")

library(clValid)
library(farff)
library(caret)
library(e1071)
library(mlrMBO)
library(randomForest)
library(EMCluster)
library(tensorflow)
library(keras3)
library(dbscan)

setwd("/Users/quad/Desktop/Machine Learning/project")

stress_data <- read.csv("StressLevelDataset.csv")

label <- paste(stress_data$mental_health_history, stress_data$stress_level, sep = ";")
stress_data$stress_level <- factor(stress_data$stress_level)
stress_data <- cbind(stress_data,label)
stress_data$label <- factor(stress_data$label)

summary(stress_data)

# funzione per calcolare la purezza ---------------------------------------

purity <- function(cluster, labels){
  if (length(cluster) != length(labels))
  {"cluster e labels hanno un numero di osservazioni diverse"}
  else
    mat <- table(cluster, labels)
  mat <- as.matrix(mat)
  k=dim(mat)[1]
  M <- numeric(k)
  for (i in 1:k) {
    M[i] = max(mat[i,])
  }
  purity = sum(M)/sum(mat)
  print(mat)
  print(purity)
}

# Variabile dipendente: stress_level --------------------------------------

# clustering gerarchico metodo di Ward ------------------------------------

# escludendo la variabile "mental_health_history"

X <- stress_data[,-c(3,21,22)]
X_scaled <- scale(X)

d <- dist(X_scaled)
h.ward <- hclust(d, method = "ward.D")
plot(h.ward)

cluster.ward <- cutree(h.ward, k = 3 )

purity(cluster.ward, stress_data$stress_level)

# EM ----------------------------------------------------------------------

set.seed(123)
emobj <- simple.init(X_scaled,nclass=3)

cluster.EM <- emcluster(X_scaled,emobj,assign.class=T)
cluster.EM

purity(cluster.EM$class, stress_data$stress_level)

# DB scan -----------------------------------------------------------------

cluster.DB <- dbscan(X_scaled, eps = 3, minPts = 5)
cluster.DB

purity(cluster.DB$cluster, stress_data$stress_level)

table(cluster.ward, cluster.EM$class)
table(cluster.ward, cluster.DB$cluster)
table(cluster.EM$class, cluster.DB$cluster)


# MBO degli iper-parametri della SVM --------------------------------------

X <- stress_data[,-c(3,22)]

par.set <- makeParamSet(
  makeDiscreteParam( "kernel", values = c( "linear", "radial") ),
  makeNumericParam( "cost", lower = -2, upper = 2,trafo = function(x) 10^x ),
  makeNumericParam( "gamma", lower = -2, upper = 2, trafo = function(x) 10^x,
                    requires = quote(kernel == "radial"))
)

ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iter = 15)
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())
tune.ctrl <- makeTuneControlMBO(mbo.control = ctrl)

task <- makeClassifTask(data = as.data.frame(X), target = "stress_level")

set.seed(1)
run <- tuneParams( makeLearner("classif.svm"), task, cv3, measures = acc,
                   par.set = par.set, control = tune.ctrl, show.info = T)

y <- getOptPathY(run$opt.path)
plot(y, type = "o", lwd = 3, col = "green")
lines(cummax(y), pch = 19, lwd = 3, col = "blue")

err <- 1-y
plot(cummin(y), type = "o", pch = 19, lwd = 3, col = "blue")

# SMV con 10 fold CV ------------------------------------------------------

ixs <- createFolds(y=stress_data$stress_level,k=10,list=T)
trnAcc <- valAcc <- numeric(length(ixs))


for( k in 1:length(ixs) ) {
  trnFold <- X[-ixs[[k]],]
  valFold <- X[ixs[[k]],]
  
  # training the model
  model <- svm( formula=stress_level ~., data=trnFold, scale=T,
                type="C-classification", cost=run$x$cost, kernel=run$x$kernel) 
  cat("* Confusion matrix on the training fold:\n")
  print( table(trnFold$stress_level,model$fitted) )
  trnAcc[k] <- mean( trnFold$stress_level == model$fitted )
  
  # validating the model
  preds <- predict( model, newdata=valFold[,-20] )
  cat("+ Confusion matrix on the validation fold:\n")
  print( table(valFold$stress_level,preds) )
  valAcc[k] <- mean( valFold$stress_level == preds )
}

plot( trnAcc, type="o", pch=19, lwd=3, col="blue", ylab="Accuracy", xlab="fold",
      ylim=range(c(1,trnAcc,valAcc)))
lines( valAcc, type="o", pch=19, lwd=3, col="green3")
abline(h=1,col="red",lty=2,lwd=2)
legend( "topright", legend=c("training","validation"), col=c("blue","green3"),
        lwd=3, pch=19, cex=1.3 )

model_full <- svm( formula=stress_level ~., data=X, scale=T,
                   type="C-classification", cost=10, kernel="linear" )
fullDsAcc <- mean( X$stress_level == model_full$fitted )

emp_error <- 1-fullDsAcc

gen_error <- mean(1-valAcc)

emp_error; gen_error

table(model_full$fitted, X[,20])


# RANDOM FOREST per STRESS LEVEL ------------------------------------------

label <- stress_data$stress_level

X_scaled <- as.data.frame(X_scaled)
X_scaled$label <- factor(label)
X_scaled <- as.data.frame(X_scaled)

ixs <- createFolds(y=X_scaled$label,k=10,list=T)
trnAcc <- valAcc <- numeric(length(ixs))
head(X_scaled)
for( k in 1:length(ixs) ) {
  trnFold <- X_scaled[-ixs[[k]],]
  valFold <- X_scaled[ixs[[k]],]
  
  # training the model
  model <- randomForest( label ~., data=trnFold ) 
  cat("* Confusion matrix on the training fold:\n")
  print( table(trnFold$label,model$predicted) )
  trnAcc[k] <- mean( trnFold$label == model$predicted )
  
  # validating the model
  preds <- predict( model, newdata=valFold[,-20] )
  cat("+ Confusion matrix on the validation fold:\n")
  print( table(valFold$label,preds) )
  valAcc[k] <- mean( valFold$label == preds )
}

plot( trnAcc, type="o", pch=19, lwd=3, col="blue", ylab="Accuracy", xlab="fold",
      ylim=range(c(1,trnAcc,valAcc)))
lines( valAcc, type="o", pch=19, lwd=3, col="green3")
abline(h=1,col="red",lty=2,lwd=2)
legend( "topright", legend=c("training","validation"), col=c("blue","green3"),
        lwd=3, pch=19, cex=1.3 )

model_full <- randomForest( label ~., data=X_scaled, ntree = 1000)

fullDsAcc <- mean( X_scaled$label == model_full$predicted)

emp_error <- 1-fullDsAcc

gen_error <- mean(1-valAcc)

emp_error; gen_error

table(model_full$predicted, X_scaled$label)
table(X_scaled$label)


purity(model_full$predicted, X_scaled$label)


# Neural Network per Stress Level -----------------------------------------

X_scaled <- X
X_scaled[,-20] <- scale(X[,-20])
X_scaled <- as.data.frame(X_scaled)

one_hot_encode <- function(labels) {
  n_labels <- length(labels)
  n_classes <- length(unique(labels))
  one_hot_matrix <- matrix(0, nrow = n_labels, ncol = n_classes)
  for (i in 1:n_labels) {
    one_hot_matrix[i, as.integer(labels[i])] <- 1
  }
  return(one_hot_matrix)
}

labels_one_hot <- one_hot_encode(as.integer(X_scaled$stress_level))

set.seed(132)
idx <- sample(1:dim(X_scaled)[1], 0.7 * dim(X_scaled)[1], replace = FALSE)
head(X_scaled)
train_cov <- as.matrix(X_scaled[idx, 1:19])
train_lab <- labels_one_hot[idx, ]

test_cov <- as.matrix(X_scaled[-idx, 1:19])
test_lab <- labels_one_hot[-idx, ]

network <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = "relu", input_shape = c(ncol(train_cov))) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 3, activation = "linear") %>%
  layer_dense(units = ncol(labels_one_hot), activation = "softmax")

network %>% compile(
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- network %>% fit(
  train_cov, train_lab, 
  epochs = 100, 
  batch_size = 64, 
  validation_split = 0.3,
  callbacks = list(callback_early_stopping(patience = 20))
)

metrics <- network %>% evaluate(test_cov, test_lab)
metrics

# variabile dipendente: label ---------------------------------------------
label <- paste(stress_data$mental_health_history, stress_data$stress_level, sep = ";")
stress_data$stress_level <- factor(stress_data$stress_level)
stress_data <- cbind(stress_data,label)

table(stress_data$label)

# clustering gerarchico metodo di Ward ------------------------------------
X <- stress_data[,-c(3,21,22)]
X_scaled <- scale(X)

cluster.ward.6k <- cutree(h.ward, k = 6)

table(cluster.ward.6k)

purity(cluster.ward.6k, stress_data$label)

# DB scan -----------------------------------------------------------------

cluster.DB <- dbscan(X_scaled, eps = 2, minPts = 5)
cluster.DB

purity(cluster.DB$cluster, stress_data$label)

# MBO degli iper-parametri della SVM --------------------------------------

X <- stress_data[,-c(3,21)]

par.set <- makeParamSet(
  makeDiscreteParam( "kernel", values = c( "linear", "radial")),
  makeNumericParam( "cost", lower = -2, upper = 2,trafo = function(x) 10^x ),
  makeNumericParam( "gamma", lower = -2, upper = 2, trafo = function(x) 10^x,
                    requires = quote(kernel == "radial"))
)

ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iter = 15)
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())
tune.ctrl <- makeTuneControlMBO(mbo.control = ctrl)

task <- makeClassifTask(data = X, target = "label")

set.seed(1)
run <- tuneParams( makeLearner("classif.svm"), task, cv3, measures = acc,
                   par.set = par.set, control = tune.ctrl, show.info = T)

y <- getOptPathY(run$opt.path)
plot(y, type = "o", lwd = 3, col = "green")
lines(cummax(y), pch = 19, lwd = 3, col = "blue")

err <- 1-y
plot(cummin(y), type = "o", pch = 19, lwd = 3, col = "blue")

# SMV con 10 fold CV ------------------------------------------------------

ixs <- createFolds(y=stress_data$stress_level,k=10,list=T)
trnAcc <- valAcc <- numeric(length(ixs))

for( k in 1:length(ixs) ) {
  trnFold <- X[-ixs[[k]],]
  valFold <- X[ixs[[k]],]
  
  # training the model
  model <- svm( formula=label ~., data=trnFold, scale=T,
                type="C-classification", cost=run$x$cost, kernel=run$x$kernel) 
  cat("* Confusion matrix on the training fold:\n")
  print( table(trnFold$label,model$fitted) )
  trnAcc[k] <- mean( trnFold$label == model$fitted )
  
  # validating the model
  preds <- predict( model, newdata=valFold[,-20] )
  cat("+ Confusion matrix on the validation fold:\n")
  print( table(valFold$label,preds) )
  valAcc[k] <- mean( valFold$label == preds )
}

plot( trnAcc, type="o", pch=19, lwd=3, col="blue", ylab="Accuracy", xlab="fold",
      ylim=range(c(1,trnAcc,valAcc)))
lines( valAcc, type="o", pch=19, lwd=3, col="green3")
abline(h=1,col="red",lty=2,lwd=2)
legend( "topright", legend=c("training","validation"), col=c("blue","green3"),
        lwd=3, pch=19, cex=1.3 )

model_full <- svm( formula=label ~., data=X, scale=T,
                   type="C-classification", cost=10, kernel="linear" )
fullDsAcc <- mean( X$label == model_full$fitted )

emp_error <- 1-fullDsAcc

gen_error <- mean(1-valAcc)

emp_error; gen_error

table(model_full$fitted, X$label)


# NN per Variabile Label --------------------------------------------------

X <- stress_data[,-c(3,21)]
X_scaled <- X
X_scaled[,-20] <- scale(X[,-20])

# Creare una funzione di codifica one-hot personalizzata
one_hot_encode <- function(labels) {
  n_labels <- length(labels)
  n_classes <- length(unique(labels))
  one_hot_matrix <- matrix(0, nrow = n_labels, ncol = n_classes)
  for (i in 1:n_labels) {
    one_hot_matrix[i, as.integer(labels[i])] <- 1
  }
  return(one_hot_matrix)
}
labels_one_hot <- one_hot_encode(as.integer(X_scaled$label))


set.seed(123)

idx <- sample(1:dim(X_scaled)[1], 0.7 * dim(X_scaled)[1], replace = FALSE)

train_cov <- as.matrix(X_scaled[idx,  1:19])
train_lab <- labels_one_hot[idx, ]

test_cov <- as.matrix(X_scaled[-idx, 1:19])
test_lab <- labels_one_hot[-idx, ]

network <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = "relu", input_shape = c(ncol(train_cov))) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 6, activation = "linear") %>%
  layer_dense(units = ncol(labels_one_hot), activation = "softmax")

network %>% compile(
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
history <- network %>% fit(
  train_cov, train_lab, 
  epochs = 100, 
  batch_size = 64, 
  validation_split = 0.3,
  callbacks = list(callback_early_stopping(patience = 20))
)

metrics <- network %>% evaluate(test_cov, test_lab)
metrics

