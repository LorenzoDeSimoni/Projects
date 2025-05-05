setwd("C:/Users/vanes/OneDrive - Università degli Studi di Milano-Bicocca/CLAMSES/SECONDO ANNO/Modelli Baysiani/Progetto")

# -------------------------------------------------------------------------
# dati
# -------------------------------------------------------------------------

df <- read.csv("JPM_var_economiche.csv")

library(bsts)
library(lubridate)
library(dplyr)
library(tseries)
library(car)

df$Date <- as.Date(df$Date)

# rendimenti logaritmici mensili --> ritorno logaritmico = variazione percentuale continua tra due mesi consecutivi
df <- df %>%
  arrange(Date) %>%
  mutate(
    JPM.Returns = c(NA, diff(log(JPM.Close))),
    CPI = c(NA, diff(CPI)),
    salary = c(NA, diff(salary)),
    Consumer_credit = c(NA, diff(Consumer_credit)),
    GEPUCURRENT = c(NA, diff(log(GEPUCURRENT))),
    VIX = scale(VIX),                   
    UNRATE = scale(UNRATE),             
    DFF = scale(DFF),
    KBW = scale(KBW)) %>%
  filter(complete.cases(.))

# costruzione di y e xreg 
y <- ts(df$JPM.Returns, start = c(2006,5), frequency = 12)

xreg <- df %>%
  select(-Date,-JPM.Close,-JPM.Volume,-JPM.Returns)

# correlazione tra regressori (possibile multicollinearità)
cor_x <- cor(xreg)
round(cor_x,2)
library(corrplot)
corrplot(cor_x, method = "color", type = "upper", tl.cex = 0.8)
# controllo VIF (Variable Inflation Factor)
vif(lm(y~., data = as.data.frame(xreg)))
# più il VIF è alto, più la variabile è lineare combinazione delle altre (collinearità)

stepwise_vif_filter <- function(y, xreg, threshold = 5){
  xreg_df <- as.data.frame(xreg)
  repeat{
    model <- lm(y~.,data=xreg_df)
    vif_values <- vif(model)
    
    max_vif <- max(vif_values)
    
    if(max_vif < threshold){
      break
    }
    
    variable_to_remove <- names(which.max(vif_values))
    cat("Rimuovo:",variable_to_remove,"con VIF = ",round(max_vif,2),"\n")
    
    xreg_df <- xreg_df %>% select(-all_of(variable_to_remove))
  }
  return(xreg_df)
}

xreg_clean <- stepwise_vif_filter(y,xreg,threshold = 5)
xreg <- xreg_clean

# -------------------------------------------------------------------------
# training e test set
# -------------------------------------------------------------------------


train_index <- length(y)-12 
y_train <- y[1:train_index]
y_test <- y[(train_index+1):length(y)]

xreg_train <- xreg[1:train_index,]
xreg_test <- xreg[(train_index+1):length(y),]


# -------------------------------------------------------------------------
# ARIMAX 
# -------------------------------------------------------------------------

# nota: in arimax serve coerenza numerica (stesse scale di misura)
xreg_arimax <- xreg
# training set --> scaling
xreg_train_arimax <- xreg_arimax[1:train_index, ]
xreg_train_scaled <- scale(xreg_train_arimax)
#
scaling_center <- attr(xreg_train_scaled,"scaled:center")
scaling_scale <- attr(xreg_train_scaled,"scaled:scale")

# applico lo scaling sul test set
xreg_test_arimax <- xreg_arimax[(train_index+1):length(y), ]
xreg_test_scaled <- scale(xreg_test_arimax, center = scaling_center, scale = scaling_scale)

# as.matrix
xreg_train_arimax <- as.matrix(xreg_train_scaled)
xreg_test_arimax <- as.matrix(xreg_test_scaled)


# ARIMAX

library(forecast)

#componenti da inserire nella funzione arimax per ogni esplicativa
train_size <- dim(xreg_train_arimax)[1]
test_size <- dim(xreg_test_arimax)[1]


arimax_model <- Arima(y_train, order = c(1, 0, 1), xreg = xreg_train_arimax)
summary(arimax_model)
arimax_forecast <- forecast(arimax_model, xreg = xreg_test_arimax, h = 12)

plot(arimax_forecast, main = "Previsioni ARIMAX")
lines((train_size+1):(train_size+test_size), y_test, col = "red")
legend("topleft", legend = c("Previsioni", "Valori reali"), col = c("blue", "red"), lty = 1)

mae_arimax <- mean(abs(y_test - arimax_forecast$mean))
rmse_arimax <- sqrt(mean((y_test - arimax_forecast$mean)^2))

cat("MAE:", mae_arimax, "\n")
cat("RMSE:", rmse_arimax, "\n")


# -------------------------------------------------------------------------
# REGRESSIONE STATICA + SPIKEandSLAB
# -------------------------------------------------------------------------


# specifica della struttura del modello
# costruzione del modello strutturale --> obiettivo: isolare la parte strutturale da quella spiegata dalle covariate
ss_static <- list()
ss_static <- AddLocalLevel(ss_static,y) # catturare una tendenza "lenta e graduale" nei rendimenti, anche non costante nel tempo
# anche se i rendimenti non hanno un trend chiaro, il local level agisce come un termine di errore strutturale dinamico (migliora il fit)
ss_static <- AddSeasonal(ss_static, y, nseasons = 12) # stagionalità mensile
# anche se i rendimenti sono "rumorosi", includere una componente stagionale aiuta a catturare ciclicità latenti (e il modello può assegnare poca varianza se non rilevante)

# modello con REGRESSIONE STATICA + SPIKE-and-SLAB
model_static <- bsts(
  formula = y ~.,
  state.specification = ss_static,
  data = data.frame(y = as.numeric(y_train), xreg_train),
  niter = 5000,
  expected.model.size = ncol(xreg_train), # prior sul numero medio di variabili che pensi siano realmente rilevanti nel modello (qui nessuna penalizzazione iniziale)
  ping = 0,
  seed = 34
)

model_static$has.regression
summary(model_static)$coefficients 
plot(model_static, "coefficients") # analisi delle variabili (Spike&Slab)
abline(v = 0.5, col = "darkblue", lty = 2, lwd = 2)
# il modello statico trova che 6-7 covariate abbiano un impatto persistente sui rendimenti di JPMorgan


# previsioni
pred_static <- predict(model_static, newdata = xreg_test, horizon = dim(xreg_test)[1])
pred_static$mean # medie a posteriori
pred_static$interval # intervallo di credibilità

mae_static <- mean(abs(pred_static$mean - y_test))
rmse_static <- sqrt(mean((pred_static$mean - y_test)^2))

plot(pred_static, main = "Confronto previsioni statiche vs valori reali", ylim = c(-0.5,0.5))
lines((length(y) - nrow(xreg_test) + 1):length(y), y_test, col = "red", lwd = 2)
legend("bottomleft", legend = c("Previsioni", "Osservati"), col = c("black", "red"), lty = 1)


# -------------------------------------------------------------------------
# REGRESSIONE DINAMICA + Spike&Slab
# -------------------------------------------------------------------------

ss_dyn <- list()
ss_dyn <- AddLocalLevel(ss_dyn,y) 
ss_dyn <- AddSeasonal(ss_dyn, y, nseasons = 12)

ss_dyn <- AddDynamicRegression(
  state.specification = ss_dyn,
  formula = y ~.,
  data = data.frame(y = as.numeric(y_train), xreg_train)
)

model_dyn <-bsts(
  formula = y~.,
  state.specification = ss_dyn,
  niter = 5000,
  ping = 0, seed = 34,
  data = data.frame(y = as.numeric(y_train),xreg_train)
)

model_dyn$has.regression
summary(model_dyn)$coefficients
# visualizzo, per ogni covariata, la probabilità a posteriori di essere inclusa nel modella (vicina a 1=importante)
plot(model_dyn, "coefficients")
# praticamente tutti i PIP (Probabilità di inclusione a posteriori) sono prossimi a zero
# solo consumer_Credit ha una posterior inclusion porbability > 0.3
# anche covariate che nel modello statico erano sopra 0.5, ora spariscono
# questo perchè nel modello dinamico la stima è molto più difficile
# lo spike-and-slab valuta se intere traiettorie b_j[t] sono "utili"
# e penalizza fortemente se non spiegano variazioni temporali significative nella serie target
# motivi:
# spike and slab in verisone dinamica, non valuta solo l'importanza media di una variabile, ma se lìintera traiettoria temporale dei coef ha impatto sulla serie target
# se i coef non variano in modo rilevante o sono vicini a zero nel tempo, vengono penalizzati fortemente e la covariata viene scartata

# estrazione delle traiettorie a posteriori dei coefficienti beta[t]
coefs <- model_dyn$dynamic.regression.coefficients # array 3D [draw,time,covariate]
# media a posteriori nel tempo per ogni beta[t]
beta_mean <- apply(coefs, c(2,3), mean) # [tempo,covariate]

# plot delle traiettorie dei beta[t]
matplot(t(beta_mean), type = "l", lty = 1, lwd = 2,
        main = "Coefficienti dinamici stimati (posterior mean)",
        xlab = "Tempo", ylab = "Beta(t)")
for(i in 1:nrow(beta_mean)){
  text(x=ncol(beta_mean)+0.1,
       y=beta_mean[i,ncol(beta_mean)],
       labels = rownames(beta_mean)[i],
       col = i, cex = 0.8, pos = 4)
}
# se i beta[t] sono quasi costanti nel tempo, allora il modello dinamico tende a considerare le covariate "non dinamiche" --> "non rilevanti"

pred_dynamic <- predict(model_dyn, newdata = xreg_test, horizon = dim(xreg_test)[1])


# valutazione performance previsive
mae_dynamic <- mean(abs(pred_dynamic$mean - y_test))
rmse_dynamic <- sqrt(mean((pred_dynamic$mean - y_test)^2))


plot(pred_dynamic, main = "Confronto previsioni dinamiche vs valori reali", ylim = c(-0.5,0.5))
lines((length(y) - nrow(xreg_test) + 1):length(y), y_test, col = "red", lwd = 2)
legend("bottomleft", legend = c("Previsioni", "Osservati"), col = c("black", "red"), lty = 1)

# confronto
par(mfrow=c(1,2))
# plot statica
plot(pred_static$mean, type = "l", col = "blue", lwd = 2, ylim = c(-0.5,0.5),
     main = "Previsioni statiche vs Osservati", ylab = "JPM Returns")
lines(y_test, col = "red", lwd = 2, lty = 2)
lines(pred_static$interval[1,], col = "gray", lwd = 2, lty = 3)
lines(pred_static$interval[2,], col = "gray", lwd = 2, lty = 3)
legend("topleft", legend = c("Previsioni","Osservati","Intervalli"), col = c("blue","red","gray"),lty = c(1,2,3))
# plot dinamica
plot(pred_dynamic$mean, type = "l", col = "blue", lwd = 2, ylim = c(-0.5,0.5),
     main = "Previsioni dinamiche vs Osservati", ylab = "JPM Returns")
lines(y_test, col = "red", lwd = 2, lty = 2)
lines(pred_dynamic$interval[1,], col = "gray", lwd = 2, lty = 3)
lines(pred_dynamic$interval[2,], col = "gray", lwd = 2, lty = 3)
legend("topleft", legend = c("Previsioni","Osservati","Intervalli"), col = c("blue","red","gray"),lty = c(1,2,3))

# previsioni troppo piatte:
# il modello (soprattutto quello dinamico) non ha trovato covariate rilevanti --> tende a prevedere la media storica
# Spike&Slab esclude tutto (o quasi) --> il modello fa previsioni con la sola componente locale/stagionale
#
# se i coefficienti dinamici sono quasi costanti nel tempo allora anche le previsioni non cambiano molto (comportamento simile a quello statico, ecco perchè previsioni simili)


# confronto statico vs dinamico
cat("MAE statico:", round(mae_static, 4), "\n")
cat("MAE dinamico:", round(mae_dynamic, 4), "\n")
cat("RMSE statico:", round(rmse_static, 4), "\n")
cat("RMSE dinamico:", round(rmse_dynamic, 4), "\n")


# install.packages("CausalImpact")
library(CausalImpact)
CompareBstsModels(list(model_static, model_dyn), colors = c("blue","green"))
# il modello dinamico ha migliore adattamento locale (mae e rmse più bassi localmente - orizzonte previsivo)
# ma ha peggiore stabilità globale (errore cumulativo assoluto più alto del modello statico) --> il modello statico è più robusto e regolare nel lungo periodo
# questo risultato riflette il tradeoff tra modelli più semplici (statico) e modelli più flessibili (dinamici)
# il modello statico "vince" sul training: ha pochi coefficienti fissi, stima bene la media e la struttura globale della serie
# accumula meno errore assoluto su lungo periodo
# il modello dinamico stima più coefficienti (che cambiano nel tempo) in modo più flessibile
# quando entri nella fase finale (di test) può meglio adattarsi ai pattern recenti anche se con rumore
# migliore fitting locale --> coglie meglio gli ultimi cambiamenti
# questo suggerisce che, in contesti con dinamiche instabili o shock recenti (tipo borsa), un approccio dinamico può fornire previsioni più reattive



# -------------------------------------------------------------------------
# STAN
# -------------------------------------------------------------------------


library(rstan)

nT <- length(y_train)
p <- ncol(xreg_train)


# -------------------------------------------------------------------------
# STAN con BETA specifici dal MODELLO STATICO
# -------------------------------------------------------------------------


coef_static <- summary(model_static)$coefficients
coef_static <- coef_static[rownames(coef_static) != c("(Intercept)","Posterior Mean"),]
beta_static <- coef_static[,1]
beta_static <- beta_static[colnames(xreg_test)]

stan_new <- list(
  T = nT,
  p = p,
  X = xreg_train,
  y = as.numeric(y_train),
  H = nrow(xreg_test),
  X_future = xreg_test,
  beta_static = beta_static
)


my_init <- function() {
  list(
    z = matrix(rnorm(p*nT,0,0.05), nrow = p), # piccoli disturbi attorno a traiettoria piatta
    sigma_beta = rep(0.1,p),
    sigma_y = 0.1
  )
}
# così ottengo beta inizializzati attorno a beta_static e la dinamica di z si esprime senza divergenza

# abilita uso di tutti i core disponibili
options(mc.cores = parallel::detectCores())

fit_new <- stan(file = "specifici beta.stan",
            data = stan_new,
            chains = 4,
            iter = 10000,
            warmup = 2000,
            seed = 123,
            init = my_init,
            control = list(adapt_delta = 0.99, max_treedepth = 25))



rstan::check_hmc_diagnostics(fit_new)

post2 <- extract(fit_new)

# traiettorie beta
beta_mean2 <- apply(post2$beta, c(2,3), mean) # [p,T]
matplot(t(beta_mean2), type = "l", lty = 1, col = 1:p,
        main = "Posterior mean dei beta[t]", ylab = expression(beta), xlab="Tempo")
legend("topright", legend = colnames(xreg_train), lty = 1, col = 1:p)

# forecast
y_future_mean2 <- apply(post2$y_future, 2, mean)

plot(y_test, type = "l", col = "black", lwd = 2,
     main = "Previsione dei Rendimenti", ylab = "y", ylim = c(-0.15,0.15))
lines(y_future_mean2, col = "red", lwd = 2)
legend("bottomright", legend = c("Osservati","Previsti"), col = c("black","red"), lty = 1)

# le previsioni seguono un pattern "inverso" rispetto ai valori osservati (curve si muovono "a specchio")

saveRDS(fit_new, file = "fit_stan_beta_statici.rds")
# readRDS("fit_stan_beta_statici.rds")


library(loo)
log_lik_mat <- extract_log_lik(fit_new, "log_lik")
waic(log_lik_mat) # WAIC
loo(log_lik_mat) # Leave-One-Out


# -------------------------------------------------------------------------
# STAN modello MISTO --> Statico + Dinamico
# -------------------------------------------------------------------------


coef_static <- summary(model_static)$coefficients
coef_static <- coef_static[rownames(coef_static) != c("(Intercept)","Posterior Mean"),]
beta_static <- coef_static[,1]
beta_static <- beta_static[colnames(xreg_test)]


# suddivisione basata su interpetazione economica e di comportamento temporale atteso
x_dyn <- xreg_train %>%
  select(DFF,ICSA,dismissal,GEPUCURRENT,VIX) %>%
  as.matrix()

x_future_dyn <- xreg_test %>%
  select(DFF,ICSA,dismissal,GEPUCURRENT,VIX) %>%
  as.matrix()

x_static <- xreg_train %>%
  select(salary,CPI,Inflation,Consumer_credit) %>%
  as.matrix()

x_future_static <- xreg_test %>%
  select(salary,CPI,Inflation,Consumer_credit) %>%
  as.matrix()


p_dyn <- ncol(x_dyn)
p_static <- ncol(x_static)
H <- nrow(x_future_dyn)

# valore iniziale dai beta statici per le variabili dinamiche
beta_init <- beta_static[c("DFF","ICSA","dismissal","GEPUCURRENT","VIX")]

stan_mix <- list(
  T = nT,
  p_dyn = p_dyn,
  p_static = p_static,
  X_dyn = x_dyn,
  X_static = x_static,
  y = as.numeric(y_train),
  H = H,
  X_future_dyn = x_future_dyn,
  X_future_static = x_future_static,
  beta_static = beta_init
)

fit_mix <- stan(
  file = "modello misto.stan",
  data = stan_mix,
  chains = 4,
  iter = 10000,
  warmup = 1000,
  seed = 123,
  control = list(max_treedepth = 25, adapt_delta = 0.95)
)


saveRDS(fit_mix,"modello_fit_misto.rds")

#
library(loo)
log_lik_mat2 <- extract_log_lik(fit_mix, "log_lik")
waic(log_lik_mat2) # WAIC
loo(log_lik_mat2) # Leave-One-Out



post_mix <- extract(fit_mix)

# traiettorie temporali dei beta
beta_mean_mix <- apply(post_mix$beta, c(2,3), mean) # [p,T]
matplot(t(beta_mean_mix), type = "l", lty = 1, col = 1:p,
        main = "Posterior mean dei beta[t]", ylab = expression(beta), xlab="Tempo")

# forecast
y_future_mean_mix <- apply(post_mix$y_future, 2, mean)
y_future_lower_mix <- apply(post_mix$y_future, 2, quantile, probs = 0.025)
y_future_upper_mix <- apply(post_mix$y_future, 2, quantile, probs = 0.975)

plot(y_test, type = "l", col = "black", lwd = 2,
     main = "Previsione dei Rendimenti", ylab = "y", ylim = c(-0.15,0.18))
lines(y_future_mean_mix, col = "red", lwd = 2)
legend("topright", legend = c("Osservati","Previsti"), col = c("black","red"), lty = 1, lwd = 2)

MAE_mix <- mean(abs(y_future_mean_mix- y_test))
RSME_mix <- sqrt(mean((y_future_mean_mix - y_test)^2))


# -------------------------------------------------------------------------
# STAN beta AUTOREGRESSIVI + covariate LAGGED
# -------------------------------------------------------------------------

library(rstan)

xreg_lagged <- xreg %>%
  mutate(across(everything(), ~lag(.x,1))) %>%
  filter(complete.cases(.))

y_lagged <- window(y, start = time(y)[2])
y_train_l <- y_lagged[1:train_index]
y_test_l <- y_lagged[(train_index+1):length(y_lagged)]

xreg_train_lagged <- xreg_lagged[1:train_index,]
xreg_test_lagged <- xreg_lagged[(train_index+1):length(y_lagged),]


# modello statico con covariate ritardate (previsione causale)
ss_static_lag <- list()
ss_static_lag <- AddLocalLevel(ss_static_lag, y_lagged) 
ss_static_lag <- AddSeasonal(ss_static_lag, y_lagged, nseasons = 12)

model_static_lag <- bsts(
  formula = y ~.,
  state.specification = ss_static_lag,
  data = data.frame(y = as.numeric(y_train_l), xreg_train_lagged),
  niter = 15000,
  expected.model.size = ncol(xreg_train_lagged),
  ping = 0,
  seed = 34
)

model_static_lag$has.regression
summary(model_static_lag)$coefficients 
plot(model_static_lag, "coefficients") # analisi delle variabili (Spike&Slab)
abline(v = 0.5, col = "darkblue", lty = 2, lwd = 2)
# il modello statico trova che 6-7 covariate abbiano un impatto persistente sui rendimenti di JPMorgan

# previsioni
pred_static_lag <- predict(model_static_lag, newdata = xreg_test_lagged, horizon = dim(xreg_test_lagged)[1])
pred_static_lag$mean # medie a posteriori
pred_static_lag$interval # intervallo di credibilità

mae_static_lag <- mean(abs(pred_static_lag$mean - y_test_l))
rmse_static_lag <- sqrt(mean((pred_static_lag$mean - y_test_l)^2))

plot(pred_static_lag, main = "Confronto previsioni statiche vs valori reali", ylim = c(-0.5,0.5))
lines((length(y) - nrow(xreg_test_lagged) + 1):length(y), y_test, col = "red", lwd = 2)
legend("bottomleft", legend = c("Previsioni", "Osservati"), col = c("black", "red"), lty = 1)

# coefficienti stimati modello STATICO + LAG covariate
coef_static_lag <- summary(model_static_lag)$coefficients
coef_static_lag <- coef_static_lag[rownames(coef_static_lag) != c("(Intercept)","Posterior Mean"),]
beta_static_lag <- coef_static_lag[,1]
beta_static_lag <- beta_static_lag[colnames(xreg_test_lagged)]


# STAN

nT <- length(y_train_l)
p <- ncol(xreg_train_lagged)

stan_ar_lag <- list(
  T = nT,
  p = p,
  X = as.matrix(xreg_train_lagged),
  y = as.numeric(y_train_l),
  H = nrow(xreg_test_lagged),
  X_future = as.matrix(xreg_test_lagged),
  beta_static = beta_static_lag
)


fit_ar_lag <- stan(file = "beta autoregressivi.stan",
                   data = stan_ar_lag,
                   chain = 4,
                   iter = 10000,
                   warmup = 1000,
                   seed = 123,
                   control = list(adapt_delta = 0.95, max_treedepth = 25))

saveRDS(fit_ar_lag, file = "fit_stan_beta_autoregressivi.rds")


post_lag <- extract(fit_ar_lag)

# traiettorie temporali dei beta
beta_mean_lag <- apply(post_lag$beta, c(2,3), mean) # [p,T]


# forecast
y_future_mean_lag <- apply(post_lag$y_future, 2, mean)
y_future_lower <- apply(post_lag$y_future, 2, quantile, probs = 0.025)
y_future_upper <- apply(post_lag$y_future, 2, quantile, probs = 0.975)

plot(y_test_l, type = "l", col = "black", lwd = 2,
     main = "Previsione dei Rendimenti", ylab = "y")
lines(y_future_mean_lag, col = "red", lwd = 2)
legend("topright", legend = c("Osservati","Previsti"), col = c("black","red"), lty = 1, lwd = 2)

MAE_ar <- mean(abs(y_future_mean_lag - y_test_l))
RSME_ar <- sqrt(mean((y_future_mean_lag - y_test_l)^2))

# le previsioni non sono più speculari e stanno seguendo un po' la dinamica reale ma con una certa "piattezza"
# il MAE è molto buono, indicando che già stiamo prevedendo meglio di una baseline o media mobile (vedi mediana di y)

log_lik_mat3 <- extract_log_lik(fit_ar_lag, "log_lik")
waic(log_lik_mat3) # WAIC
loo(log_lik_mat3) # Leave-One-Out
