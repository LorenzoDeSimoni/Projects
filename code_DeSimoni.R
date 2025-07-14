
# PROGETTO DATA MINING ----------------------------------------------------
library(skimr)
library(corrplot)
library(caret)
library(dplyr)
library(glmnet)
library(grpreg)
library(stringr)
library(tidyr)
library(purrr)
library(ranger)


#IMPORTO DATI  ------------------------------------------------------

#data <- read.csv('training.csv', header = TRUE)
data <- read.csv('test.csv', header = TRUE)
head(data)
View(data)
dim(data)


# PREPROCESSING -----------------------------------------------------

skimr::skim(data)

freq_missing <- apply(data, 2, function(x) sum(is.na(x))) 
freq_missing[freq_missing > 0] 

#par(mfrow = c(1,2))
#hist(data$selling_price, xlab = "Price", main = "SalePrice")
#hist(log(data$selling_price), xlab = "Price", main = "Logarithm of SalePrice")


#var:rooms_number
table(data$rooms_number,useNA = "always")
data$rooms_number[data$rooms_number == '5+'] <- '6'
data$rooms_number = as.integer(data$rooms_number)


#var: bathroom number ha 25 NA  
table(data$bathrooms_number,useNA = "always")
data[is.na(data$bathrooms_number),] #ci sono case che costano di piu e di meno, provo ad imputare
data$bathrooms_number[data$bathrooms_number == '3+'] <- 4
#trasformo in numeric
data$bathrooms_number <- as.integer(data$bathrooms_number)

#imputo con numero medio di bagni per numero di stanze
bathroom_means_by_room <- data %>%
  group_by(rooms_number) %>%
  summarise(mean_baths = mean(bathrooms_number, na.rm = TRUE))
data <- left_join(data, bathroom_means_by_room, by = "rooms_number")
data$bathrooms_number[is.na(data$bathrooms_number)] <- data$mean_baths[is.na(data$bathrooms_number)]
data$mean_baths <- NULL
data$bathrooms_number = round(data$bathrooms_number)
data$bathrooms_number= as.integer(data$bathrooms_number)
View(data)


#VAR:floor
#metto tutto minuscolo 
data$floor <- tolower(data$floor)
table(data$floor,useNA = "always")

#imputo con -1,0,0.5
data$floor[data$floor == "semi-basement"] <- -1
data$floor[data$floor == "ground floor"] <- 0
data$floor[data$floor == "mezzanine"] <- 0.5
data$floor
data$floor <- as.numeric(data$floor)


#var:total_floors
table(data$total_floors_in_building,useNA = "always")
data$total_floors_in_building[data$total_floors_in_building == '1 floor'] <- 1
data$total_floors_in_building = as.numeric(data$total_floors_in_building)
#imputo a 20 i valori 21,22,23,24 e 27 che sono poche case
data$total_floors_in_building[data$total_floors_in_building > 16] <- 17 #QUAAAAAAAAAA
#per l'imputazione di NA posso prendere di riferimento il piano a cui si trova la casa 
median_floors <- median(data$total_floors_in_building, na.rm = TRUE)
data$total_floors_in_building[is.na(data$total_floors_in_building)] <- 
  pmax(ceiling(data$floor[is.na(data$total_floors_in_building)]), median_floors)
str(data$total_floors_in_building)

#var:lift
table(data$lift,useNA = "always")
data$lift = as.numeric(as.factor(data$lift))-1
mode_lift <- as.numeric(names(sort(table(data$lift), decreasing = TRUE))[1])
#imputazione usando il numero di piani nell'edificio 
data$lift[is.na(data$lift) & data$total_floors_in_building >= 4] <- 1
data$lift[is.na(data$lift) & data$total_floors_in_building <= 2] <- 0
data$lift[is.na(data$lift) & data$total_floors_in_building == 3] <- mode_lift
data$lift = as.factor(data$lift)


#var: car pariking
table(data$car_parking,useNA = "always")
unique(data$car_parking)
#inizializza nuove colonne a 0
data$n_parking_box <- 0
data$n_parking_shared <- 0
#funzione per estrarre numeri in garage/box
extract_parking <- function(string, type) {
  pattern <- paste0("(\\d+) in ", type)
  match <- regmatches(string, regexpr(pattern, string))
  ifelse(length(match) > 0,
         as.numeric(gsub(paste0(" in ", type), "", match)),
         0)
}
#applica funzione a ogni riga
for (i in seq_len(nrow(data))) {
  string <- tolower(data$car_parking[i])
  data$n_parking_box[i] <- extract_parking(string, "garage/box")
  data$n_parking_shared[i] <- extract_parking(string, "shared parking")
}
#View(data)
data$car_parking <- NULL



#VAR:zone  #zone uniche = quadrilatero della moda->Brera , parco lambro , via fra cristoforo, 1 NA
table(data$zone,useNA = "always")

unique(data$zone)
#cerco le modalità con meno di 10 osservazioni per accorparle a quella piu vicina 
table(data$zone)[table(data$zone) < 10]
data$zone[data$zone == "via fra' cristoforo"] <- "famagosta"
data$zone[data$zone == "quadrilatero della moda"] <- "brera"
data$zone[data$zone == "parco lambro"] <- "cimiano"
data$zone[data$zone == "cascina gobba"] <- "crescenzago"
#data$zone[data$zone == "figino"] <- "gallaratese"
data$zone[data$zone == "lanza"] <- "brera"
#data$zone[data$zone == "bovisasca"] <- "bovisa"
data$zone[data$zone == "qt8"] <- "monte stella"
data$zone[data$zone == "rogoredo"] <- "santa giulia"
data$zone[data$zone == "san babila"] <- "duomo"
data$zone[data$zone == "sant'ambrogio"] <- "cadorna - castello"
data$zone[data$zone == "scala - manzoni"] <- "duomo"
data$zone[data$zone == "via calizzano"] <- "comasina"
data$zone[data$zone == "via canelli"] <- "udine"
data$zone[data$zone == "largo caioroli 2"] <- "brera"
data$zone[data$zone == "via marignano, 3"] <- 'santa giulia' 
data$zone[data$zone == "corso magenta"] <- "cadorna - castello"

#imputo con moda che è città studi l'unica osservazione NA
data$zone[is.na(data$zone)] <- "città studi"

# #calcolo distanza dal duomo 

# zone_data <- c(
#   "affori" = 6.5,
#   "amendola - buonarroti" = 3.0,"arco della pace" = 1.5,"arena" = 1.2,"argonne - corsica" = 3.5,
#   "ascanio sforza" = 2.0,"baggio" = 8.0,"bande nere" = 4.5,"barona" = 5.5,
#   "bicocca" = 6.5,"bignami - ponale" = 7.0,"bisceglie" = 7.5,
#   "bocconi" = 2.0,"bologna - sulmona" = 3.5,"borgogna - largo augusto" = 1.0,
#   "bovisa" = 5.0,"bovisasca" = 6.0,"brera" = 0.8,"bruzzano" = 7.0,
#   "buenos aires" = 2.0,"ca' granda" = 5.5,"cadore" = 2.0,"cadorna - castello" = 1.0,
#   "cantalupa - san paolo" = 5.0,"carrobbio" = 1.0,"cascina dei pomi" = 7.5,
#   "cascina gobba" = 8.0,"cascina merlata - musocco" = 8.5,"casoretto" = 4.0,
#   "cenisio" = 3.0,"centrale" = 3.0,"cermenate - abbiategrasso" = 4.5,"certosa" = 7.0,
#   "chiesa rossa" = 5.0,"cimiano" = 5.5,"città studi" = 3.5,"city life" = 2.5,"comasina" = 7.5,
#   "corso genova" = 1.5,"corso san gottardo" = 2.0,"corvetto" = 4.0,
#   "crescenzago" = 6.5,"de angeli" = 3.5,"dergano" = 4.5,"dezza" = 2.5,"duomo" = 0.0,"famagosta" = 4.5,
#   "farini" = 3.5,"figino" = 9.0,"frua" = 3.0,
#   "gallaratese" = 7.5,"gambara" = 4.0,
#   "garibaldi - corso como" = 2.0,"ghisolfa - mac mahon" = 4.0,"giambellino" = 4.5,
#   "gorla" = 6.0,"gratosoglio" = 7.0,"greco - segnano" = 5.5,
#   "guastalla" = 1.5,"indipendenza" = 2.0,"inganni" = 5.0,"isola" = 2.5,
#   "istria" = 5.0,"lambrate" = 5.0,"lanza" = 1.0,"lodi - brenta" = 3.0,"lorenteggio" = 5.0,
#   "maggiolina" = 4.0,"martini - insubria" = 3.0,"melchiorre gioia" = 3.5,
#   "missori" = 0.5,"molise - cuoco" = 4.0,
#   "monte rosa - lotto" = 4.0,"monte stella" = 5.5,"montenero" = 1.5,"morgagni" = 3.0,"moscova" = 1.5,"muggiano" = 9.5,
#   "navigli - darsena" = 2.0,"niguarda" = 6.5,
#   "ortica" = 5.5,"pagano" = 2.5,"palestro" = 1.5,
#   "paolo sarpi" = 2.0,"parco lambro" = 6.0,"parco trotter" = 4.5,
#   "pasteur" = 3.5,"pezzotti - meda" = 3.5,"piave - tricolore" = 1.5,
#   "piazza napoli" = 3.5,
#   "piazzale siena" = 4.0,"plebisciti - susa" = 2.5,"ponte lambro" = 7.5,
#   "ponte nuovo" = 6.5,"porta nuova" = 1.5,
#   "porta romana - medaglie d'oro" = 1.5,"porta venezia" = 1.5,"porta vittoria" = 2.0,
#   "portello - parco vittoria" = 4.0,"prato centenaro" = 5.5,
#   "precotto" = 6.0,"primaticcio" = 5.0,
#   "qt8" = 6.0,"quadrilatero della moda" = 0.5,
#   "quadronno - crocetta" = 1.0,
#   "quartiere adriano" = 7.0,"quartiere feltre" = 5.0,
#   "quartiere forlanini" = 6.0,"quartiere olmi" = 8.5,
#   "quarto cagnino" = 7.0,"quarto oggiaro" = 7.5,
#   "quinto romano" = 8.0,"quintosole - chiaravalle" = 8.5,"repubblica" = 2.0,
#   "ripamonti" = 5.0,"rogoredo" = 6.5,"roserio" = 8.0,"rovereto" = 4.0,
#   "rubattino" = 5.5,"san babila" = 0.5,"san carlo" = 1.0,
#   "san siro" = 5.5,"san vittore" = 1.5,"sant'ambrogio" = 1.0,"santa giulia" = 6.5,"scala - manzoni" = 0.5,
#   "sempione" = 2.0,"solari" = 2.5,"ticinese" = 1.5,
#   "tre castelli - faenza" = 6.5,"trenno" = 8.0,"tripoli - soderini" = 4.5,"turati" = 1.5,"turro" = 5.0,
#   "udine" = 5.0,"vercelli - wagner" = 3.5,
#   "via calizzano" = 7.0,
#   "via canelli" = 6.0,"via fra' cristoforo" = 3.0,"vialba" = 8.0,
#   "viale ungheria - mecenate" = 7.0,
#   "vigentino - fatima" = 6.0,
#   "villa san giovanni" = 6.5,"vincenzo monti" = 2.0,"washington" = 3.0,"zara" = 3.5
# )
# 
# 
# 
# 
# zone_data <- data.frame(
#   Zona = c(
#     "quadronno - crocetta", "porta romana - medaglie d'oro", "gallaratese",
#     "martini - insubria", "navigli - darsena", "giambellino",
#     "morgagni", "ticinese", "de angeli", "palestro",
#     "cermenate - abbiategrasso", "vigentino - fatima", "rovereto", "brera",
#     "pezzotti - meda", "dergano", "ghisolfa - mac mahon", "crescenzago",
#     "villa san giovanni", "barona", "dezza", "quartiere olmi", "famagosta",
#     "pagano", "tre castelli - faenza", "viale ungheria - mecenate",
#     "cascina dei pomi", "san siro", "san vittore", "baggio", "isola",
#     "cantalupa - san paolo", "corvetto", "bruzzano", "indipendenza",
#     "bisceglie", "quinto romano", "amendola - buonarroti",
#     "portello - parco vittoria", "città studi", "porta venezia", "pasteur",
#     "cascina merlata - musocco", "niguarda", "bovisa", "cimiano",
#     "quartiere adriano", "arco della pace", "corso genova",
#     "piave - tricolore", "piazzale siena", "parco trotter", "cadore",
#     "repubblica", "molise - cuoco", "roserio", "centrale", "greco - segnano",
#     "gambara", "ortica", "rogoredo", "buenos aires", "bignami - ponale",
#     "certosa", "tripoli - soderini", "bologna - sulmona", "lodi - brenta",
#     "udine", "precotto","monte rosa - lotto", "sempione", "corso san gottardo", "montenero", "turro",
#     "turati", "chiesa rossa", "bande nere", "quartiere forlanini", "ponte nuovo",
#     "borgogna - largo augusto", "cenisio", "gorla", "bovisasca", "primaticcio",
#     "affori", "argonne - corsica", "quarto oggiaro", "porta vittoria", "maggiolina",
#     "bocconi", "san carlo", "missori", "paolo sarpi", "ripamonti",
#     "casoretto", "vercelli - wagner", "istria", "ca' granda", "vialba",
#     "prato centenaro", "guastalla", "quintosole - chiaravalle", "vincenzo monti", "rubattino",
#     "farini", "moscova", "inganni", "comasina", "washington",
#     "ascanio sforza", "solari", "melchiorre gioia", "quarto cagnino", "zara",
#     "arena", "city life", "frua", "garibaldi - corso como", "gratosoglio",
#     "monte stella", "duomo", "bicocca", "ponte lambro", "trenno",
#     "lambrate", "carrobbio", "lorenteggio", "piazza napoli", "plebisciti - susa",
#     "muggiano", "quartiere feltre", "porta nuova", "cadorna - castello","santa giulia"),
#   Latitudine = c(
#     45.4565, 45.4522, 45.4898, 45.4559, 45.4509, 45.4399,
#     45.4789, 45.4550, 45.4680, 45.4689, 45.4322, 45.4290, 45.5012, 45.4705,
#     45.4450, 45.5000, 45.4920, 45.5060, 45.5240, 45.4360, 45.4570, 45.4440,
#     45.4300, 45.4660, 45.4720, 45.4450, 45.4870, 45.4780, 45.4600, 45.4660,
#     45.4860, 45.4350, 45.4370, 45.5310, 45.4640, 45.4540, 45.4660, 45.4700,
#     45.4840, 45.4735, 45.4720, 45.4890, 45.5060, 45.5200, 45.5060, 45.5060,
#     45.5190, 45.4720, 45.4560, 45.4660, 45.4580, 45.4890, 45.4550, 45.4790,
#     45.4480, 45.5140, 45.4840, 45.5100, 45.4600, 45.4730, 45.4290, 45.4780,
#     45.5300, 45.5020, 45.4520, 45.4430, 45.4450, 45.4870, 45.5140, 45.478053,
#     45.47583, 45.451507, 45.4545, 45.49806,
#     45.47500, 45.43056, 45.4617, 45.4563, 45.4975,
#     45.4647, 45.4850, 45.5060, 45.5080, 45.4590,
#     45.5180, 45.4640, 45.5160, 45.4560, 45.4950,
#     45.4470, 45.4650, 45.4580, 45.4810, 45.4380,
#     45.4850, 45.4640, 45.5060, 45.5080, 45.5230,
#     45.5090, 45.4600, 45.4090, 45.4700, 45.4720,
#     45.4910, 45.4770, 45.4560, 45.5300, 45.4590,
#     45.4480, 45.4580, 45.4880, 45.4700, 45.4890,
#     45.4760, 45.4780, 45.4620, 45.4830, 45.4080,
#     45.4880, 45.4640, 45.5260, 45.4160, 45.4910,
#     45.4820, 45.4570, 45.4370, 45.4550, 45.4620,
#     45.4590, 45.4750, 45.4810, 45.4700, 45.4295
#   ),
#   Longitudine = c(
#     9.1942, 9.2025, 9.1172, 9.2206, 9.1716, 9.1192,
#     9.2153, 9.1800, 9.1530, 9.2015, 9.1705, 9.1960, 9.2220, 9.1874,
#     9.1840, 9.1810, 9.1540, 9.2360, 9.2190, 9.1500, 9.1670, 9.0740,
#     9.1670, 9.1650, 9.1820, 9.2550, 9.1430, 9.1230, 9.1680, 9.0900,
#     9.1900, 9.1400, 9.2190, 9.1650, 9.2100, 9.1070, 9.0700, 9.1570,
#     9.1510, 9.2225, 9.2010, 9.2150, 9.0950, 9.1830, 9.1700, 9.2360,
#     9.2460, 9.1720, 9.1770, 9.2100, 9.1330, 9.2200, 9.2100, 9.1960,
#     9.2220, 9.1000, 9.2040, 9.2000, 9.1350, 9.2360, 9.2460, 9.2100,
#     9.2140, 9.1300, 9.1400, 9.2200, 9.2100, 9.2370, 9.2250, 9.145017,
#     9.17250, 9.180135, 9.2041, 9.225554,
#     9.19389, 9.17250, 9.1372, 9.2475, 9.2491,
#     9.1970, 9.1650, 9.2220, 9.1800, 9.1280,
#     9.1740, 9.2290, 9.1310, 9.2120, 9.2050,
#     9.1890, 9.2000, 9.1890, 9.1770, 9.1990,
#     9.2260, 9.1530, 9.1900, 9.2000, 9.1240,
#     9.2100, 9.2020, 9.2360, 9.1680, 9.2460,
#     9.1840, 9.1870, 9.1200, 9.1700, 9.1570,
#     9.1820, 9.1630, 9.1980, 9.1030, 9.1930,
#     9.1790, 9.1540, 9.1430, 9.1880, 9.1650,
#     9.1400, 9.1890, 9.2130, 9.2650, 9.1070,
#     9.2420, 9.1830, 9.1230, 9.1540, 9.2220,
#     9.0480, 9.2370, 9.1900, 9.1730, 9.2462
#   )
# )
# 
# # Definisci le coordinate del Duomo di Milano
# duomo_lat <- 45.4642
# duomo_lon <- 9.1900
# 
# # Funzione per calcolare la distanza usando la formula dell'Haversine
# haversine <- function(lat1, lon1, lat2, lon2) {
#   R <- 6371  # Raggio medio della Terra in km
#   dlat <- (lat2 - lat1) * pi / 180
#   dlon <- (lon2 - lon1) * pi / 180
#   a <- sin(dlat/2)^2 + cos(lat1 * pi / 180) * cos(lat2 * pi / 180) * sin(dlon/2)^2
#   c <- 2 * atan2(sqrt(a), sqrt(1 - a))
#   R * c
# }
# 
# # Applica la funzione a ogni riga del dataset
# zone_data$dist_from_duomo_km <- mapply(
#   haversine,
#   zone_data$Latitudine,
#   zone_data$Longitudine,
#   duomo_lat,
#   duomo_lon
# )
# zone_data=zone_data[,-c(2,3)]
# setdiff(data$zone, zone_data$Zona)
# dim(data)
# dim(zone_data)

#data = merge(data, zone_data, by.x = 'zone', by.y = 'Zona')

#VAR:condominium fees
table(data$condominium_fees,useNA = "always")
unique(data$condominium_fees)
data$condominium_fees[data$condominium_fees == 'No condominium fees'] <- '0'
data$condominium_fees <- as.numeric(gsub(",", "", data$condominium_fees))

data$condominium_fees <- ifelse(data$condominium_fees>5000,
                                NA, data$condominium_fees)
data$condominium_fees <- ifelse(data$condominium_fees>=1 & data$condominium_fees < 10,
                                NA, data$condominium_fees)
#posso fare meglio forse ci sono case con 1euro di condominium fees?
#IMPUTAZIONE NA CON MEDIANA
library(dplyr)
#calcolo mediana delle spese per ogni zona
zone_medians <- data %>%
  group_by(zone) %>%
  summarise(median_fee = median(condominium_fees, na.rm = TRUE))
#unisco la mediana alla tabella
data <- left_join(data, zone_medians, by = "zone")
#imputa NA con la mediana di zona
data$condominium_fees <- ifelse(
  is.na(data$condominium_fees),
  data$median_fee,
  data$condominium_fees)
data$median_fee <- NULL






skimr::skim(data)

#VAR:heating centralized
table(data$heating_centralized,useNA = "always")
unique(data$heating_centralized)
data$heating_centralized[is.na(data$heating_centralized) &
                                data$total_floors_in_building <= 3 &
                                data$year_of_construction > 1980] <- "independent"
data$heating_centralized[is.na(data$heating_centralized)] <- "central"
data$heating_centralized = as.numeric(as.factor(data$heating_centralized))-1
data$heating_centralized = as.factor(data$heating_centralized)


#VAR:conditions
table(data$conditions,useNA = "always")
unique(data$conditions)
data$conditions[is.na(data$conditions)] <- "excellent / refurbished"
data$conditions<- recode(data$conditions,
                                  "to be refurbished" = 1,
                                  "good condition / liveable" = 2,
                                  "excellent / refurbished" = 3,
                                  "new / under construction" = 4
)



# #VAR:year of constructions 
 table(data$year_of_construction ,useNA = "always")
 unique(data$year_of_construction)
 
 data <- data %>%
   group_by(zone) %>%
   mutate(year_of_construction = ifelse(is.na(year_of_construction),
                                        floor(median(year_of_construction, na.rm = TRUE)),
                                        year_of_construction)) %>%
   ungroup()
str(data$year_of_construction)


 #VAR:energy efficency class
 table(data$energy_efficiency_class,useNA = "always")
 unique(data$conditions)
 data$energy_efficiency_class[data$energy_efficiency_class == ','] <- NA
 #crea una nuova variabile di supporto con le classi imputate
 data$energy_efficiency_class_imputed <- NA
 
 data$energy_efficiency_class_imputed[data$year_of_construction < 1945] <- "g"
 data$energy_efficiency_class_imputed[data$year_of_construction >= 1945 & data$year_of_construction < 1970] <- "f"
 data$energy_efficiency_class_imputed[data$year_of_construction >= 1970 & data$year_of_construction < 1990] <- "e"
 data$energy_efficiency_class_imputed[data$year_of_construction >= 1990 & data$year_of_construction < 2005] <- "d"
 data$energy_efficiency_class_imputed[data$year_of_construction >= 2005 & data$year_of_construction < 2015] <- "c"
 data$energy_efficiency_class_imputed[data$year_of_construction >= 2015 & data$year_of_construction <= 2020] <- "b"
 data$energy_efficiency_class_imputed[data$year_of_construction > 2020] <- "a"
 
 data$energy_efficiency_class[is.na(data$energy_efficiency_class)] <- 
   data$energy_efficiency_class_imputed[is.na(data$energy_efficiency_class)]

data$energy_efficiency_class_imputed = NULL
data$energy_efficiency_class <- as.numeric(factor(
  data$energy_efficiency_class,
  levels = c("g", "f", "e", "d", "c", "b", "a"),
  ordered = TRUE
))


#VAR: avability
table(data$availability,useNA = "always")
unique(data$availability)
data <- data %>%
  mutate(
    availability_status = case_when(
      availability == "available" ~ "available",
      grepl("^available from", availability) ~ "soon available",
      is.na(availability) ~ "not available"
    ) %>% factor(levels = c("available", "soon available", "not available"))
  )



data$availability_status
data$availability = NULL

skimr::skim(data)



#VAR: Other Features
table(data$other_features,useNA = "always")
unique(data$other_features)

data$other_features[is.na(data$other_features)] <- "No"
skimr::skim(data)

df <- data %>%
  mutate(
    other_features = tolower(other_features),
    other_features = str_replace_all(other_features, "\\s+", " "),
    other_features = str_trim(other_features),
    id = row_number()
  )

df_tokens <- df %>%
  separate_rows(other_features, sep = "\\s*\\|\\s*")

#ACCORPO WINDOW ED EXPOSURE
exposure_tokens <- c(
  "double exposure", "internal exposure", "external exposure",
  "exposure north", "exposure south", "exposure east", "exposure west",
  "exposure north, south", "exposure north, east", "exposure north, west", 
  "exposure south, east", "exposure south, west", "exposure east, west",
  "exposure north, south, east", "exposure north, south, west", 
  "exposure north, south, east, west", "exposure south, east, west",
  "exposure north, east, west"
)

window_tokens <- df_tokens %>%
  filter(str_detect(other_features, "window frames")) %>%
  distinct(other_features) %>%
  pull()

#SELEZIONO IO I TOKEN MANUALMENTE
selected_tokens <- c(
  "security door", "balcony", "cellar", "terrace", "furnished",
  "partially furnished", "alarm system", "optic fiber", "video entryphone",
  "electric gate", "shared garden", "centralized tv system","pool",'full day concierge',
  'closet', 'terrace', 'private garden', 'tavern','half-day concierge') 

luxury_tokens <- c(
  "tennis court", "hydromassage", "reception",
  "private and shared garden", "6 balconies", "no",
  "disabled access", "8 balconies", "property land1 balcony"
)

base_id <- df %>% dplyr::select(id)

#PIU FREQUENTI
df_dummies_selected <- df_tokens %>%
  filter(other_features %in% selected_tokens) %>%
  distinct(id, other_features) %>%
  mutate(value = 1) %>%
  pivot_wider(names_from = other_features, values_from = value)


#LUXURY FEATURES
df_luxury <- df_tokens %>%
  filter(other_features %in% luxury_tokens) %>%
  count(id, name = "luxury_feature_count")

#EXPOSURE
df_exposure <- df %>%
  mutate(
    exposure_dirs = str_extract_all(other_features, "(north|south|east|west)"),
    exposure_north = map_int(exposure_dirs, ~as.integer("north" %in% .x)),
    exposure_south = map_int(exposure_dirs, ~as.integer("south" %in% .x)),
    exposure_east  = map_int(exposure_dirs, ~as.integer("east"  %in% .x)),
    exposure_west  = map_int(exposure_dirs, ~as.integer("west"  %in% .x)),
    exposure_double = as.integer(str_detect(other_features, "double exposure") | lengths(exposure_dirs) >= 2),
    exposure_internal = as.integer(str_detect(other_features, "internal exposure")),
    exposure_external = as.integer(str_detect(other_features, "external exposure"))
  ) %>%
  dplyr::select(id, starts_with("exposure_"))

#WINDOWS
df_windows <- df %>%
  mutate(
    has_double_glass_window = str_detect(other_features, "double glass") %>% as.integer(),
    has_triple_glass_window = str_detect(other_features, "triple glass") %>% as.integer()
  ) %>%
  dplyr::select(id, has_double_glass_window, has_triple_glass_window)


df_dummies <- base_id %>%
  left_join(df_dummies_selected, by = "id") %>%
  #left_join(df_misc_token, by = "id") %>%
  left_join(df_luxury, by = "id") %>%
  mutate(across(where(is.numeric), ~replace_na(., 0)))

df_final <- df %>%
  dplyr::select(-other_features) %>%
  right_join(df_dummies, by = "id") %>%
  right_join(df_exposure, by = "id") %>%
  right_join(df_windows, by = "id") %>%
  dplyr::select(-id)

skimr::skim(df_final)
dim(df_final)

data = df_final

# token_freq <- df_tokens %>%
#   count(other_features, sort = TRUE)
# 
# # Visualizza i primi token più frequenti
# print(token_freq, n=84)

#creo un altra variabile per capire a che posizione si trova l'appartamento nell'edificio
data$floor_ratio <- data$floor / data$total_floors_in_building
#creo age 
data$age = rep(2025,nrow(data)) - data$year_of_construction 
#data$age <- pmax(data$age, 0)



#var:square meters
table(data$square_meters,useNA = "always")
which(data$square_meters == 15)
data[c(1174,2501,2571,5854,7776),]
data[3037,]
sum(data$square_meters == 15)
data$square_meters[data$square_meters == 1]
#imposta a NA tutte le case con superficie < 15 mq
data$square_meters[data$square_meters < 14] <- NA
#imputa NA con la mediana dei metri quadrati per numero di camere
library(dplyr)
data <- data %>%
  group_by(rooms_number) %>%
  mutate(square_meters = ifelse(is.na(square_meters),
                                median(square_meters, na.rm = TRUE),
                                square_meters)) %>%
  ungroup()



# PULIZIA X  --------------------------------------------------------------

#rimozione variabili inutili 
skimr::skim(data)
x = data
x$ID = NULL
x$exposure_dirs = NULL
x$total_floors_in_building = NULL
#x$selling_price = log(x$selling_price)
x$square_meters = log(x$square_meters)
x$age = as.integer(x$age)
x$year_of_construction = NULL
x = as.data.frame(x)
head(x)
dim(x)

p = ncol(x)
skimr::skim(x) 

#faccio divisione in train e val
dim(x)
library(caret)
set.seed(123)
train_index <- createDataPartition(x$selling_price, p = 0.75, list = FALSE)
train_data <- x[train_index, ]
test_data <- x[-train_index, ]

plot(train_data$square_meters, log(train_data$selling_price),
     xlab = "Square Meters", ylab = "Log Sale Price", pch = 16, cex = 0.8
) 

plot(train_data$rooms_number, train_data$selling_price,
     xlab = "Total Basement SF", ylab = "Sale Price", pch = 16, cex = 0.8
)

str(train_data)


# MODELLI -----------------------------------------------------------------

names(train_data) <- make.names(names(train_data), unique = TRUE)
names(test_data) <- make.names(names(test_data), unique = TRUE)
head(x)
#LM CON QUALCHE VARIABILE (PROVO ZONA, METRI QUADRATI, NROOM,NBAGNI,FLOOR_RATIO)
head(train_data)
mod1 = lm(log(selling_price)~square_meters+bathrooms_number+rooms_number+condominium_fees+zone+floor_ratio+ pool+age, data = train_data )
summary(mod1)
#PROVO LM
lm_model = lm(log(selling_price)~., data = train_data)
summary(lm_model)
length(lm_model$coefficients)
head(x)
plot(lm_model)
#PROVO RF

rf_model<- ranger(log(selling_price) ~ ., data = train_data, num.trees = 2000, mtry = sqrt(p))

#PREDICTIONS 
preds3 <- predict(mod1, newdata = test_data)

preds2 <- predict(lm_model, newdata = test_data)

preds <- exp(predict(rf_model, data = test_data, type = "response")$predictions)

# #PROVO LASSO 

x$log_price <- log(x$selling_price)
X <- model.matrix(log_price ~ . -selling_price, data = x)[, -1]
y <- x$log_price

set.seed(123)
train_index <- createDataPartition(y, p = 0.75, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1, standardize = TRUE)
best_lambda <- cv_lasso$lambda.min
lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda, standardize = TRUE)
coef(lasso_model)
y_pred_log <- predict(lasso_model, s = best_lambda, newx = X_test)
y_pred <- exp(y_pred_log)
y_true <- exp(y_test)


# MAE ---------------------------------------------------------------------
#creo funzione per calcolo mae
MAE <- function(y, y_fit) {
  mean(abs(y - y_fit))
}

MAE(exp(preds3), test_data$selling_price)
MAE(exp(preds2), test_data$selling_price)
MAE(preds, test_data$selling_price)
MAE(y_pred, y_true)

y_hat_full <- exp(preds2)
y_hat_rf <- preds
y_hat_simple <- exp(preds3)
y_hat_lasso <- y_pred

n_test <- nrow(test_data)
final_summary <- data.frame(
  Predictions = c(y_hat_simple, y_hat_full, y_hat_rf, y_hat_lasso),
  Model = rep(c("Simple", "Full model","Random Forest", "Lasso"), each = n_test),
  Truth = test_data$selling_price
)
final_summary$Errors <- final_summary$Predictions - final_summary$Truth
tapply(final_summary$Errors, final_summary$Model, function(x) mean(abs(x)))




# SUBMISSION --------------------------------------------------------------
library(readr)

 submission <- data.frame(
   ID = data$ID,
   selling_price = exp(preds2)
 )
 write_csv(submission, "submission7.csv")

