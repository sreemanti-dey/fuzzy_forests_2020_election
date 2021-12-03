# Written by Sreemanti Dey and R. Michael Alvarez.
# California Institute of Technology
# Nov. 17, 2021

# Data preprocessing for "Fuzzy Forests For Feature Selection
# in High-Dimensional Survey Data: An Application to the 2020 U.S.
# Presidential Election."

# remove unused libraries
library(dplyr)
library(tidyr)
library(magrittr)
library(mixtools)
library(ggplot2)
library(Amelia)
library(caret)
library(simputation)

set.seed(1)

col_names_2020 = c(
  "commonweight", "inputstate", "gender", "birthyr",
  "educ", "race", "CC20_401", "CC20_403", "CC20_410",
  "CC20_411_GA1", "CC20_420_1", "CC20_420_2", "CC20_420_3",
  "CC20_420_4", "CC20_420_5", "CC20_420_6", "CC20_420_7",
  "CC20_421r", "CC20_422r", "CC20_430a_1", "CC20_430a_2",
  "CC20_430a_3", "CC20_430a_4", "CC20_430a_5", "CC20_430a_6",
  "CC20_430a_7", "CC20_430a_8", "CC20_431a", "CC20_432a",
  "CC20_433a", "CC20_440a", "CC20_440b", "CC20_440c", "CC20_440d",
  "CC20_441a", "CC20_441b", "CC20_442a", "CC20_442b", "CC20_442c",
  "CC20_442d", "CC20_442e", "CC20_443_1", "CC20_443_2",
  "CC20_443_3", "CC20_443_4", "CC20_443_5", "employ", "gunown",
  "numchildren", "edloan", "CC20_300_1", "CC20_300_2",
  "CC20_300_3", "CC20_300_4", "CC20_300_5", "CC20_302",
  "CC20_303", "CC20_305_1", "CC20_305_2", "CC20_305_3",
  "CC20_305_4", "CC20_305_5", "CC20_305_6", "CC20_305_7",
  "CC20_305_9", "CC20_305_10", "CC20_305_11", "CC20_305_12",
  "CC20_305_13", "CC20_307", "CC20_309a_1", "CC20_309a_2",
  "CC20_309a_3", "CC20_309a_4", "CC20_309a_5", "CC20_309c_1",
  "CC20_309c_2", "CC20_309c_3", "CC20_309c_4", "CC20_309c_5",
  "CC20_309c_6", "CC20_309c_7", "CC20_309c_8", "CC20_309c_9",
  "CC20_309c_10", "CC20_309e", "CC20_327a", "CC20_327b",
  "CC20_327c", "CC20_327d", "CC20_327e", "CC20_327f", "CC20_330a",
  "CC20_330b", "CC20_330c", "CC20_356a", "CC20_356", "CC20_360",
  "CC20_361", "urbancity", "CC20_363", "ideo5", "pew_religimp",
  "pew_churatd", "religpew", "marstat", "cit1", "immstat",
  "dualcit", "ownhome", "newsint", "faminc_new", "union",
  "investor", "internethome", "presvote16post", "sexuality",
  "CC20_332a", "CC20_332a", "CC20_332b", "CC20_332c", "CC20_332d",
  "CC20_332e", "CC20_332f", "CC20_332g"
)

# numerics: commonweight, birthyr, cc20_421r, cc20_422r

# making the raw dataset

cces_2020_full <- read.csv(file="CCES20_Common_OUTPUT.csv")

garbage = 1:61000
cces_2020_raw <- data.frame(garbage)

for (i in 1:length(col_names_2020)){
  if (col_names_2020[i] %in% names(cces_2020_full)){
    cces_2020_raw[,col_names_2020[i]] <- cces_2020_full[,col_names_2020[i]]
  }
}

cces_2020_raw <- within(cces_2020_raw, rm(garbage))

rm(garbage)
rm(cces_2020_full)


# pmm hot deck imputation


# factorizing NAs in the Georgia-specific question
cces_2020_raw$CC20_411_GA1[is.na(cces_2020_raw$CC20_411_GA1)] <- 97

# refactorizing religpew and faminc_new
cces_2020_raw$religpew <- as.factor(as.character(cces_2020_raw$religpew))
cces_2020_raw$faminc_new <- as.factor(as.character(cces_2020_raw$faminc_new))

# religpew combined levels 4, 6, 7, 8 into level 97
levels(cces_2020_raw$religpew) <- c(levels(cces_2020_raw$religpew), 97)
cces_2020_raw$religpew[cces_2020_raw$religpew == 4] <- 97
cces_2020_raw$religpew[cces_2020_raw$religpew == 6] <- 97
cces_2020_raw$religpew[cces_2020_raw$religpew == 7] <- 97
cces_2020_raw$religpew[cces_2020_raw$religpew == 8] <- 97
cces_2020_raw$religpew <- factor(cces_2020_raw$religpew)

# faminc_new combined levels merge 2 successive categories in
# the original; so >10k and 10k-19k are combined, etc.
cces_2020_raw$faminc_new[cces_2020_raw$faminc_new == 2] <- 1
cces_2020_raw$faminc_new[cces_2020_raw$faminc_new == 4] <- 3
cces_2020_raw$faminc_new[cces_2020_raw$faminc_new == 6] <- 5
cces_2020_raw$faminc_new[cces_2020_raw$faminc_new == 8] <- 7
cces_2020_raw$faminc_new[cces_2020_raw$faminc_new == 10] <- 9
cces_2020_raw$faminc_new[cces_2020_raw$faminc_new == 12] <- 11
cces_2020_raw$faminc_new[cces_2020_raw$faminc_new == 14] <- 13
cces_2020_raw$faminc_new[cces_2020_raw$faminc_new == 16] <- 15
cces_2020_raw$faminc_new <- factor(cces_2020_raw$faminc_new)

# adding same-day registration variable
cces_2020_raw <- mutate(cces_2020_raw, same_day = inputstate)

state_data_2020 <- read.csv(file='state_data_2020.csv', check.names=FALSE)
for (i in 1:61000){
  cces_2020_raw[i, "same_day"] <- state_data_2020[1, as.character(cces_2020_raw[i, "inputstate"])]
}

cces_2020_raw$same_day = as.factor(cces_2020_raw$same_day)

# doing hot deck imputation
hotdeck_imputation <- impute_pmm(subset(cces_2020_raw, select=-c(CC20_410)),
                                 . ~ .,
                                 predictor = impute_cart,
                                 pool = c("complete",
                                          "univariate",
                                          "multivariate"))

hotdeck_imputation$CC20_410 <- cces_2020_raw$CC20_410
hotdeck_imputation$CC20_410 <- as.numeric(as.character(hotdeck_imputation$CC20_410))

candidate_choice_data <- hotdeck_imputation[hotdeck_imputation$CC20_410 == 1 | hotdeck_imputation$CC20_410 == 2,]
candidate_choice_data$CC20_410 = factor(candidate_choice_data$CC20_410)
candidate_choice_data <- na.omit(candidate_choice_data)

# candidate choice data should have all cols numeric
candidate_choice_data$religpew = as.numeric(as.character(candidate_choice_data$religpew))
candidate_choice_data$faminc_new = as.numeric(as.character(candidate_choice_data$faminc_new))
candidate_choice_data$same_day = as.numeric(as.character(candidate_choice_data$same_day))


# onehot encoding

pre_onehot_data <- subset(candidate_choice_data, select=-c(CC20_410))
pre_onehot_data <- data.frame(sapply(pre_onehot_data, as.character))
pre_onehot_data <- pre_onehot_data %>% mutate_if(is.character, as.factor)

pre_onehot_data$commonweight <- as.numeric(pre_onehot_data$commonweight)
pre_onehot_data$birthyr <- as.numeric(pre_onehot_data$birthyr)
pre_onehot_data$CC20_421r <- as.numeric(pre_onehot_data$CC20_421r)
pre_onehot_data$CC20_422r <- as.numeric(pre_onehot_data$CC20_422r)


dmy = dummyVars(" ~ .", data = pre_onehot_data)
onehot_data <- data.frame(predict(dmy, newdata = pre_onehot_data))
onehot_data <- onehot_data %>% mutate_if(is.factor, as.character)
onehot_data <- onehot_data %>% mutate_if(is.character, as.numeric)
onehot_data$CC20_410 <- candidate_choice_data$CC20_410
save(onehot_data, file="onehot_data.RData")

rm(cces_2020_raw)
rm(candidate_choice_data)
rm(dmy)
rm(hotdeck_imputation)
rm(pre_onehot_data)
rm(state_data_2020)
