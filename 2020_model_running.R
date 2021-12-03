# Written by Sreemanti Dey and R. Michael Alvarez.
# California Institute of Technology
# Nov. 17, 2021

# Cross-validation and figure production for "Fuzzy Forests For 
# Feature Selection in High-Dimensional Survey Data: An Application
# to the 2020 U.S. Presidential Election."

library(dplyr)
library(tidyr)
library(magrittr)
library(mixtools)
library(ggplot2)
library(vcd)
library(caret)
library(ROCR)

library(randomForest)
library(fuzzyforest)
library(WGCNA)


set.seed(1)

load("onehot_data.RData")

# -------------------------------------------------------------------- #

# fuzzy forest setup

# org mtry is 1
screen_params <- screen_control(
  drop_fraction = c(0.5, 0.25),
  keep_fraction = c(0.25),
  min_ntree = c(1000),
  ntree_factor = c(1),
  mtry_factor = c(5))

select_params <- select_control(
  drop_fraction = c(0.5, 0.25),
  number_selected = c(20),
  min_ntree = c(1000),
  ntree_factor = c(1),
  mtry_factor = c(10))


# wgcna scale free topology check
power_threshold <- pickSoftThreshold(subset(onehot_data, select=-c(CC20_410)))
windowsFonts(Times=windowsFont("Times New Roman"))
plot(x=power_threshold$fitIndices$Power,
     y=power_threshold$fitIndices$truncated.R.sq,
     family="Times",
     type="p",
     pch=19,
     main="Scale Free Topology Plot",
     xlab="Soft Threshold (Power)",
     ylab="Signed R^2",
     col="black")
grid(NULL, NULL)


# 10-fold cross validation for fuzzy forest

ff_accuracies = rep(NA, 10)
ff_models = list()
ff_rocrs = list()
ff_roc_curves = list()
ff_aucs = list()
ff_times = list()

# prepare modules
weight_frame <- data.frame(onehot_data$commonweight)
for (i in 2:441){
  weight_frame[,i] <- onehot_data$commonweight
}

wgcna_2020 <- blockwiseModules(datExpr=subset(onehot_data, 
                                              select=-c(CC20_410,
                                                        commonweight)),
                               weights=weight_frame,
                               power=2,
                               minModuleSize=5,
                               nThreads=0,
                               TOMType=c("unsigned"))

# Create 10 equally size folds
ff_folds <- cut(seq(1,nrow(onehot_data)),breaks=10,labels=FALSE)

# Perform 10 fold cross validation
for(i in 1:10){
  # Segment data
  testIndexes <- which(ff_folds==i, arr.ind=TRUE)
  testData <- onehot_data[testIndexes, ]
  trainData <- onehot_data[-testIndexes, ]

  # into ff
  ff_iteration_2020 <- ff(subset(trainData, select=-c(CC20_410,
                                                      commonweight)),
                      trainData$CC20_410,
                      module_membership = wgcna_2020$colors,
                      num_processors = 2,
                      screen_params = screen_params,
                      select_params = select_params,
                      final_ntree = c(1000),
                      nodesize = c(1))

  # raw accuracy
  ff_preds = predict(ff_iteration_2020, subset(testData, select=-c(CC20_410)))
  ff_accuracies[i] = sum(abs(ff_preds == testData$CC20_410)) / 4389
  print(ff_accuracies[i])

  # store model
  ff_models[[i]] = ff_iteration_2020


  ff_iteration_begin = proc.time()

  # roc and auc
  ff_iteration_finalrf_predict <- predict(ff_iteration_2020$final_rf,
                                          newdata = testData,
                                          type = "prob")
  ff_iteration_rocr <- ROCR::prediction(ff_iteration_finalrf_predict[,2],
                                        testData$CC20_410)
  ff_iteration_roc_curve <- ROCR::performance(ff_iteration_rocr,
                                              measure="tpr",
                                              x.measure="fpr")
  ff_iteration_auc <- performance(ff_iteration_rocr, "auc")


  ff_rocrs[[i]] = ff_iteration_rocr
  ff_roc_curves[[i]] = ff_iteration_roc_curve
  ff_aucs[[i]] = ff_iteration_auc

  # timing
  ff_times[[i]] = proc.time() - ff_iteration_begin
}

save(ff_accuracies, file="ff_cv_results.RData")
save(ff_roc_curves, file="ff_roc_curves.RData")
save(ff_aucs, file="ff_aucs.RData")
save(ff_times, file="ff_times.RData")

rm(ff_iteration_2020)
rm(ff_preds)
rm(ff_iteration_begin)
rm(ff_iteration_finalrf_predict)
rm(ff_iteration_rocr)
rm(ff_iteration_roc_curve)
rm(ff_iteration_auc)


# variable importance plot for ff

cces_label <- data.frame(rep(NA, 20))
cces_label$variable <- ff_models[[1]]$feature_list$feature_name
cces_label <- within(cces_label, rm(rep.NA..20.))
cces_label$label <- rep(NA, 20)

cces_label$label[1] = "Consider myself a Democrat"
cces_label$label[2] = "RBG should be replaced after new govt."
cces_label$label[3] = "Consider myself a Republican"
cces_label$label[4] = "Voted for Clinton in 2016"
cces_label$label[5] = "Support Amy Coney Barrett as SC Justice"
cces_label$label[6] = "Voted for Trump in 2016"
cces_label$label[7] = "RBG should be replaced before new govt."
cces_label$label[8] = "Support natl. emergency to build Mexico wall"
cces_label$label[9] = "Registered Republican"
cces_label$label[10] = "Oppose Amy Coney Barrett as SC justice"
cces_label$label[11] = "*See Below"
cces_label$label[12] = "Support withdrawing from Iran Nuclear Accord"
cces_label$label[13] = "Support repealing the entire ACA"
cces_label$label[14] = "**See Below"
cces_label$label[15] = "Registered Democrat"
cces_label$label[16] = "Oppose natl. emergency to build Mexico wall"
cces_label$label[17] = "Scale of spending cuts to tax increase in your state"
cces_label$label[18] = "***See Below"
cces_label$label[19] = "Birth year"
cces_label$label[20] = "Oppose repealing the ACA"

# *Strongly agree: White people in the U.S. have certain advantages because of 
# the color of their skin.

# **Strongly agree: Generations of slavery and discrimination have
# created conditions that make it difficult for Blacks to work their
# way out of the lower class.

# ***Strongly disagree: Generations of slavery and discrimination
# have created conditions that make it difficult for Blacks to work
# their way out of the lower class.


# 184-204 Adapted from: https://github.com/sysilviakim/turnout2016/blob/master/fuzzy_forest_utility.R

windowsFonts(Times=windowsFont("Times New Roman"))
varImpPlot <- ff_models[[1]]$feature_list %>%
  dplyr::mutate(
    variable_importance = variable_importance
  ) %>%
  dplyr::select(-module_membership) %>%
  rowwise() %>%
  dplyr::left_join(., cces_label, by = c("feature_name" = "variable")) %>%
  ggplot() +
  aes(
    x = variable_importance,
    y = reorder(label, variable_importance)
  ) +
  geom_point() +
  ylab("Variables") +
  xlab(paste0("Variable Importance",
              "\nMean Decrease in Accuracy")) +
  theme_bw() +
  theme(text=element_text(family="Times", size=12))

print(varImpPlot)


# -------------------------------------------------------------------- #


# 10-fold cross validation for random forest

rf_accuracies = rep(NA, 10)
rf_rocrs = list()
rf_roc_curves = list()
rf_aucs = list()
rf_times = list()

#Create 10 equally size folds
rf_folds <- cut(seq(1,nrow(onehot_data)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
for(i in 1:10){
  # Segment data
  testIndexes <- which(rf_folds==i, arr.ind=TRUE)
  testData <- onehot_data[testIndexes, ]
  trainData <- onehot_data[-testIndexes, ]

  # into rf
  rf_iteration_2020 <- randomForest(formula=CC20_410 ~ . , data=trainData)

  # raw accuracy
  rf_preds <- as.vector(predict(rf_iteration_2020,
                                newdata=testData,
                                type="class"))
  rf_accuracies[i] = sum(abs(rf_preds == testData$CC20_410)) / 4389
  print(rf_accuracies[i])


  rf_iteration_begin = proc.time()

  # roc and auc
  rf_iteration_predict <- predict(rf_iteration_2020,
                                  newdata = testData,
                                  type = "prob")
  rf_iteration_rocr <- ROCR::prediction(rf_iteration_predict[,2],
                                        testData$CC20_410)
  rf_iteration_roc_curve <- ROCR::performance(rf_iteration_rocr,
                                              measure="tpr",
                                              x.measure="fpr")
  rf_iteration_auc <- performance(rf_iteration_rocr, "auc")


  rf_rocrs[[i]] = rf_iteration_rocr
  rf_roc_curves[[i]] = rf_iteration_roc_curve
  rf_aucs[[i]] = rf_iteration_auc

  # timing
  rf_times[[i]] = proc.time() - rf_iteration_begin
}

save(rf_accuracies, file="rf_cv_results.RData")
save(rf_roc_curves, file="rf_roc_curves.RData")
save(rf_aucs, file="rf_aucs.RData")
save(rf_times, file="rf_times.RData")

rm(rf_iteration_2020)
rm(rf_preds)
rm(rf_iteration_begin)
rm(rf_iteration_predict)
rm(rf_iteration_rocr)
rm(rf_iteration_roc_curve)
rm(rf_iteration_auc)


# -------------------------------------------------------------------- #


# 10-fold cross validation for logit

glm_accuracies = rep(NA, 10)
glm_rocrs = list()
glm_roc_curves = list()
glm_aucs = list()

# Create 10 equally size folds
glm_folds <- cut(seq(1,nrow(onehot_data)),breaks=10,labels=FALSE)

# Perform 10 fold cross validation
for(i in 1:10){
  # Segment data
  testIndexes <- which(glm_folds==i, arr.ind=TRUE)

  testData <- onehot_data[testIndexes, ]
  testData <- testData %>% mutate(CC20_410 = ifelse(CC20_410 == 2, 1, 0))

  trainData <- onehot_data[-testIndexes, ]
  trainData <- trainData %>% mutate(CC20_410 = ifelse(CC20_410 == 2, 1, 0))

  # into glm
  glm_iteration_2020 <- glm(CC20_410 ~ .,
                            family=binomial(link='logit'),
                            data=trainData)

  # predictions
  glm_preds <- predict(glm_iteration_2020, testData, type = "response")
  glm_preds <- ifelse(glm_preds > 0.5, 1, 0)
  glm_accuracies[i] = sum(glm_preds == testData$CC20_410) / 4389
  print(glm_accuracies[i])

  # roc and auc
  glm_iteration_predict <- predict(glm_iteration_2020,
                                   newdata = testData,
                                   type = "response")
  glm_iteration_rocr <- ROCR::prediction(glm_iteration_predict,
                                         testData$CC20_410)
  glm_iteration_roc_curve <- ROCR::performance(glm_iteration_rocr,
                                               measure="tpr",
                                               x.measure="fpr")
  glm_iteration_auc <- performance(glm_iteration_rocr, "auc")


  glm_rocrs[[i]] = glm_iteration_rocr
  glm_roc_curves[[i]] = glm_iteration_roc_curve
  glm_aucs[[i]] = glm_iteration_auc
}

save(glm_accuracies, file="glm_cv_results.RData")
save(glm_roc_curves, file="glm_roc_curves.RData")
save(glm_aucs, file="glm_aucs.RData")

rm(glm_iteration_2020)
rm(glm_preds)
rm(glm_iteration_predict)
rm(glm_iteration_rocr)
rm(glm_iteration_roc_curve)
rm(glm_iteration_auc)


# -------------------------------------------------------------------- #


# roc/auc comparison plot

plot(ff_roc_curves[[9]], main="ROC and AUC Comparison")
plot(rf_roc_curves[[9]], add=TRUE, col=3)
plot(glm_roc_curves[[9]], add=TRUE, col=4)
legend("bottomright",
       legend=c("Fuzzy Forest (0.9869)",
                "Random Forest (0.9919)",
                "Logit (0.9921)"),
        col=c("black", "green", "blue"), lty=1:1.5, cex=0.6, box.lty=0)



counterTestData <- onehot_data[39501:43890,]
counterTestData$CC20_356a.2[counterTestData$CC20_356a.2 == 0] <- 2
counterTestData$CC20_356a.2[counterTestData$CC20_356a.2 == 1] <- 0
counterTestData$CC20_356a.2[counterTestData$CC20_356a.2 == 2] <- 1
counter_predict <- as.data.frame(predict(ff_models[[1]]$final_rf,
                                        newdata = counterTestData,
                                        type = "prob"))
counter_predict <- counter_predict[,1]
counter_predict[counter_predict >= 0.5] <- 1
counter_predict[counter_predict < 0.5] <- 2
